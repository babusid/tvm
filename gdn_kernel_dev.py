import tvm
from tvm import DataType, te, tir, s_tir, topi
from tvm.script import tir as T
import math


def _gen_causal_conv1d_update(
    hidden_size: int,
    state_len: int,
    dtype: str = "float32",
    activation: str = "silu",
    has_bias: bool = True,
):
    """
    Generate a TIR primfunc for causal 1D convolution update.

    Uses closure pattern to capture constants (hidden_size, state_len, etc.) at build time.
    Uses TVMScript decorator syntax for cleaner, more readable TIR code.
    Returns a tir.PrimFunc that can be compiled and executed.
    
    Note: Two separate primfuncs are defined (with and without bias) to ensure
    the bias buffer is only present in the signature when has_bias=True.
    """
    # Capture constants in closure
    _hidden_size = hidden_size
    _state_len = state_len
    _dtype = dtype
    _activation = activation

    @T.prim_func
    def causal_conv1d_update_with_bias(
        hidden_states: T.handle,
        conv_state: T.handle,
        weight: T.Buffer((T.int64(_hidden_size), T.int64(1), T.int64(_state_len)), _dtype),
        bias: T.Buffer((T.int64(_hidden_size),), _dtype),
        next_conv_state: T.handle,
        out: T.handle,
    ):
        T.func_attr({"global_symbol": "causal_conv1d_update", "tir.noalias": True})

        # Dynamic symbolic variables extracted from handles
        batch_size = T.int64()
        seq_len = T.int64()

        # Match buffers with dynamic shapes
        hidden_states_buf = T.match_buffer(
            hidden_states, (batch_size, T.int64(_hidden_size), seq_len), dtype=_dtype
        )
        conv_state_buf = T.match_buffer(
            conv_state, (batch_size, T.int64(_hidden_size), T.int64(_state_len)), dtype=_dtype
        )
        next_conv_state_buf = T.match_buffer(
            next_conv_state, (batch_size, T.int64(_hidden_size), T.int64(_state_len)), dtype=_dtype
        )
        out_buf = T.match_buffer(out, (batch_size, T.int64(_hidden_size), seq_len), dtype=_dtype)

        # Allocate intermediate buffers
        hidden_states_new = T.alloc_buffer(
            (batch_size, T.int64(_hidden_size), T.int64(_state_len) + seq_len), dtype=_dtype
        )
        conv_sum = T.alloc_buffer((batch_size, T.int64(_hidden_size), seq_len), dtype=_dtype)

        # Step 1: Concatenate conv_state and hidden_states along last dimension
        for b, h, l in T.grid(batch_size, T.int64(_hidden_size), T.int64(_state_len) + seq_len):
            if l < T.int64(_state_len):
                hidden_states_new[b, h, l] = conv_state_buf[b, h, l]
            else:
                hidden_states_new[b, h, l] = hidden_states_buf[b, h, l - T.int64(_state_len)]

        # Step 2: Extract last state_len elements as next_conv_state
        for b, h, s in T.grid(batch_size, T.int64(_hidden_size), T.int64(_state_len)):
            next_conv_state_buf[b, h, s] = hidden_states_new[b, h, s + seq_len]

        # Step 3: Initialize conv_sum to 0
        for b, h, s in T.grid(batch_size, T.int64(_hidden_size), seq_len):
            conv_sum[b, h, s] = T.cast(T.float32(0.0), _dtype)

        # Step 4: Compute causal convolution (sliding window)
        # For each position s in output, convolve with state_len window ending at s+state_len
        for b, h, s, kw in T.grid(batch_size, T.int64(_hidden_size), seq_len, T.int64(_state_len)):
            conv_sum[b, h, s] = (
                conv_sum[b, h, s]
                + hidden_states_new[b, h, s + kw + T.int64(1)] * weight[h, T.int64(0), kw]
            )

        # Step 5: Add bias and apply activation
        for b, h, s in T.grid(batch_size, T.int64(_hidden_size), seq_len):
            # Add bias
            biased_val = conv_sum[b, h, s] + bias[h]

            # Apply activation if enabled (closure constant)
            if _activation == "silu":
                # SiLU(x) = x * sigmoid(x)
                val_float = T.cast(biased_val, "float32")
                out_buf[b, h, s] = T.cast(val_float * T.sigmoid(val_float), _dtype)
            else:
                out_buf[b, h, s] = biased_val

       
    @T.prim_func
    def causal_conv1d_update_no_bias(
        hidden_states: T.handle,
        conv_state: T.handle,
        weight: T.Buffer((T.int64(_hidden_size), T.int64(1), T.int64(_state_len)), _dtype),
        next_conv_state: T.handle,
        out: T.handle,
    ):
        T.func_attr({"global_symbol": "causal_conv1d_update", "tir.noalias": True})

        # Dynamic symbolic variables extracted from handles
        batch_size = T.int64()
        seq_len = T.int64()

        # Match buffers with dynamic shapes
        hidden_states_buf = T.match_buffer(
            hidden_states, (batch_size, T.int64(_hidden_size), seq_len), dtype=_dtype
        )
        conv_state_buf = T.match_buffer(
            conv_state, (batch_size, T.int64(_hidden_size), T.int64(_state_len)), dtype=_dtype
        )
        next_conv_state_buf = T.match_buffer(
            next_conv_state, (batch_size, T.int64(_hidden_size), T.int64(_state_len)), dtype=_dtype
        )
        out_buf = T.match_buffer(out, (batch_size, T.int64(_hidden_size), seq_len), dtype=_dtype)

        # Allocate intermediate buffers
        hidden_states_new = T.alloc_buffer(
            (batch_size, T.int64(_hidden_size), T.int64(_state_len) + seq_len), dtype=_dtype
        )
        conv_sum = T.alloc_buffer((batch_size, T.int64(_hidden_size), seq_len), dtype=_dtype)

        # Step 1: Concatenate conv_state and hidden_states along last dimension
        for b, h, l in T.grid(batch_size, T.int64(_hidden_size), T.int64(_state_len) + seq_len):
            if l < T.int64(_state_len):
                hidden_states_new[b, h, l] = conv_state_buf[b, h, l]
            else:
                hidden_states_new[b, h, l] = hidden_states_buf[b, h, l - T.int64(_state_len)]

        # Step 2: Extract last state_len elements as next_conv_state
        for b, h, s in T.grid(batch_size, T.int64(_hidden_size), T.int64(_state_len)):
            next_conv_state_buf[b, h, s] = hidden_states_new[b, h, s + seq_len]

        # Step 3: Initialize conv_sum to 0
        for b, h, s in T.grid(batch_size, T.int64(_hidden_size), seq_len):
            conv_sum[b, h, s] = T.cast(T.float32(0.0), _dtype)

        # Step 4: Compute causal convolution (sliding window)
        # For each position s in output, convolve with state_len window ending at s+state_len
        for b, h, s, kw in T.grid(batch_size, T.int64(_hidden_size), seq_len, T.int64(_state_len)):
            conv_sum[b, h, s] = (
                conv_sum[b, h, s]
                + hidden_states_new[b, h, s + kw + T.int64(1)] * weight[h, T.int64(0), kw]
            )

        # Step 5: Apply activation (no bias)
        for b, h, s in T.grid(batch_size, T.int64(_hidden_size), seq_len):
            val = conv_sum[b, h, s]

            # Apply activation if enabled (closure constant)
            if _activation == "silu":
                # SiLU(x) = x * sigmoid(x)
                val_float = T.cast(val, "float32")
                out_buf[b, h, s] = T.cast(val_float * T.sigmoid(val_float), _dtype)
            else:
                out_buf[b, h, s] = val

    if has_bias:
        return causal_conv1d_update_with_bias

    return causal_conv1d_update_no_bias


def _chunk_gated_delta_rule(
    linear_num_value_heads: int,
    linear_value_head_dim: int,
    linear_num_key_heads: int,
    linear_key_head_dim: int,
    chunk_size: int = 64,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    dtype: str = "float32",
):
    _linear_num_value_heads = tir.IntImm("int64", linear_num_value_heads)
    _linear_value_head_dim = tir.IntImm("int64", linear_value_head_dim)
    _linear_num_key_heads = tir.IntImm("int64", linear_num_key_heads)
    _linear_key_head_dim = tir.IntImm("int64", linear_key_head_dim)
    _chunk_size = tir.IntImm("int64", chunk_size)
    _seq_len = tir.Var("seq_len", dtype="int64")  # changes at runtime
    _batch_size = tir.Var("batch_size", dtype="int64")  # changes at runtime
    _dtype = DataType(dtype)

    inputs = []

    query: te.Tensor = te.placeholder(
        shape=(_batch_size, _seq_len, _linear_num_value_heads, _linear_key_head_dim),
        dtype=_dtype,
        name="query",
    )

    key: te.Tensor = te.placeholder(
        shape=(_batch_size, _seq_len, _linear_num_value_heads, _linear_key_head_dim),
        dtype=_dtype,
        name="key",
    )

    value: te.Tensor = te.placeholder(
        shape=(_batch_size, _seq_len, _linear_num_value_heads, _linear_value_head_dim),
        dtype=_dtype,
        name="value",
    )

    g: te.Tensor = te.placeholder((_batch_size, _seq_len, _linear_num_value_heads))

    beta: te.Tensor = te.placeholder(
        (_batch_size, _seq_len, _linear_num_value_heads), dtype=_dtype, name="beta"
    )

    inputs.extend([query, key, value, g, beta])
    if use_qk_l2norm_in_kernel:
        # do l2 norm on query and key here
        query_sq = te.compute(
            query.shape, lambda b, s, n, d: query[b, s, n, d] * query[b, s, n, d], name="query_sq"
        )
        key_sq = te.compute(
            key.shape, lambda b, s, n, d: key[b, s, n, d] * key[b, s, n, d], name="key_sq"
        )

        rv = te.reduce_axis((0, _linear_key_head_dim), name="rv")
        query_sum = te.compute(
            (_batch_size, _seq_len, _linear_num_value_heads, 1),
            lambda i, j, k, l: te.sum(query_sq[i, j, k, rv], axis=rv),
            name="query_sum",
        )
        key_sum = te.compute(
            (_batch_size, _seq_len, _linear_num_value_heads, 1),
            lambda i, j, k, l: te.sum(key_sq[i, j, k, rv], axis=rv),
            name="key_sum",
        )
        eps = tir.FloatImm("float32", 1e-6)  # hardcode for now
        q_norm = te.compute(
            (_batch_size, _seq_len, _linear_num_value_heads, _linear_key_head_dim),
            lambda i, j, k, l: query[i, j, k, l] * te.rsqrt(query_sum[i, j, k, 0] + eps),
            "q_norm",
        )
        k_norm = te.compute(
            (_batch_size, _seq_len, _linear_num_value_heads, _linear_key_head_dim),
            lambda i, j, k, l: key[i, j, k, l] * te.rsqrt(key_sum[i, j, k, 0] + eps),
            "k_norm",
        )
        # set query and key to l2normed versions
        query = q_norm
        key = k_norm

    transaxes = [0, 2, 1, 3]
    query = topi.transpose(query, transaxes)
    key = topi.transpose(key, transaxes)
    value = topi.transpose(value, transaxes)

    transaxes = [0, 2, 1]
    beta = topi.transpose(beta, transaxes)
    g = topi.transpose(g, transaxes)

    pad_size = (_chunk_size - _seq_len % _chunk_size) % _chunk_size
    query, key, value = tuple(
        topi.nn.pad(x, pad_before=[0, 0, 0, 0], pad_after=[0, 0, pad_size, 0], pad_value=0.0)
        for x in (query, key, value)
    )
    beta, g = tuple(
        topi.nn.pad(x, pad_before=[0, 0, 0], pad_after=[0, 0, pad_size], pad_value=0.0)
        for x in (beta, g)
    ) 
    
    total_seq_len = _seq_len + pad_size
    num_chunks = tir.ceildiv(total_seq_len, _chunk_size)  # Explicitly compute number of chunks
    
    # scale = tir.rsqrt(tir.Cast("float33", query.shape[-1]))
    scale = tir.FloatImm("float32", 1 / math.sqrt(linear_key_head_dim))  # resolve at compile
    query *= scale

    v_beta = value * topi.expand_dims(beta, axis=-1)
    k_beta = key * topi.expand_dims(beta, axis=-1)

    query = topi.reshape(
        query, (_batch_size, _linear_num_value_heads, num_chunks, _chunk_size, query.shape[-1])
    )
    key = topi.reshape(key, (_batch_size, _linear_num_value_heads, num_chunks, _chunk_size, key.shape[-1]))
    value = topi.reshape(
        value, (_batch_size, _linear_num_value_heads, num_chunks, _chunk_size, value.shape[-1])
    )
    k_beta = topi.reshape(
        k_beta, (_batch_size, _linear_num_value_heads, num_chunks, _chunk_size, k_beta.shape[-1])
    )
    v_beta = topi.reshape(
        v_beta, (_batch_size, _linear_num_value_heads, num_chunks, _chunk_size, v_beta.shape[-1])
    )
    g = topi.reshape(g, (_batch_size, _linear_num_value_heads, num_chunks, _chunk_size))

    mask = topi.trilu(
        topi.full((_chunk_size, _chunk_size), dtype="int32", fill_value=1),
        k=tir.IntImm("int32", 0),
        upper=True,
    )
    g = topi.cumsum(g, axis=-1)
    g_col = topi.expand_dims(g, axis=-1)
    g_row = topi.expand_dims(g, axis=-2)
    decay_mask = topi.trilu(
        (topi.exp(topi.trilu(topi.subtract(g_col, g_row), k=tir.IntImm("int32", 0), upper=False))),
        k=tir.IntImm("int32", 0),
        upper=False,
    )

    # batched matmul, transposed key
    attn_mm_reduce_k = te.reduce_axis((0, _linear_key_head_dim), name="attm_mm_reduce_k")
    attn = te.compute(
        (k_beta.shape[0], k_beta.shape[1], k_beta.shape[2], _chunk_size, _chunk_size),
        lambda b, h, c, i, j: te.sum(
            k_beta[b, h, c, i, attn_mm_reduce_k] * key[b, h, c, j, attn_mm_reduce_k],
            axis=attn_mm_reduce_k,
        ),
        name="attn",
    )
    attn *= decay_mask
    attn *= topi.subtract(topi.full_like(mask, 1), topi.cast(mask, "float32"))
    attn = topi.negative(attn)
    
    def gen_prefix_scan_ir(attn_in, attn_out):
        from tvm.script.ir_builder import IRBuilder
        from tvm.script.ir_builder import tir as T
        
        attn_buf = T.buffer_proxy(attn_in)
        out_buf = T.buffer_proxy(attn_out)
        
        with IRBuilder() as ib:
            with T.seq_scope():
                with T.serial(0, _batch_size) as b:
                    with T.serial(0, _linear_num_value_heads) as h:
                        with T.serial(0, num_chunks) as c:
                            # Sequential inner loops with symbolic bounds
                            with T.serial(1, _chunk_size) as i:
                                # j from 0 to i (lower triangle, has to do the prefix)
                                with T.serial(0, i) as j:
                                    acc = T.float32(0.0)
                                    # k from j+1 to i (inner product)
                                    with T.serial(j+1, i) as k:
                                        rowvalue = attn_buf[b, h, c, i, k] # grab kth value of ith row
                                        colvalue = attn_buf[b, h, c, k, j] # grab kth row of jth column, j < i
                                        acc += rowvalue * colvalue # accumulate a rowsum of the dot product
                                    out_buf[b, h, c, i, j] = attn_buf[b, h, c, i, j] + acc # store the rowsum plus original
                                # Upper triangle copy: j from 0 to chunk_size-i
                                with T.serial(0, _chunk_size - i) as j:
                                    out_buf[b, h, c, i, i + j] = attn_buf[b, h, c, i, i + j]
        
        return ib.get()
    
    attn = te.extern(
        attn.shape,
        [attn],
        lambda ins, outs: gen_prefix_scan_ir(ins[0], outs[0]),
        name="prefix_scan_attn_update",
        dtype=dtype,
    )


    outputs = [query, key, value, mask, g, decay_mask, attn]
    basefunc: tir.PrimFunc = te.create_prim_func(inputs + outputs)
    return basefunc
    # TODO: once these functions actually work, experiment with loop fusions
    # etc. to make the kernels performant. Can also reorder ops and inline things
    # into more compound te.compute blocks. May have to switch to hand-written
    # tir to really optimize it.


def _chunk_recurrent_gated_delta_rule():
    inputs = []
    outputs = []
    return te.create_prim_func(inputs + outputs)


print(_gen_causal_conv1d_update(128, 128).script())
print(_chunk_gated_delta_rule(32, 128, 16, 128, use_qk_l2norm_in_kernel=False).script())
print(_chunk_gated_delta_rule(32, 128, 16, 128, use_qk_l2norm_in_kernel=True).script())
