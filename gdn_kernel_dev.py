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
    _seq_len = tir.Var(
        "seq_len", dtype="int64"
    )  # changes at runtime depending on prefill or decode
    _batch_size = tir.Var("batch_size", dtype="int64")  # changes at runtime
    _hidden_size = tir.IntImm(dtype="int64", value=hidden_size)  # from config
    _state_len = tir.IntImm(dtype="int64", value=state_len)  # from config
    _dtype = DataType(dtype)  # static
    hidden_states = te.placeholder(
        (_batch_size, _hidden_size, _seq_len), dtype=_dtype, name="hidden_states"
    )
    conv_state = te.placeholder(
        (_batch_size, _hidden_size, _state_len), dtype=_dtype, name="conv_state"
    )
    weight = te.placeholder((_hidden_size, 1, _state_len), dtype=_dtype, name="weight")
    bias = None
    if has_bias:
        bias = te.placeholder((_hidden_size,), dtype=_dtype, name="bias")

    # Concatenation across the last dim
    total_len = _state_len + _seq_len
    hidden_states_new = te.compute(
        (_batch_size, _hidden_size, total_len),
        lambda b, h, l: te.if_then_else(
            l < _state_len, conv_state[b, h, l], hidden_states[b, h, l - _state_len]
        ),
        name="hidden_states_new",
    )

    # Create new conv state only last state_len on last dim
    next_conv_state = te.compute(
        (_batch_size, _hidden_size, _state_len),
        lambda b, h, s: hidden_states_new[b, h, s + _seq_len],
        name="next_conv_state",
    )

    # Compute the causal convolution across the last dim of new_hidden_states
    kw = te.reduce_axis((0, _state_len), name="kw")
    # conv_sum last dim is iterating across the input sequence (_seq_len)
    # cs[b,h,0] looks at hidden_states_new[b,h,0] -> hidden_states_new[b,h, state_len]
    # then, the next token slides the window one to the right, so it uses
    # hidden_states_new[b,h,1] -> hidden_states_new[b,h,state_len+1]
    # which includes token 0's input hidden state, and excludes the oldest cached hidden state
    conv_sum = te.compute(
        (_batch_size, _hidden_size, _seq_len),
        lambda b, h, s: te.sum(hidden_states_new[b, h, s + kw + 1] * weight[h, 0, kw], axis=kw),
        name="conv_sum",
    )

    def bias_and_act_logic(b, h, s):
        val = conv_sum[b, h, s]
        if has_bias:
            val += bias[h]
        if activation == "silu":
            val = val * te.sigmoid(val)
        return val

    out = te.compute((_batch_size, _hidden_size, _seq_len), bias_and_act_logic, name="out")

    # actually create primfunc
    inputs = [hidden_states, conv_state, weight]
    if has_bias:
        inputs.append(bias)

    # return both, next_conv needed for cache update
    outputs = [next_conv_state, out]

    return te.create_prim_func(inputs + outputs)


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
                # Parallelize over leading dimensions 
                with T.parallel(0, _batch_size) as b:
                    with T.parallel(0, _linear_num_value_heads) as h:
                        with T.parallel(0, num_chunks) as c:
                            # inner loop has to be sequential
                            # for some reason python loop works but
                            # T.serial is failing
                            for i in range(1, chunk_size): 
                                for j in range(i):
                                    acc = T.float32(0.0)
                                    for k in range(j + 1, i):
                                        rowvalue = attn_buf[b, h, c, i, k]
                                        colvalue = attn_buf[b, h, c, k, j]
                                        acc += rowvalue * colvalue
                                    out_buf[b, h, c, i, j] = attn_buf[b, h, c, i, j] + acc
                                for j in range(int(_chunk_size.value) - i):
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
