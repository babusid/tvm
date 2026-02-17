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
        for b, h, l in T.grid(batch_size, T.int64(_hidden_size), T.int64(_state_len)):
            with T.sblock("copy_conv_state_to_total_hidden_state"):
                vb, vh, vl = T.axis.remap("SSS", [b, h, l])
                hidden_states_new[vb, vh, vl] = conv_state_buf[vb, vh, vl]

        for b, h, l in T.grid(batch_size, T.int64(_hidden_size), seq_len):
            with T.sblock("copy_new_hidden_state_to_total_hidden_state"):
                vb, vh, vl = T.axis.remap("SSS", [b, h, l])
                hidden_states_new[vb, vh, vl + T.int64(_state_len)] = hidden_states_buf[vb, vh, vl]

        # Step 2: Extract last state_len elements as next_conv_state
        for b, h, s in T.grid(batch_size, T.int64(_hidden_size), T.int64(_state_len)):
            with T.sblock("create_next_conv_state"):
                vb, vh, vs = T.axis.remap("SSS", [b, h, s])
                next_conv_state_buf[vb, vh, vs] = hidden_states_new[vb, vh, vs + seq_len]

        # Step 3: Compute causal convolution (sliding window)
        # For each position s in output, convolve with state_len window ending at s+state_len
        for b, h, s, kw in T.grid(batch_size, T.int64(_hidden_size), seq_len, T.int64(_state_len)):
            with T.sblock("compute_causal_conv"):
                vb, vh, vs, vkw = T.axis.remap("SSSR", [b, h, s, kw])
                with T.init():
                    conv_sum[vb, vh, vs] = T.float32(0.0)
                conv_sum[vb, vh, vs] += (
                    hidden_states_new[vb, vh, vs + vkw + T.int64(1)] * weight[vh, T.int64(0), vkw]
                )

        # Step 4: Add bias and apply activation
        if _activation == "silu":
            for b, h, s in T.grid(batch_size, T.int64(_hidden_size), seq_len):
                with T.sblock("apply_bias_activation_and_writeback"):
                    vb, vh, vs = T.axis.remap("SSS", [b, h, s])
                    biased_val = conv_sum[vb, vh, vs] + bias[vh]
                    val_float = T.cast(biased_val, "float32")
                    out_buf[vb, vh, vs] = T.cast(val_float * T.sigmoid(val_float), _dtype)
        else:
            for b, h, s in T.grid(batch_size, T.int64(_hidden_size), seq_len):
                with T.sblock("apply_bias_and_writeback"):
                    vb, vh, vs = T.axis.remap("SSS", [b, h, s])
                    out_buf[vb, vh, vs] = conv_sum[vb, vh, vs] + bias[vh]

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
        for b, h, l in T.grid(batch_size, T.int64(_hidden_size), T.int64(_state_len)):
            with T.sblock("copy_conv_state_to_total_hidden_state"):
                vb, vh, vl = T.axis.remap("SSS", [b, h, l])
                hidden_states_new[vb, vh, vl] = conv_state_buf[vb, vh, vl]

        for b, h, l in T.grid(batch_size, T.int64(_hidden_size), seq_len):
            with T.sblock("copy_new_hidden_state_to_total_hidden_state"):
                vb, vh, vl = T.axis.remap("SSS", [b, h, l])
                hidden_states_new[vb, vh, vl + T.int64(_state_len)] = hidden_states_buf[vb, vh, vl]

        # Step 2: Extract last state_len elements as next_conv_state
        for b, h, s in T.grid(batch_size, T.int64(_hidden_size), T.int64(_state_len)):
            with T.sblock("create_next_conv_state"):
                vb, vh, vs = T.axis.remap("SSS", [b, h, s])
                next_conv_state_buf[vb, vh, vs] = hidden_states_new[vb, vh, vs + seq_len]

        # Step 3: Compute causal convolution (sliding window)
        # For each position s in output, convolve with state_len window ending at s+state_len
        for b, h, s, kw in T.grid(batch_size, T.int64(_hidden_size), seq_len, T.int64(_state_len)):
            with T.sblock("compute_causal_conv"):
                vb, vh, vs, vkw = T.axis.remap("SSSR", [b, h, s, kw])
                with T.init():
                    conv_sum[vb, vh, vs] = T.float32(0.0)
                conv_sum[vb, vh, vs] += (
                    hidden_states_new[vb, vh, vs + vkw + T.int64(1)] * weight[vh, T.int64(0), vkw]
                )

        # Step 5: Apply activation (no bias)
        if _activation == "silu":
            for b, h, s in T.grid(batch_size, T.int64(_hidden_size), seq_len):
                with T.sblock("apply_activation_and_writeback"):
                    vb, vh, vs = T.axis.remap("SSS", [b, h, s])
                    val = conv_sum[vb, vh, vs]
                    # Apply activation if enabled
                    # SiLU(x) = x * sigmoid(x)
                    val_float = T.cast(val, "float32")
                    out_buf[vb, vh, vs] = T.cast(val_float * T.sigmoid(val_float), _dtype)
        else:
            for b, h, s in T.grid(batch_size, T.int64(_hidden_size), seq_len):
                with T.sblock("writeback"):
                    vb, vh, vs = T.axis.remap("SSS", [b, h, s])
                    val = conv_sum[vb, vh, vs]
                    out_buf[vb, vh, vs] = val

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
    """
    Strategy for prefix scan:
      - Use T.grid(batch, heads, chunks) for independent dimensions
      - Use bare range() loops for the sequential row scan (i) and inner (j, k)
      - Accumulate directly into output buffer locations to avoid the
        local-variable-mutation bug (where TVMScript creates new let-bindings
        instead of mutating).
    """

    # Capture compile-time constants
    _linear_num_value_heads = linear_num_value_heads
    _linear_value_head_dim = linear_value_head_dim
    _linear_num_key_heads = linear_num_key_heads
    _linear_key_head_dim = linear_key_head_dim
    _val_dim = linear_value_head_dim
    _chunk_size = chunk_size
    _output_final_state = output_final_state
    _dtype = dtype

    # additional constants
    _scale = 1.0 / math.sqrt(_linear_key_head_dim)
    _eps = 1e-6  # hardcode for now

    @T.prim_func
    def chunk_gated_delta_rule(
        query: T.handle, key: T.handle, value: T.handle, g: T.handle, beta: T.handle
    ):
        # only known at runtime
        batch_size = T.int64()
        seq_len = T.int64()
        T.evaluate(T.assume(seq_len > 0))
        T.evaluate(T.assume(batch_size > 0))
        # input buffers
        query_buf = T.match_buffer(
            query,
            (batch_size, seq_len, _linear_num_value_heads, _linear_key_head_dim),
            dtype=_dtype,
        )
        key_buf = T.match_buffer(
            key, (batch_size, seq_len, _linear_num_value_heads, _linear_key_head_dim), dtype=_dtype
        )
        value_buf = T.match_buffer(
            value,
            (batch_size, seq_len, _linear_num_value_heads, _linear_value_head_dim),
            dtype=_dtype,
        )
        g_buf = T.match_buffer(g, (batch_size, seq_len, _linear_num_value_heads), dtype=_dtype)
        beta_buf = T.match_buffer(
            beta, (batch_size, seq_len, _linear_num_value_heads), dtype=_dtype
        )

        # intermediate buffers have to be at root level, cant declare them later
        # pad_size = (_chunk_size - seq_len % _chunk_size) % _chunk_size
        query_T = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size),
                _linear_key_head_dim,
            ),
            dtype=_dtype,
        )
        key_T = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size),
                _linear_key_head_dim,
            ),
            dtype=_dtype,
        )
        k_beta = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size),
                _linear_key_head_dim,
            ),
            dtype=_dtype,
        )
        value_T = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size),
                _linear_value_head_dim,
            ),
            dtype=_dtype,
        )
        v_beta = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size),
                _linear_value_head_dim,
            ),
            dtype=_dtype,
        )
        g_T = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size),
            ),
            dtype=_dtype,
        )
        beta_T = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size),
            ),
            dtype=_dtype,
        )
        query_chunked = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                (seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size)) // _chunk_size,
                _chunk_size,
                _linear_key_head_dim,
            ),
            dtype=_dtype,
        )
        key_chunked = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                (seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size)) // _chunk_size,
                _chunk_size,
                _linear_key_head_dim,
            ),
            dtype=_dtype,
        )
        value_chunked = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                (seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size)) // _chunk_size,
                _chunk_size,
                _linear_value_head_dim,
            ),
            dtype=_dtype,
        )
        k_beta_chunked = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                (seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size)) // _chunk_size,
                _chunk_size,
                _linear_key_head_dim,
            ),
            dtype=_dtype,
        )
        v_beta_chunked = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                (seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size)) // _chunk_size,
                _chunk_size,
                _linear_value_head_dim,
            ),
            dtype=_dtype,
        )
        g_chunked = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                (seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size)) // _chunk_size,
                _chunk_size,
            ),
            dtype=_dtype,
        )
        mask = T.alloc_buffer((_chunk_size, _chunk_size), dtype=_dtype)

        # TODO: conditional l2 norm of q, k here

        # transpose, pad, scale query
        for b, s, nv, kd in T.grid(
            batch_size, seq_len, _linear_num_value_heads, _linear_key_head_dim
        ):
            with T.sblock("transpose_qk_scale_q"):
                vb, vs, vnv, vkd = T.axis.remap("SSSS", [b, s, nv, kd])
                T.reads(query_buf[vb, vs, vnv, vkd], key_buf[vb, vs, vnv, vkd])
                T.writes(query_T[vb, vnv, vs, vkd], key_T[vb, vnv, vs, vkd])
                # inline scaling of query while doing the transpose
                query_T[vb, vnv, vs, vkd] = query_buf[vb, vs, vnv, vkd] * (
                    1 / (_linear_value_head_dim**0.5)
                )
                key_T[vb, vnv, vs, vkd] = key_buf[vb, vs, vnv, vkd]

        for b, nv, s, kd in T.grid(
            batch_size,
            _linear_num_value_heads,
            ((_chunk_size - seq_len % _chunk_size) % _chunk_size),
            _linear_key_head_dim,
        ):
            with T.sblock("pad_fill_qk"):
                vb, vnv, vs, vkd = T.axis.remap("SSSS", [b, nv, s, kd])
                query_T[vb, vnv, seq_len + vs, vkd] = 0.0
                key_T[vb, vnv, seq_len + vs, vkd] = 0.0

        for b, s, nv, vd in T.grid(
            batch_size, seq_len, _linear_num_value_heads, _linear_value_head_dim
        ):
            with T.sblock("transpose_v"):
                vb, vs, vnv, vvd = T.axis.remap("SSSS", [b, s, nv, vd])
                T.reads(value_buf[vb, vs, vnv, vvd])
                T.writes(value_T[vb, vnv, vs, vvd])
                value_T[vb, vnv, vs, vvd] = value_buf[vb, vs, vnv, vvd]

        for b, nv, s, vd in T.grid(
            batch_size,
            _linear_num_value_heads,
            ((_chunk_size - seq_len % _chunk_size) % _chunk_size),
            _linear_value_head_dim,
        ):
            with T.sblock("pad_fill_v"):
                vb, vnv, vs, vvd = T.axis.remap("SSSS", [b, nv, s, vd])
                value_T[vb, vnv, seq_len + vs, vvd] = 0.0

        for b, s, nv in T.grid(batch_size, seq_len, _linear_num_value_heads):
            with T.sblock("transpose_gb"):
                vb, vs, vnv = T.axis.remap("SSS", [b, s, nv])
                T.reads(g_buf[vb, vs, vnv], beta_buf[vb, vs, vnv])
                T.writes(g_T[vb, vnv, vs], beta_T[vb, vnv, vs])
                g_T[vb, vnv, vs] = g_buf[vb, vs, vnv]
                beta_T[vb, vnv, vs] = beta_buf[vb, vs, vnv]

        for b, nv, s in T.grid(
            batch_size,
            _linear_num_value_heads,
            ((_chunk_size - seq_len % _chunk_size) % _chunk_size),
        ):
            with T.sblock("pad_fill_gb"):
                vb, vnv, vs = T.axis.remap("SSS", [b, nv, s])
                g_T[vb, vnv, seq_len + vs] = 0.0
                beta_T[vb, vnv, seq_len + vs] = 0.0

        for b, nv, s, vd in T.grid(
            batch_size,
            _linear_num_value_heads,
            seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size),
            _linear_value_head_dim,
        ):
            with T.sblock("v_beta"):
                vb, vnv, vs, vvd = T.axis.remap("SSSS", [b, nv, s, vd])
                v_beta[vb, vnv, vs, vvd] = value_T[vb, vnv, vs, vvd] * beta_T[vb, vnv, vs]

        for b, nk, s, kd in T.grid(
            batch_size,
            _linear_num_value_heads,
            seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size),
            _linear_key_head_dim,
        ):
            with T.sblock("k_beta"):
                vb, vnk, vs, vkd = T.axis.remap("SSSS", [b, nk, s, kd])
                k_beta[vb, vnk, vs, vkd] = key_T[vb, vnk, vs, vkd] * beta_T[vb, vnk, vs]

        for b, nv, s, kd in T.grid(
            batch_size,
            _linear_num_value_heads,
            seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size),
            _linear_key_head_dim,
        ):
            with T.sblock("chunk_reshape_q_k_kbeta"):
                vb, vnv, vs, vkd = T.axis.remap("SSSS", [b, nv, s, kd])
                query_chunked[vb, vnv, vs // _chunk_size, vs % _chunk_size, vkd] = query_T[
                    vb, vnv, vs, vkd
                ]
                key_chunked[vb, vnv, vs // _chunk_size, vs % _chunk_size, vkd] = key_T[
                    vb, vnv, vs, vkd
                ]
                k_beta_chunked[vb, vnv, vs // _chunk_size, vs % _chunk_size, vkd] = k_beta[
                    vb, vnv, vs, vkd
                ]

        for b, nv, s, vd in T.grid(
            batch_size,
            _linear_num_value_heads,
            seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size),
            _linear_value_head_dim,
        ):
            with T.sblock(("chunk_reshape_vbeta")):
                vb, vnv, vs, vvd = T.axis.remap("SSSS", [b, nv, s, vd])
                v_beta_chunked[vb, vnv, vs // _chunk_size, vs % _chunk_size, vvd] = v_beta[
                    vb, vnv, vs, vvd
                ]

        for b, nv, s in T.grid(
            batch_size,
            _linear_num_value_heads,
            seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size),
        ):
            with T.sblock("chunk_reshape_g"):
                vb, vnv, vs = T.axis.remap("SSS", [b, nv, s])
                g_chunked[vb, vnv, vs // _chunk_size, vs % _chunk_size] = g_T[vb, vnv, vs]

        for c1, c2 in T.grid(
                _chunk_size,
                _chunk_size
        ):
            with T.sblock("create_upper_triu_mask"):
                vc1, vc2 = T.axis.remap("SS", [c1,c2])
                # iterate from start column == row idx to end of row
                # clamp to last index
                mask[vc1, T.min(vc1 + vc2, _chunk_size-1)] = T.float32(1.0)



    if use_qk_l2norm_in_kernel:
        return None
    else:
        return chunk_gated_delta_rule


def _chunk_recurrent_gated_delta_rule():
    inputs = []
    outputs = []
    return te.create_prim_func(inputs + outputs)



mod = tvm.IRModule(
    {
        # "conv1d_biased_act": _gen_causal_conv1d_update(128, 128, has_bias=True),
        # "conv1d_biased": _gen_causal_conv1d_update(128, 128, has_bias=True, activation=None),
        # "conv1d_nobias_act": _gen_causal_conv1d_update(128, 128, has_bias=False),
        # "conv1d_nobias": _gen_causal_conv1d_update(128, 128, has_bias=False, activation=None),
        # "chunk_gated_delta_l2_norm": _chunk_gated_delta_rule(
        #    32, 128, 16, 128, use_qk_l2norm_in_kernel=True
        # ),
        "chunk_gated_delta": _chunk_gated_delta_rule(
            32, 128, 16, 128, use_qk_l2norm_in_kernel=False
        ),
    }
)

mod.show()

# from tvm import dlight as dl
# Target must be in scope or explicitly passed
target = tvm.target.Target("cuda")
from tvm.s_tir import dlight as dl

# DLight is now a transformation pass
with target:
    mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
    mod.show()

# Try to build to CUDA
print("\n\n=== Attempting tvm.build to CUDA ===")
try:
    built = tvm.build(mod, target=target)
    print("BUILD SUCCEEDED")
    # Print generated CUDA source
    print(built.imports[0].inspect_source())
except Exception as e:
    print(f"BUILD FAILED: {type(e).__name__}: {e}")
