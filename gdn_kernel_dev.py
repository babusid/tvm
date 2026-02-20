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
        query: T.handle,
        key: T.handle,
        value: T.handle,
        g: T.handle,
        beta: T.handle,
        last_recurrent_state: T.handle,
        core_attn_out: T.handle,
        recurrent_state_out: T.handle,
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

        last_recurrent_state_buf = T.match_buffer(
            last_recurrent_state,
            (
                batch_size,
                _linear_num_value_heads,
                _linear_key_head_dim,
                _linear_value_head_dim,
            ),
            dtype=_dtype,
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
        g_cumsum = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                (seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size)) // _chunk_size,
                _chunk_size,
            ),
            dtype=_dtype,
        )
        decay_mask = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                (seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size)) // _chunk_size,
                _chunk_size,
                _chunk_size,
            ),
            dtype=_dtype,
        )
        attn_mm_out = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                (seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size)) // _chunk_size,
                _chunk_size,
                _chunk_size,
            ),
            dtype=_dtype,
        )
        attn_decay_neg_mask_out = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                (seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size)) // _chunk_size,
                _chunk_size,
                _chunk_size,
            ),
            dtype=_dtype,
        )
        attn_associative_scan_out = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                (seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size)) // _chunk_size,
                _chunk_size,
                _chunk_size,
            ),
            dtype=_dtype,
        )

        attn_identity_add_out = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                (seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size)) // _chunk_size,
                _chunk_size,
                _chunk_size,
            ),
            dtype=_dtype,
        )

        value_attn_vbeta_matmul_out = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                (seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size)) // _chunk_size,
                _chunk_size,
                _linear_value_head_dim,
            )
        )

        k_beta_x_g_cumsum_exp_tmp = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                (seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size)) // _chunk_size,
                _chunk_size,
                _linear_key_head_dim,
            ),
            dtype=_dtype,
        )
        k_cumdecay = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                (seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size)) // _chunk_size,
                _chunk_size,
                _linear_key_head_dim,
            ),
            dtype=_dtype,
        )
        recurrent_state_update_attn_out = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                (seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size)) // _chunk_size,
                _chunk_size,
                _chunk_size,
            ),
            dtype=_dtype,
        )

        recurrent_state_update_vprime_out = T.alloc_buffer(
            (
                batch_size,
                _linear_num_value_heads,
                (seq_len + ((_chunk_size - seq_len % _chunk_size) % _chunk_size)) // _chunk_size,
                _chunk_size,
                _linear_value_head_dim
                ), dtype=_dtype,
        )

        padding = (_chunk_size - seq_len % _chunk_size) % _chunk_size
        padded_len = seq_len + padding
        num_chunks = padded_len // _chunk_size
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
                query_T[vb, vnv, vs, vkd] = query_buf[vb, vs, vnv, vkd] * _scale
                key_T[vb, vnv, vs, vkd] = key_buf[vb, vs, vnv, vkd]

        for b, nv, s, kd in T.grid(
            batch_size,
            _linear_num_value_heads,
            padding,
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
            padding,
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
            padding,
        ):
            with T.sblock("pad_fill_gb"):
                vb, vnv, vs = T.axis.remap("SSS", [b, nv, s])
                g_T[vb, vnv, seq_len + vs] = 0.0
                beta_T[vb, vnv, seq_len + vs] = 0.0

        for b, nv, s, vd in T.grid(
            batch_size,
            _linear_num_value_heads,
            padded_len,
            _linear_value_head_dim,
        ):
            with T.sblock("v_beta"):
                vb, vnv, vs, vvd = T.axis.remap("SSSS", [b, nv, s, vd])
                v_beta[vb, vnv, vs, vvd] = value_T[vb, vnv, vs, vvd] * beta_T[vb, vnv, vs]

        for b, nk, s, kd in T.grid(
            batch_size,
            _linear_num_value_heads,
            padded_len,
            _linear_key_head_dim,
        ):
            with T.sblock("k_beta"):
                vb, vnk, vs, vkd = T.axis.remap("SSSS", [b, nk, s, kd])
                k_beta[vb, vnk, vs, vkd] = key_T[vb, vnk, vs, vkd] * beta_T[vb, vnk, vs]

        for b, nv, s, kd in T.grid(
            batch_size,
            _linear_num_value_heads,
            padded_len,
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
            padded_len,
            _linear_value_head_dim,
        ):
            with T.sblock(("chunk_reshape_v_vbeta")):
                vb, vnv, vs, vvd = T.axis.remap("SSSS", [b, nv, s, vd])
                value_chunked[vb, vnv, vs // _chunk_size, vs % _chunk_size, vvd] = value_T[
                    vb, vnv, vs, vvd
                ]
                v_beta_chunked[vb, vnv, vs // _chunk_size, vs % _chunk_size, vvd] = v_beta[
                    vb, vnv, vs, vvd
                ]

        for b, nv, s in T.grid(
            batch_size,
            _linear_num_value_heads,
            padded_len,
        ):
            with T.sblock("chunk_reshape_g"):
                vb, vnv, vs = T.axis.remap("SSS", [b, nv, s])
                g_chunked[vb, vnv, vs // _chunk_size, vs % _chunk_size] = g_T[vb, vnv, vs]

        for b, nv, s, c, r in T.grid(
            batch_size,
            _linear_num_value_heads,
            num_chunks,
            _chunk_size,
            _chunk_size,
        ):
            # there's got to be a better way to do this than the n**2 approach
            # how can we do prefix scan?
            with T.sblock("g_cumsum"):
                vb, vnv, vs, vc, vr = T.axis.remap("SSSSR", [b, nv, s, c, r])
                with T.init():
                    g_cumsum[vb, vnv, vs, vc] = T.float32(0.0)
                g_cumsum[vb, vnv, vs, vc] = g_cumsum[vb, vnv, vs, vc] + T.if_then_else(
                    vr <= vc, g_chunked[vb, vnv, vs, vr], T.float32(0.0)
                )

        for b, nv, s, c1, c2 in T.grid(
            batch_size,
            _linear_num_value_heads,
            num_chunks,
            _chunk_size,
            _chunk_size,
        ):
            with T.sblock("create_decay_mask"):
                # step 1: g.unsqueeze(-1) - g.unsqueeze(-2)
                # b,v,s,c,1 - b,v,s,1,c
                # first one treats last dim (c) as a column, second one treats last dim (c) as a row
                # algo: let i, j be iterators on c
                #       we form a matrix where each i,j cell represents b,v,s,i - b,v,s,j
                # collapse the tril / exp / tril into just one stage
                vb, vnv, vs, i, j = T.axis.remap("SSSSS", [b, nv, s, c1, c2])
                decay_mask[vb, vnv, vs, i, j] = T.if_then_else(
                    i >= j,
                    T.exp(g_cumsum[vb, vnv, vs, i] - g_cumsum[vb, vnv, vs, j]),
                    T.float32(0.0),
                )

        # attn = k_beta_chunked @ key_chunked^T
        for b, nv, nc, c1, c2, kd in T.grid(
            batch_size,
            _linear_num_value_heads,
            num_chunks,
            _chunk_size,
            _chunk_size,
            _linear_key_head_dim,
        ):
            with T.sblock("attn_mm"):
                vb, vnv, vnc, vc1, vc2, vkd = T.axis.remap("SSSSSR", [b, nv, nc, c1, c2, kd])
                with T.init():
                    attn_mm_out[vb, vnv, vnc, vc1, vc2] = T.float32(0.0)

                attn_mm_out[vb, vnv, vnc, vc1, vc2] += (
                    k_beta_chunked[vb, vnv, vnc, vc1, vkd] * key_chunked[vb, vnv, vnc, vc2, vkd]
                )

        for b, nv, nc, c1, c2 in T.grid(
            batch_size,
            _linear_num_value_heads,
            num_chunks,
            _chunk_size,
            _chunk_size,
        ):
            with T.sblock("attn_decay_neg_mask"):
                vb, vnv, vnc, vc1, vc2 = T.axis.remap("SSSSS", [b, nv, nc, c1, c2])
                attn_decay_neg_mask_out[vb, vnv, vnc, vc1, vc2] = T.if_then_else(
                    vc2 >= vc1,
                    T.float32(0.0),
                    attn_mm_out[vb, vnv, vnc, vc1, vc2] * (-1 * decay_mask[vb, vnv, vnc, vc1, vc2]),
                )

        # Associative scan: for each row i, update row i using rows 0..i-1
        # attn[i, j] = attn[i, j] + sum_{k=0}^{i-1} attn[i, k] * attn[k, j]
        # Strategy: Parallelize over (batch, heads, chunks), sequential scan within each thread
        # Using explicit thread_binding here since s_tir.Schedule API doesn't support triangular bounds?
        for b in T.thread_binding(batch_size, thread="blockIdx.x"):
            for nv in T.thread_binding(_linear_num_value_heads, thread="blockIdx.y"):
                for nc in T.thread_binding(num_chunks, thread="threadIdx.x"):
                    # Sequential scan over rows - each row i depends on rows 0..i-1
                    for i in T.serial(1, _chunk_size):
                        # For each column j in row i (j < i for lower triangular)
                        for j in T.serial(0, i):
                            # Start with original value
                            attn_associative_scan_out[b, nv, nc, i, j] = attn_decay_neg_mask_out[
                                b, nv, nc, i, j
                            ]
                            # selects each element in the row we're on, and the jth element of each each row below
                            # current row. this mirrors the broadcast and elementwise multiply
                            # use inline += to mirror the column sum
                            for k in T.serial(0, i):
                                attn_associative_scan_out[b, nv, nc, i, j] += (
                                    attn_associative_scan_out[b, nv, nc, i, k]
                                    * attn_associative_scan_out[b, nv, nc, k, j]
                                )

        for b, nv, nc, c1, c2 in T.grid(
            batch_size,
            _linear_num_value_heads,
            num_chunks,
            _chunk_size,
            _chunk_size,
        ):
            with T.sblock("attn_add_identity"):
                vb, vnv, vnc, vc1, vc2 = T.axis.remap("SSSSS", [b, nv, nc, c1, c2])
                attn_identity_add_out[vb, vnv, vnc, vc1, vc2] = attn_associative_scan_out[
                    vb, vnv, vnc, vc1, vc2
                ] + T.if_then_else(vc2 == vc1, T.float32(1.0), T.float32(0.0))

        # value = attn @ v_beta
        for b, nv, nc, c, vd, r in T.grid(
            batch_size,
            _linear_value_head_dim,
            num_chunks,
            _chunk_size,
            _linear_value_head_dim,
            _chunk_size,
        ):
            with T.sblock("value_attn_vbeta_matmul"):
                vb, vnv, vnc, vc, vvd, vr = T.axis.remap("SSSSSR", [b, nv, nc, c, vd, r])
                with T.init():
                    value_attn_vbeta_matmul_out[vb, vnv, vnc, vc, vvd] = T.float32(0.0)

                value_attn_vbeta_matmul_out[vb, vnv, vnc, vc, vvd] += (
                    attn_identity_add_out[vb, vnv, vnc, vc, vr]
                    * v_beta_chunked[vb, vnv, vnc, vr, vvd]
                )

        # k_cumdecay = attn @ (k_beta x g.exp().unsqueeze(-1))
        # attn_identity_add_out
        # k_beta_chunked
        # g_cumsum
        for b, nv, nc, c, vd in T.grid(
            batch_size, _linear_num_value_heads, num_chunks, _chunk_size, _linear_value_head_dim
        ):
            with T.sblock("k_cumdecay_internal"):
                vb, vnv, vnc, vc, vvd = T.axis.remap("SSSSS", [b, nv, nc, c, vd])
                k_beta_x_g_cumsum_exp_tmp[vb, vnv, vnc, vc, vvd] = k_beta_chunked[
                    vb, vnv, vnc, vc, vvd
                ] * T.exp(g_cumsum[vb, vnv, vnc, vc])

        for b, nv, nc, c, kd, r in T.grid(
            batch_size,
            _linear_value_head_dim,
            num_chunks,
            _chunk_size,
            _linear_key_head_dim,
            _chunk_size,
        ):
            with T.sblock("k_cumdecay_mm"):
                vb, vnv, vnc, vc, vkd, vr = T.axis.remap("SSSSSR", [b, nv, nc, c, kd, r])
                with T.init():
                    k_cumdecay[vb, vnv, vnc, vc, vkd] = T.float32(0.0)

                k_cumdecay[vb, vnv, vnc, vc, vkd] += (
                    attn_identity_add_out[vb, vnv, vnc, vc, vr]
                    * k_beta_x_g_cumsum_exp_tmp[vb, vnv, vnc, vr, vkd]
                )

        # for i in range(0, total_sequence_length // chunk_size):
        #   q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        #   attn = (q_i @ k_i.transpose(-1,-2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        for b, nv, nc, c1, c2, r in T.grid(
            batch_size,
            _linear_num_value_heads,
            num_chunks,
            _chunk_size,
            _chunk_size,
            _linear_key_head_dim,
        ):
            with T.sblock("recurrent_state_mm_1"):
                vb, vnv, vnc, vc1, vc2, vr = T.axis.remap("SSSSSR", [b, nv, nc, c1, c2, r])
                with T.init():
                    recurrent_state_update_attn_out[vb, vnv, vnc, vc1, vc2] = T.float32(0.0)

                recurrent_state_update_attn_out[vb, vnv, vnc, vc1, vc2] = T.if_then_else(
                    vc2 < vc1,
                    (
                        recurrent_state_update_attn_out[vb, vnv, vnc, vc1, vc2]
                        + (
                            query_chunked[vb, vnv, vnc, vc1, vr]
                            * key_chunked[vb, vnv, vnc, vc2, vr]
                        )
                        * decay_mask[vb, vnv, vnc, vc1, vc2]
                    ),
                    T.float32(0.0),
                )
        for b, nv, nc, c, vd, r in T.grid(
                batch_size,
                _linear_num_value_heads,
                num_chunks,
                _chunk_size,
                _linear_value_head_dim,
                _linear_key_head_dim
        ):
            with T.sblock("recurrent_state_mm_2"):
                vb, vnv, vnc, vc, vvd, vr = T.axis.remap("SSSSSR", [b, nv, nc, c, vd, r])
                with T.init():
                    recurrent_state_update_vprime_out[vb, vnv, vnc, vc, vvd] = T.float32(0.0)
                
                recurrent_state_update_vprime_out[vb, vnv, vnc, vc, vvd] += (
                        k_cumdecay[vb, vnv, vnc, vc, vr] * last_recurrent_state_buf[vb, vnv, vr, vvd]
                )

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

print("\n\n=== raw tir  ===")
mod.show()

target = tvm.target.Target("cuda")

# ======================================================================
# Manual scheduling for all blocks.
# dlight's Fallback scheduler cannot handle some of our weirder blocks
# like the 6d matmuls and the associative scan
# ======================================================================

MAX_THREADS = 1024

sch = s_tir.Schedule(mod["chunk_gated_delta"])


def schedule_spatial_fallback(sch, block_name):
    """
    Replicate dlight Fallback for a pure-spatial block:
      fuse all loops -> split [blockIdx.x, threadIdx.x(1024)]
    """
    block = sch.get_sblock(block_name)
    loops = sch.get_loops(block)
    fused = sch.fuse(*loops)
    bx, tx = sch.split(fused, factors=[None, MAX_THREADS])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")


def schedule_reduction_fallback(sch, block_name):
    """
    Replicate dlight Fallback for a reduction block:
      reorder spatial loops before reduction loops,
      fuse spatial -> split [blockIdx.x, threadIdx.x(1024)],
      decompose_reduction at the first reduction loop.
    """
    block = sch.get_sblock(block_name)
    loops = sch.get_loops(block)
    # The last loop is the reduction loop (R), all others are spatial (S)
    s_loops = loops[:-1]
    r_loop = loops[-1]
    fused = sch.fuse(*s_loops)
    bx, tx = sch.split(fused, factors=[None, MAX_THREADS])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")
    sch.decompose_reduction(block, r_loop)


# --- Schedule all spatial blocks (pure element-wise / transpose / pad / reshape) ---
spatial_blocks = [
    "transpose_qk_scale_q",
    "pad_fill_qk",
    "transpose_v",
    "pad_fill_v",
    "transpose_gb",
    "pad_fill_gb",
    "v_beta",
    "k_beta",
    "chunk_reshape_q_k_kbeta",
    "chunk_reshape_v_vbeta",
    "chunk_reshape_g",
    "create_decay_mask",
    "attn_decay_neg_mask",
    "attn_add_identity",
    "k_cumdecay_internal",
    # Note: attn_assoc_scan uses explicit thread_binding in TIR, so not scheduled here
]

for name in spatial_blocks:
    schedule_spatial_fallback(sch, name)

# Schedule g_cumsum (5D with reduction on last axis) ---
# Loops: b, nv, num_chunks, chunk_size, chunk_size(R)
schedule_reduction_fallback(sch, "g_cumsum")


def schedule_6d_batched_matmul(sch, block_name, spatial_dim_names, reduction_name):
    """
    Schedule a 6D batched matmul: (b, nv, nc, d1, d2, r)
    Strategy: bind b->blockIdx.x, nv->blockIdx.y, nc->blockIdx.z,
              fuse d1*d2 -> split to threads, keep r as serial reduction
    """
    block = sch.get_sblock(block_name)
    loops = sch.get_loops(block)

    # Unpack: b, nv, nc, d1, d2, r
    b, nv, nc = loops[0], loops[1], loops[2]
    d1, d2 = loops[3], loops[4]
    r = loops[5]

    sch.bind(b, "blockIdx.x")
    sch.bind(nv, "blockIdx.y")
    sch.bind(nc, "blockIdx.z")

    # Fuse the two constant spatial dims and split to threads
    d1_d2 = sch.fuse(d1, d2)
    d1_d2_o, d1_d2_i = sch.split(d1_d2, factors=[None, MAX_THREADS])
    sch.bind(d1_d2_o, "vthread.x")
    sch.bind(d1_d2_i, "threadIdx.x")

    # Decompose reduction
    sch.decompose_reduction(block, r)


# manually schedule the high dim batched matmuls
# Pattern: (batch, heads, chunks, dim1, dim2, reduction)
# Strategy: bind outer 3 dims to blocks, fuse inner 2 constant dims to threads

# attn_mm: (b, nv, nc, c1, c2, kd) where c1=c2=64, kd=128
schedule_6d_batched_matmul(sch, "attn_mm", ["c1", "c2"], "kd")

# value_attn_vbeta_matmul: (b, nv, nc, c, vd, r) where c=64, vd=128, r=64
schedule_6d_batched_matmul(sch, "value_attn_vbeta_matmul", ["c", "vd"], "r")

# k_cumdecay_mm: (b, nv, nc, c, kd, r) where c=64, kd=128, r=64
schedule_6d_batched_matmul(sch, "k_cumdecay_mm", ["c", "kd"], "r")

# recurrent_state_mm_1_out: (b, nv, nc, c1, c2, r)
schedule_6d_batched_matmul(sch, "recurrent_state_mm_1", ["c1", "c2"], "r")

# recurrent_state_mm_2:
schedule_6d_batched_matmul(sch,"recurrent_state_mm_2", ["c", "vd"], "r")

# Apply is_scheduled attribute so dlight/default GPU schedule skips this
mod["chunk_gated_delta"] = sch.mod["main"].with_attr("tir.is_scheduled", True)

print("\n\n=== manually scheduled tir  ===")
mod.show()


# Try to build to CUDA
print("\n\n=== Attempting tvm.build to CUDA ===")
try:
    built = tvm.build(mod, target=target)
    print("BUILD SUCCEEDED")
    # Print generated CUDA source
    cuda_src = built.imports[0].inspect_source()
    print(cuda_src)
except Exception as e:
    print(f"BUILD FAILED: {type(e).__name__}: {e}")
