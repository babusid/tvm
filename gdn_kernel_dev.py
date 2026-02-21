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
        # Intermediate buffers (externalized for debugging)
        query_T_ptr: T.handle,
        key_T_ptr: T.handle,
        k_beta_ptr: T.handle,
        value_T_ptr: T.handle,
        v_beta_ptr: T.handle,
        g_T_ptr: T.handle,
        beta_T_ptr: T.handle,
        query_chunked_ptr: T.handle,
        key_chunked_ptr: T.handle,
        value_chunked_ptr: T.handle,
        k_beta_chunked_ptr: T.handle,
        v_beta_chunked_ptr: T.handle,
        g_chunked_ptr: T.handle,
        g_cumsum_ptr: T.handle,
        decay_mask_ptr: T.handle,
        attn_mm_out_ptr: T.handle,
        attn_decay_neg_mask_out_ptr: T.handle,
        attn_associative_scan_out_ptr: T.handle,
        attn_identity_add_out_ptr: T.handle,
        value_attn_vbeta_matmul_out_ptr: T.handle,
        k_beta_x_g_cumsum_exp_tmp_ptr: T.handle,
        k_cumdecay_ptr: T.handle,
        recurrent_state_update_attn_out_ptr: T.handle,
        recurrent_state_update_v_buf_ptr: T.handle,
        recurrent_state_update_attn_inter_ptr: T.handle,
        core_attn_out_inter_ptr: T.handle,
    ):
        # only known at runtime
        batch_size = T.int64()
        seq_len = T.int64()
        T.evaluate(T.assume(seq_len > 0))
        T.evaluate(T.assume(batch_size > 0))
        # output buffers
        recurrent_state_out_buf = T.match_buffer(
            recurrent_state_out,
            (batch_size, _linear_num_value_heads, _linear_key_head_dim, _linear_value_head_dim),
            dtype=_dtype,
        )
        
        # Final output shape: (batch, seq_len, heads, value_dim)
        # This is after unchunking, removing padding, and transposing
        core_attn_out_buf = T.match_buffer(
            core_attn_out,
            (batch_size, seq_len, _linear_num_value_heads, _linear_value_head_dim),
            dtype=_dtype,
        )
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

        # intermediate buffers - externalized for debugging
        query_T = T.match_buffer(
            query_T_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size)*_chunk_size,
                _linear_key_head_dim,
            ),
            dtype=_dtype,
        )
        key_T = T.match_buffer(
            key_T_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size)*_chunk_size,
                _linear_key_head_dim,
            ),
            dtype=_dtype,
        )
        k_beta = T.match_buffer(
            k_beta_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size)*_chunk_size,
                _linear_key_head_dim,
            ),
            dtype=_dtype,
        )
        value_T = T.match_buffer(
            value_T_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size)*_chunk_size,
                _linear_value_head_dim,
            ),
            dtype=_dtype,
        )
        v_beta = T.match_buffer(
            v_beta_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size)*_chunk_size,
                _linear_value_head_dim,
            ),
            dtype=_dtype,
        )
        g_T = T.match_buffer(
            g_T_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size)*_chunk_size,
            ),
            dtype=_dtype,
        )
        beta_T = T.match_buffer(
            beta_T_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size)*_chunk_size,
            ),
            dtype=_dtype,
        )

        query_chunked = T.match_buffer(
            query_chunked_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
                _linear_key_head_dim,
            ),
            dtype=_dtype,
        )
        key_chunked = T.match_buffer(
            key_chunked_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
                _linear_key_head_dim,
            ),
            dtype=_dtype,
        )
        value_chunked = T.match_buffer(
            value_chunked_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
                _linear_value_head_dim,
            ),
            dtype=_dtype,
        )
        k_beta_chunked = T.match_buffer(
            k_beta_chunked_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
                _linear_key_head_dim,
            ),
            dtype=_dtype,
        )
        v_beta_chunked = T.match_buffer(
            v_beta_chunked_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
                _linear_value_head_dim,
            ),
            dtype=_dtype,
        )
        g_chunked = T.match_buffer(
            g_chunked_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
            ),
            dtype=_dtype,
        )
        g_cumsum = T.match_buffer(
            g_cumsum_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
            ),
            dtype=_dtype,
        )
        decay_mask = T.match_buffer(
            decay_mask_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
                _chunk_size,
            ),
            dtype=_dtype,
        )
        attn_mm_out = T.match_buffer(
            attn_mm_out_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
                _chunk_size,
            ),
            dtype=_dtype,
        )
        attn_decay_neg_mask_out = T.match_buffer(
            attn_decay_neg_mask_out_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
                _chunk_size,
            ),
            dtype=_dtype,
        )
        attn_associative_scan_out = T.match_buffer(
            attn_associative_scan_out_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
                _chunk_size,
            ),
            dtype=_dtype,
        )

        attn_identity_add_out = T.match_buffer(
            attn_identity_add_out_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
                _chunk_size,
            ),
            dtype=_dtype,
        )

        value_attn_vbeta_matmul_out = T.match_buffer(
            value_attn_vbeta_matmul_out_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
                _linear_value_head_dim,
            ),
            dtype=_dtype,
        )

        k_beta_x_g_cumsum_exp_tmp = T.match_buffer(
            k_beta_x_g_cumsum_exp_tmp_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
                _linear_key_head_dim,
            ),
            dtype=_dtype,
        )
        k_cumdecay = T.match_buffer(
            k_cumdecay_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
                _linear_key_head_dim,
            ),
            dtype=_dtype,
        )
        recurrent_state_update_attn_out = T.match_buffer(
            recurrent_state_update_attn_out_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
                _chunk_size,
            ),
            dtype=_dtype,
        )

        recurrent_state_update_v_buf = T.match_buffer(
            recurrent_state_update_v_buf_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
                _linear_value_head_dim,
            ),
            dtype=_dtype,
        )

        recurrent_state_update_attn_inter = T.match_buffer(
            recurrent_state_update_attn_inter_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
                _linear_value_head_dim,
            ),
            dtype=_dtype,
        )

        # Intermediate output buffer in chunked format
        # Will be transformed to final output shape at the end
        core_attn_out_inter = T.match_buffer(
            core_attn_out_inter_ptr,
            (
                batch_size,
                _linear_num_value_heads,
                tir.ceildiv(seq_len, _chunk_size),
                _chunk_size,
                _linear_value_head_dim,
            ),
            dtype=_dtype,
        )

        # TODO: conditional l2 norm of q, k here
        
        # Compute padding size for fill operations
        # Use ceildiv to avoid TVM generating negative modulo expressions
        padded_len = tir.ceildiv(seq_len, _chunk_size) * _chunk_size
        padding = padded_len - seq_len


        # TODO: copy the recurrent_state_buf to the recurrent_state_out_buf as an init step
        for b, nv, kd, vd in T.grid(
                batch_size, _linear_num_value_heads, _linear_key_head_dim, _linear_value_head_dim
                ):
            with T.sblock("copy_last_recurrent_to_out"):
                vb, vnv, vkd, vvd = T.axis.remap("SSSS", [b,nv,kd,vd])
                recurrent_state_out_buf[vb, vnv, vkd, vvd] = last_recurrent_state_buf[vb, vnv, vkd, vvd]

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
            tir.ceildiv(seq_len, _chunk_size),
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
            tir.ceildiv(seq_len, _chunk_size),
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
            tir.ceildiv(seq_len, _chunk_size),
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
            tir.ceildiv(seq_len, _chunk_size),
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
                for nc in T.thread_binding(tir.ceildiv(seq_len, _chunk_size), thread="threadIdx.x"):
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
                            # CRITICAL: Must read from PREVIOUS rows (attn_decay_neg_mask_out) not current buffer!
                            for k in T.serial(0, i):
                                attn_associative_scan_out[b, nv, nc, i, j] += (
                                    attn_decay_neg_mask_out[b, nv, nc, i, k]
                                    * attn_associative_scan_out[b, nv, nc, k, j]
                                )

        for b, nv, nc, c1, c2 in T.grid(
            batch_size,
            _linear_num_value_heads,
            tir.ceildiv(seq_len, _chunk_size),
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
            _linear_num_value_heads,
            tir.ceildiv(seq_len, _chunk_size),
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
            batch_size, _linear_num_value_heads, tir.ceildiv(seq_len, _chunk_size), _chunk_size, _linear_value_head_dim
        ):
            with T.sblock("k_cumdecay_internal"):
                vb, vnv, vnc, vc, vvd = T.axis.remap("SSSSS", [b, nv, nc, c, vd])
                k_beta_x_g_cumsum_exp_tmp[vb, vnv, vnc, vc, vvd] = k_beta_chunked[
                    vb, vnv, vnc, vc, vvd
                ] * T.exp(g_cumsum[vb, vnv, vnc, vc])

        for b, nv, nc, c, kd, r in T.grid(
            batch_size,
            _linear_num_value_heads,
            tir.ceildiv(seq_len, _chunk_size),
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
        #   attn = (q_i @ k_i.transpose(-1,-2) * decay_mask_buf[:, :, i]).masked_fill_(mask, 0)
        # Sequential inter-chunk loop
        # NOTE: The intra-chunk attention (q @ k.T * decay_mask) is computed fresh inside the loop
        # for each chunk, so we don't pre-compute it here
        # Parallelize over (batch, heads), serialize over chunks for recurrent state dependency
        # Each element within the chunk (chunk_size) can be done in parallel though.
        for b in T.thread_binding(batch_size, thread="blockIdx.x"):
            for nv in T.thread_binding(_linear_num_value_heads, thread="blockIdx.y"):
                for i in T.serial(tir.ceildiv(seq_len, _chunk_size)):  # Sequential over chunks
                    # First, compute intra-chunk attention for this chunk
                    # PyTorch line 368: attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
                    # where mask = torch.triu(..., diagonal=1) - masks strictly upper triangular (excludes diagonal)
                    for c1 in T.thread_binding(_chunk_size, thread="threadIdx.x"):
                        for c2 in T.serial(_chunk_size):
                            # Compute q_i @ k_i.T
                            recurrent_state_update_attn_out[b, nv, i, c1, c2] = T.float32(0.0)
                            for kd in T.serial(_linear_key_head_dim):
                                recurrent_state_update_attn_out[b, nv, i, c1, c2] += (
                                    query_chunked[b, nv, i, c1, kd] 
                                    * key_chunked[b, nv, i, c2, kd]
                                )
                            # Apply decay mask and causal mask (c2 > c1 → 0)
                            recurrent_state_update_attn_out[b, nv, i, c1, c2] = T.if_then_else(
                                c2 > c1,  # Strictly upper triangular (diagonal=1)
                                T.float32(0.0),
                                recurrent_state_update_attn_out[b, nv, i, c1, c2] * decay_mask[b, nv, i, c1, c2]
                            )
                    
                    for c in T.thread_binding(_chunk_size, thread="threadIdx.x"):
                        # v_prime = k_cumdecay_buf[:, :, i] @ last_recurrent_state
                        # Shape: k_cumdecay_buf[b, nv, i, chunk_size, key_dim] @ last_recurrent_state[b, nv, key_dim, value_dim]
                        #        -> v_prime[b, nv, i, chunk_size, value_dim]
                        for vd in T.serial(_linear_value_head_dim):
                            # Initialize v_prime for this (c, vd) position
                            recurrent_state_update_v_buf[b, nv, i, c, vd] = T.float32(0.0)
                            # Reduction over key_dim
                            for kd in T.serial(_linear_key_head_dim):
                                recurrent_state_update_v_buf[b, nv, i, c, vd] += (
                                    k_cumdecay[b, nv, i, c, kd]
                                    * recurrent_state_out_buf[b, nv, kd, vd]
                                )
                            # calculate the v_new in place: v_new = v_i - v_prime
                            # where v_i is v[:,:,i]
                            # BUG FIX: Remove redundant loop - this operation doesn't use kd
                            recurrent_state_update_v_buf[b, nv, i, c, vd] = (
                                value_attn_vbeta_matmul_out[b, nv, i, c, vd]
                                - recurrent_state_update_v_buf[b, nv, i, c, vd]
                            )
                        for vd in T.serial(_linear_value_head_dim):
                            # (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
                            # initialize this position
                            recurrent_state_update_attn_inter[b, nv, i, c, vd] = T.float32(0.0)
                            # reduction over key dim with inlined elemw matrix
                            for kd in T.serial(_linear_key_head_dim):
                                elemw = query_chunked[b, nv, i, c, kd] * T.exp(
                                    g_cumsum[b, nv, i, c]
                                )
                                recurrent_state_update_attn_inter[b, nv, i, c, vd] += (
                                    elemw * recurrent_state_out_buf[b, nv, kd, vd]
                                )
                        
                        for vd in T.serial(_linear_value_head_dim):
                            # compute core_attn_out to intermediate buffer
                            # initialize to the attn_inter value
                            core_attn_out_inter[b, nv, i, c, vd] = recurrent_state_update_attn_inter[b, nv, i, c, vd]

                            for r in T.serial(_chunk_size):
                                # Use the intra-chunk attention computed fresh for this chunk
                                attn_val = recurrent_state_update_attn_out[b, nv, i, c, r]
                                v_new_val = recurrent_state_update_v_buf[b, nv, i, r, vd]
                                core_attn_out_inter[b, nv, i, c, vd] += attn_val * v_new_val
                    
                    # CRITICAL FIX: Recurrent state update must happen OUTSIDE thread binding to avoid race condition
                    # Update recurrent state after processing chunk i
                    # PyTorch: last_recurrent_state = last_recurrent_state * g[:, :, i, -1, None, None].exp()
                    #          + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
                    for kd in T.serial(_linear_key_head_dim):
                        for vd in T.serial(_linear_value_head_dim):
                            # Part 1: Decay old state by exp(g[i, -1])
                            # g_cumsum_buf[b, nv, i, chunk_size-1] is the cumulative sum at the last position
                            g_last = g_cumsum[b, nv, i, _chunk_size - 1]
                            recurrent_state_out_buf[b, nv, kd, vd] = (
                                recurrent_state_out_buf[b, nv, kd, vd] * T.exp(g_last)
                            )
                            
                            # Part 2: Add contribution from current chunk: k.T @ (v_new * decay)
                            # decay[c_pos] = exp(g[i, -1] - g[i, c_pos]) for each position c_pos in chunk
                            # This is an outer product reduction: sum over chunk positions
                            for c_pos in T.serial(_chunk_size):
                                # Compute decay factor for this position
                                g_curr = g_cumsum[b, nv, i, c_pos]
                                decay_factor = T.exp(g_last - g_curr)
                                
                                # k[c_pos, kd] * (v_new[c_pos, vd] * decay_factor)
                                k_val = key_chunked[b, nv, i, c_pos, kd]
                                v_new_val = recurrent_state_update_v_buf[b, nv, i, c_pos, vd]
                                
                                recurrent_state_out_buf[b, nv, kd, vd] += (
                                    k_val * v_new_val * decay_factor
                                )
        
        # Transform intermediate output to final output shape
        # 1. Unchunk: (batch, heads, tir.ceildiv(seq_len, _chunk_size), chunk_size, value_dim) -> implicit flatten
        # 2. Remove padding: only copy first seq_len positions
        # 3. Transpose: (batch, heads, seq_len, value_dim) -> (batch, seq_len, heads, value_dim)
        for b, s, nv, vd in T.grid(batch_size, seq_len, _linear_num_value_heads, _linear_value_head_dim):
            with T.sblock("transform_output"):
                vb, vs, vnv, vvd = T.axis.remap("SSSS", [b, s, nv, vd])
                # Compute which chunk and position within chunk
                chunk_idx = vs // _chunk_size
                chunk_pos = vs % _chunk_size
                # Read from intermediate buffer and write to final output with transpose
                core_attn_out_buf[vb, vs, vnv, vvd] = core_attn_out_inter[vb, vnv, chunk_idx, chunk_pos, vvd]

    if use_qk_l2norm_in_kernel:
        return None
    else:
        return chunk_gated_delta_rule


def _chunk_recurrent_gated_delta_rule():
    inputs = []
    outputs = []
    return te.create_prim_func(inputs + outputs)


# ============================================================================
# Builder functions for testing and usage
# ============================================================================

def build_causal_conv1d_update(
    hidden_size: int,
    state_len: int,
    has_bias: bool = True,
    activation: str = "silu",
    target: str = "cuda",
    dtype: str = "float32",
):
    """
    Build and return a compiled TVM runtime module for causal_conv1d_update.
    
    Args:
        hidden_size: Number of hidden dimensions
        state_len: Length of convolution state (kernel_size - 1)
        has_bias: Whether to include bias
        activation: Activation function ("silu" or None)
        target: Target platform (default: "cuda")
        dtype: Data type (default: "float32")
    
    Returns:
        Compiled TVM runtime module
    """
    from tvm.s_tir import dlight as dl
    
    func = _gen_causal_conv1d_update(
        hidden_size=hidden_size,
        state_len=state_len,
        dtype=dtype,
        activation=activation,
        has_bias=has_bias,
    )
    mod = tvm.IRModule({"causal_conv1d_update": func})
    target_obj = tvm.target.Target(target)
    
    # Apply dlight scheduling for GPU
    with target_obj:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)

    with tvm.transform.PassContext(config={"tir.disable_storage_rewrite": True}):
        built = tir.build(mod, target=target_obj)
    return built


def build_chunk_gated_delta_rule(
    linear_num_value_heads: int,
    linear_value_head_dim: int,
    linear_num_key_heads: int,
    linear_key_head_dim: int,
    chunk_size: int = 64,
    target: str = "cuda",
    dtype: str = "float32",
):
    """
    Build and return a compiled TVM runtime module for chunk_gated_delta_rule.
    Includes manual scheduling for optimal performance.
    
    Args:
        linear_num_value_heads: Number of value heads
        linear_value_head_dim: Dimension of value heads
        linear_num_key_heads: Number of key heads
        linear_key_head_dim: Dimension of key heads
        chunk_size: Size of chunks (default: 64)
        target: Target platform (default: "cuda")
        dtype: Data type (default: "float32")
    
    Returns:
        Compiled TVM runtime module
    """
    MAX_THREADS = 1024
    
    func = _chunk_gated_delta_rule(
        linear_num_value_heads=linear_num_value_heads,
        linear_value_head_dim=linear_value_head_dim,
        linear_num_key_heads=linear_num_key_heads,
        linear_key_head_dim=linear_key_head_dim,
        chunk_size=chunk_size,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        dtype=dtype,
    )
    
    mod = tvm.IRModule({"chunk_gated_delta": func})
    sch = s_tir.Schedule(mod["chunk_gated_delta"])
    
    # Helper functions for scheduling
    def schedule_spatial_fallback(sch_local, block_name):
        """Fallback scheduling for spatial blocks."""
        block = sch_local.get_sblock(block_name)
        loops = sch_local.get_loops(block)
        fused = sch_local.fuse(*loops)
        bx, tx = sch_local.split(fused, factors=[None, MAX_THREADS])
        sch_local.bind(bx, "blockIdx.x")
        sch_local.bind(tx, "threadIdx.x")
    
    def schedule_padding_block(sch_local, block_name):
        """Specialized scheduling for padding blocks to avoid fusing symbolic padding dimension."""
        block = sch_local.get_sblock(block_name)
        loops = sch_local.get_loops(block)
        # Padding blocks have structure: (batch, num_heads, padding, dim)
        # Don't fuse the 'padding' dimension (symbolic expression)
        if len(loops) == 4:
            b, nv, s, d = loops
            # Fuse only batch and num_heads
            fused = sch_local.fuse(b, nv)
            sch_local.bind(fused, "blockIdx.x")
            # Keep padding dimension serial (it's typically small: 0-63)
            # Bind the inner constant dimension to threads
            sch_local.bind(d, "threadIdx.x")
        elif len(loops) == 3:
            # For pad_fill_gb: (batch, num_heads, padding)
            b, nv, s = loops
            fused = sch_local.fuse(b, nv)
            sch_local.bind(fused, "blockIdx.x")
            # Keep padding dimension serial
    
    def schedule_reduction_fallback(sch_local, block_name):
        """Fallback scheduling for reduction blocks."""
        block = sch_local.get_sblock(block_name)
        loops = sch_local.get_loops(block)
        s_loops = loops[:-1]
        r_loop = loops[-1]
        fused = sch_local.fuse(*s_loops)
        bx, tx = sch_local.split(fused, factors=[None, MAX_THREADS])
        sch_local.bind(bx, "blockIdx.x")
        sch_local.bind(tx, "threadIdx.x")
        sch_local.decompose_reduction(block, r_loop)
    
    def schedule_6d_batched_matmul(sch_local, block_name, spatial_dim_names, reduction_name):
        """Schedule a 6D batched matmul."""
        block = sch_local.get_sblock(block_name)
        loops = sch_local.get_loops(block)
        b, nv, nc = loops[0], loops[1], loops[2]
        d1, d2 = loops[3], loops[4]
        r = loops[5]
        sch_local.bind(b, "blockIdx.x")
        sch_local.bind(nv, "blockIdx.y")
        sch_local.bind(nc, "blockIdx.z")
        d1_d2 = sch_local.fuse(d1, d2)
        d1_d2_o, d1_d2_i = sch_local.split(d1_d2, factors=[None, MAX_THREADS])
        sch_local.bind(d1_d2_o, "vthread.x")
        sch_local.bind(d1_d2_i, "threadIdx.x")
        sch_local.decompose_reduction(block, r)
    
    # Apply scheduling to all blocks
    spatial_blocks = [
        "copy_last_recurrent_to_out",
        "transpose_qk_scale_q",
        "transpose_v",
        "transpose_gb",
        "v_beta",
        "k_beta",
        "chunk_reshape_q_k_kbeta",
        "chunk_reshape_v_vbeta",
        "chunk_reshape_g",
        "create_decay_mask",
        "attn_decay_neg_mask",
        "attn_add_identity",
        "k_cumdecay_internal",
        "transform_output",
    ]
    
    padding_blocks = [
        "pad_fill_qk",
        "pad_fill_v",
        "pad_fill_gb",
    ]
    
    for name in spatial_blocks:
        schedule_spatial_fallback(sch, name)
    
    for name in padding_blocks:
        schedule_padding_block(sch, name)
    
    schedule_reduction_fallback(sch, "g_cumsum")
    
    schedule_6d_batched_matmul(sch, "attn_mm", ["c1", "c2"], "kd")
    schedule_6d_batched_matmul(sch, "value_attn_vbeta_matmul", ["c", "vd"], "r")
    schedule_6d_batched_matmul(sch, "k_cumdecay_mm", ["c", "kd"], "r")
    # recurrent_state_mm_1 block removed - attention is computed fresh inside inter-chunk loop
    
    # Mark as scheduled and build
    mod["chunk_gated_delta"] = sch.mod["main"].with_attr("tir.is_scheduled", True)
    
    target_obj = tvm.target.Target(target)
    with tvm.transform.PassContext(config={"tir.disable_storage_rewrite": True}):
        built = tir.build(mod, target=target_obj)
    return built


def get_intermediate_buffer_shapes(
    batch_size: int,
    seq_len: int,
    linear_num_value_heads: int,
    linear_value_head_dim: int,
    linear_key_head_dim: int,
    chunk_size: int = 64,
):
    """
    Helper function to compute shapes for all intermediate buffers.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        linear_num_value_heads: Number of value heads
        linear_value_head_dim: Value head dimension
        linear_key_head_dim: Key head dimension
        chunk_size: Chunk size (default: 64)
    
    Returns:
        Dictionary mapping buffer names to their shapes (as tuples)
    """
    import math
    
    # Compute padded length and number of chunks
    num_chunks = math.ceil(seq_len / chunk_size)
    padded_len = num_chunks * chunk_size
    
    shapes = {
        # Transposed/padded buffers
        "query_T": (batch_size, linear_num_value_heads, padded_len, linear_key_head_dim),
        "key_T": (batch_size, linear_num_value_heads, padded_len, linear_key_head_dim),
        "k_beta": (batch_size, linear_num_value_heads, padded_len, linear_key_head_dim),
        "value_T": (batch_size, linear_num_value_heads, padded_len, linear_value_head_dim),
        "v_beta": (batch_size, linear_num_value_heads, padded_len, linear_value_head_dim),
        "g_T": (batch_size, linear_num_value_heads, padded_len),
        "beta_T": (batch_size, linear_num_value_heads, padded_len),
        
        # Chunked buffers
        "query_chunked": (batch_size, linear_num_value_heads, num_chunks, chunk_size, linear_key_head_dim),
        "key_chunked": (batch_size, linear_num_value_heads, num_chunks, chunk_size, linear_key_head_dim),
        "value_chunked": (batch_size, linear_num_value_heads, num_chunks, chunk_size, linear_value_head_dim),
        "k_beta_chunked": (batch_size, linear_num_value_heads, num_chunks, chunk_size, linear_key_head_dim),
        "v_beta_chunked": (batch_size, linear_num_value_heads, num_chunks, chunk_size, linear_value_head_dim),
        "g_chunked": (batch_size, linear_num_value_heads, num_chunks, chunk_size),
        "g_cumsum": (batch_size, linear_num_value_heads, num_chunks, chunk_size),
        
        # Attention buffers
        "decay_mask": (batch_size, linear_num_value_heads, num_chunks, chunk_size, chunk_size),
        "attn_mm_out": (batch_size, linear_num_value_heads, num_chunks, chunk_size, chunk_size),
        "attn_decay_neg_mask_out": (batch_size, linear_num_value_heads, num_chunks, chunk_size, chunk_size),
        "attn_associative_scan_out": (batch_size, linear_num_value_heads, num_chunks, chunk_size, chunk_size),
        "attn_identity_add_out": (batch_size, linear_num_value_heads, num_chunks, chunk_size, chunk_size),
        
        # Matmul/computation buffers
        "value_attn_vbeta_matmul_out": (batch_size, linear_num_value_heads, num_chunks, chunk_size, linear_value_head_dim),
        "k_beta_x_g_cumsum_exp_tmp": (batch_size, linear_num_value_heads, num_chunks, chunk_size, linear_key_head_dim),
        "k_cumdecay": (batch_size, linear_num_value_heads, num_chunks, chunk_size, linear_key_head_dim),
        
        # Recurrent state update buffers
        "recurrent_state_update_attn_out": (batch_size, linear_num_value_heads, num_chunks, chunk_size, chunk_size),
        "recurrent_state_update_v_buf": (batch_size, linear_num_value_heads, num_chunks, chunk_size, linear_value_head_dim),
        "recurrent_state_update_attn_inter": (batch_size, linear_num_value_heads, num_chunks, chunk_size, linear_value_head_dim),
        
        # Output buffer
        "core_attn_out_inter": (batch_size, linear_num_value_heads, num_chunks, chunk_size, linear_value_head_dim),
    }
    
    return shapes


# ============================================================================
# Development/testing code (run with: python gdn_kernel_dev.py)
# ============================================================================

if __name__ == "__main__":
    # Example usage: build and test the kernels
    print("=== Building kernels for development ===\n")
    
    # Build causal conv1d
    print("Building causal_conv1d_update...")
    try:
        conv1d_kernel = build_causal_conv1d_update(
            hidden_size=128,
            state_len=3,
            has_bias=True,
            activation="silu",
        )
        print("✓ causal_conv1d_update built successfully\n")
        print("=== Generated CUDA source ===")
        cuda_src = conv1d_kernel.imports[0].inspect_source()
        print(cuda_src)
        print("...\n")

    except Exception as e:
        print(f"✗ causal_conv1d_update failed: {e}\n")
    
    # Build chunk gated delta rule
    print("Building chunk_gated_delta_rule...")
    try:
        chunk_kernel = build_chunk_gated_delta_rule(
            linear_num_value_heads=32,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_key_head_dim=128,
            chunk_size=64,
        )
        print("✓ chunk_gated_delta_rule built successfully\n")
        
        # Print generated CUDA source for inspection
        print("=== Generated CUDA source ===")
        cuda_src = chunk_kernel.imports[0].inspect_source()
        print(cuda_src)
        print("...\n")
    except Exception as e:
        print(f"✗ chunk_gated_delta_rule failed: {e}\n")
