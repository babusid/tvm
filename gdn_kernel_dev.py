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
                vb, vh, vl = T.axis.remap("SSS", [b,h,l])
                hidden_states_new[vb,vh,vl] = conv_state_buf[vb,vh,vl]

        for b, h, l in T.grid(batch_size, T.int64(_hidden_size), seq_len):
            with T.sblock("copy_new_hidden_state_to_total_hidden_state"):
                vb, vh, vl = T.axis.remap("SSS", [b,h,l])
                hidden_states_new[vb,vh,vl + T.int64(_state_len)] = hidden_states_buf[vb,vh,vl]

        # Step 2: Extract last state_len elements as next_conv_state
        for b, h, s in T.grid(batch_size, T.int64(_hidden_size), T.int64(_state_len)):
            with T.sblock("create_next_conv_state"):
                vb, vh, vs = T.axis.remap("SSS", [b,h,s])
                next_conv_state_buf[vb, vh, vs] = hidden_states_new[vb, vh, vs + seq_len]

        # Step 3: Compute causal convolution (sliding window)
        # For each position s in output, convolve with state_len window ending at s+state_len
        for b, h, s, kw in T.grid(batch_size, T.int64(_hidden_size), seq_len, T.int64(_state_len)):
            with T.sblock("compute_causal_conv"):
                vb, vh, vs, vkw = T.axis.remap("SSSR", [b,h,s,kw])
                with T.init():
                    conv_sum[vb, vh, vs] = T.float32(0.0)
                conv_sum[vb, vh, vs] += hidden_states_new[vb, vh, vs + vkw + T.int64(1)] * weight[vh, T.int64(0), vkw]

        # Step 4: Add bias and apply activation
        if _activation == 'silu':
            for b, h, s in T.grid(batch_size, T.int64(_hidden_size), seq_len):
                with T.sblock("apply_bias_activation_and_writeback"):
                    vb, vh, vs = T.axis.remap("SSS", [b,h,s])
                    biased_val = conv_sum[vb, vh, vs] + bias[vh]
                    val_float = T.cast(biased_val, "float32")
                    out_buf[vb, vh, vs] = T.cast(val_float * T.sigmoid(val_float), _dtype)
        else:
            for b, h, s in T.grid(batch_size, T.int64(_hidden_size), seq_len):
                with T.sblock("apply_bias_and_writeback"):
                    vb, vh, vs = T.axis.remap("SSS", [b,h,s])
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
                vb, vh, vl = T.axis.remap("SSS", [b,h,l])
                hidden_states_new[vb,vh,vl] = conv_state_buf[vb,vh,vl]

        for b, h, l in T.grid(batch_size, T.int64(_hidden_size), seq_len):
            with T.sblock("copy_new_hidden_state_to_total_hidden_state"):
                vb, vh, vl = T.axis.remap("SSS", [b,h,l])
                hidden_states_new[vb,vh,vl + T.int64(_state_len)] = hidden_states_buf[vb,vh,vl]

        # Step 2: Extract last state_len elements as next_conv_state
        for b, h, s in T.grid(batch_size, T.int64(_hidden_size), T.int64(_state_len)):
            with T.sblock("create_next_conv_state"):
                vb, vh, vs = T.axis.remap("SSS", [b,h,s])
                next_conv_state_buf[vb, vh, vs] = hidden_states_new[vb, vh, vs + seq_len]

        # Step 3: Compute causal convolution (sliding window)
        # For each position s in output, convolve with state_len window ending at s+state_len
        for b, h, s, kw in T.grid(batch_size, T.int64(_hidden_size), seq_len, T.int64(_state_len)):
            with T.sblock("compute_causal_conv"):
                vb, vh, vs, vkw = T.axis.remap("SSSR", [b,h,s,kw])
                with T.init():
                    conv_sum[vb, vh, vs] = T.float32(0.0)
                conv_sum[vb, vh, vs] += hidden_states_new[vb, vh, vs + vkw + T.int64(1)] * weight[vh, T.int64(0), vkw]

        # Step 5: Apply activation (no bias)
        if _activation == 'silu':
            for b, h, s in T.grid(batch_size, T.int64(_hidden_size), seq_len):
                with T.sblock("apply_activation_and_writeback"):
                    vb, vh, vs = T.axis.remap("SSS", [b,h,s])
                    val = conv_sum[vb, vh, vs]
                    # Apply activation if enabled
                    # SiLU(x) = x * sigmoid(x)
                    val_float = T.cast(val, "float32")
                    out_buf[vb, vh, vs] = T.cast(val_float * T.sigmoid(val_float), _dtype)
        else:
            for b, h, s in T.grid(batch_size, T.int64(_hidden_size), seq_len):
                with T.sblock("writeback"):
                    vb, vh, vs = T.axis.remap("SSS", [b,h,s])
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
    Generate a TIR primfunc for chunked gated delta rule (prefill path).

    Refactored from TE/TOPI to pure TVMScript @T.prim_func to get explicit
    control over loop serialization in the prefix scan. The prefix scan has
    a sequential dependency: row i depends on the fully-computed output of
    rows 0..i-1. The prior TE-based te.extern approach with nested IRBuilder
    T.serial loops produced broken IR where the inner accumulation loop was
    compiled away to a no-op.

    Strategy for prefix scan:
      - Use T.grid(batch, heads, chunks) for independent dimensions
      - Use bare range() loops for the sequential row scan (i) and inner (j, k)
      - Accumulate directly into output buffer locations to avoid the
        local-variable-mutation bug (where TVMScript creates new let-bindings
        instead of mutating).

    Reference: qwen3_gdn.py torch_chunk_gated_delta_rule lines 146-263
    """
    # Capture compile-time constants
    _num_heads = linear_num_value_heads
    _key_dim = linear_key_head_dim
    _val_dim = linear_value_head_dim
    _C = chunk_size
    _dtype = dtype
    _scale = 1.0 / math.sqrt(linear_key_head_dim)
    _eps = 1e-6

    if use_qk_l2norm_in_kernel:
        @T.prim_func
        def chunk_gated_delta_rule(
            var_query: T.handle,
            var_key: T.handle,
            var_value: T.handle,
            var_g: T.handle,
            var_beta: T.handle,
            # Outputs
            var_out_query: T.handle,
            var_out_key: T.handle,
            var_out_value: T.handle,
            out_mask: T.Buffer((T.int64(_C), T.int64(_C)), "int32"),
            var_out_g_cumsum: T.handle,
            var_out_decay_mask: T.handle,
            var_out_attn: T.handle,
        ):
            T.func_attr({"global_symbol": "chunk_gated_delta_rule", "tir.noalias": True})

            # --- Symbolic shape variables ---
            # All declared as bare T.int64() so match_buffer and alloc_buffer
            # stay at PrimFunc top (no LetFrames pushed).
            # The caller must pass buffers with:
            #   num_chunks = ceildiv(seq_len, chunk_size)
            # total_len is not a separate symbolic var; we use
            # num_chunks * chunk_size wherever total_len was needed.
            batch_size = T.int64()
            seq_len = T.int64()
            num_chunks = T.int64()

            # --- Input buffers ---
            query = T.match_buffer(var_query, (batch_size, seq_len, T.int64(_num_heads), T.int64(_key_dim)), _dtype)
            key = T.match_buffer(var_key, (batch_size, seq_len, T.int64(_num_heads), T.int64(_key_dim)), _dtype)
            value = T.match_buffer(var_value, (batch_size, seq_len, T.int64(_num_heads), T.int64(_val_dim)), _dtype)
            g_in = T.match_buffer(var_g, (batch_size, seq_len, T.int64(_num_heads)), _dtype)
            beta_in = T.match_buffer(var_beta, (batch_size, seq_len, T.int64(_num_heads)), _dtype)

            # --- Output buffers ---
            out_query = T.match_buffer(var_out_query, (batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_key_dim)), _dtype)
            out_key = T.match_buffer(var_out_key, (batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_key_dim)), _dtype)
            out_value = T.match_buffer(var_out_value, (batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_val_dim)), _dtype)
            out_g_cumsum = T.match_buffer(var_out_g_cumsum, (batch_size, T.int64(_num_heads), num_chunks, T.int64(_C)), _dtype)
            out_decay_mask = T.match_buffer(var_out_decay_mask, (batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)), _dtype)
            out_attn = T.match_buffer(var_out_attn, (batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)), _dtype)

            # --- Intermediate buffers ---
            # Transposed + padded inputs: (batch, heads, num_chunks*C, dim)
            q_tp = T.alloc_buffer((batch_size, T.int64(_num_heads), num_chunks * T.int64(_C), T.int64(_key_dim)), _dtype)
            k_tp = T.alloc_buffer((batch_size, T.int64(_num_heads), num_chunks * T.int64(_C), T.int64(_key_dim)), _dtype)
            v_tp = T.alloc_buffer((batch_size, T.int64(_num_heads), num_chunks * T.int64(_C), T.int64(_val_dim)), _dtype)
            beta_tp = T.alloc_buffer((batch_size, T.int64(_num_heads), num_chunks * T.int64(_C)), _dtype)
            g_tp = T.alloc_buffer((batch_size, T.int64(_num_heads), num_chunks * T.int64(_C)), _dtype)

            # L2 norm intermediates
            q_norm_sq_sum = T.alloc_buffer((batch_size, seq_len, T.int64(_num_heads)), _dtype)
            k_norm_sq_sum = T.alloc_buffer((batch_size, seq_len, T.int64(_num_heads)), _dtype)

            # Chunked intermediates
            g_cumsum = T.alloc_buffer((batch_size, T.int64(_num_heads), num_chunks, T.int64(_C)), _dtype)
            k_beta_chunk = T.alloc_buffer((batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_key_dim)), _dtype)
            attn_raw = T.alloc_buffer((batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)), _dtype)
            attn_masked = T.alloc_buffer((batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)), _dtype)

            # ================================================================
            # Step 1: L2 normalize query and key
            # ================================================================
            # Compute sum of squares per (batch, seq, head)
            for b, s, n in T.grid(batch_size, seq_len, T.int64(_num_heads)):
                with T.sblock("q_norm_init"):
                    vb = T.axis.spatial(batch_size, b)
                    vs = T.axis.spatial(seq_len, s)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    q_norm_sq_sum[vb, vs, vn] = T.float32(0)
                    k_norm_sq_sum[vb, vs, vn] = T.float32(0)
            for b, s, n, d in T.grid(batch_size, seq_len, T.int64(_num_heads), T.int64(_key_dim)):
                with T.sblock("q_norm_update"):
                    vb = T.axis.spatial(batch_size, b)
                    vs = T.axis.spatial(seq_len, s)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vd = T.axis.reduce(T.int64(_key_dim), d)
                    q_norm_sq_sum[vb, vs, vn] = q_norm_sq_sum[vb, vs, vn] + query[vb, vs, vn, vd] * query[vb, vs, vn, vd]
                    k_norm_sq_sum[vb, vs, vn] = k_norm_sq_sum[vb, vs, vn] + key[vb, vs, vn, vd] * key[vb, vs, vn, vd]

            # ================================================================
            # Step 2: Transpose (b,s,n,d) -> (b,n,s,d), apply L2 norm, scale
            #         query, and pad to chunk boundary. Also transpose+pad
            #         beta and g from (b,s,n) -> (b,n,s).
            # ================================================================
            for b, n, s, d in T.grid(batch_size, T.int64(_num_heads), num_chunks * T.int64(_C), T.int64(_key_dim)):
                with T.sblock("transpose_qk"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vs = T.axis.spatial(num_chunks * T.int64(_C), s)
                    vd = T.axis.spatial(T.int64(_key_dim), d)
                    if vs < seq_len:
                        q_tp[vb, vn, vs, vd] = query[vb, vs, vn, vd] * T.rsqrt(q_norm_sq_sum[vb, vs, vn] + T.float32(_eps)) * T.float32(_scale)
                        k_tp[vb, vn, vs, vd] = key[vb, vs, vn, vd] * T.rsqrt(k_norm_sq_sum[vb, vs, vn] + T.float32(_eps))
                    else:
                        q_tp[vb, vn, vs, vd] = T.float32(0)
                        k_tp[vb, vn, vs, vd] = T.float32(0)

            for b, n, s, d in T.grid(batch_size, T.int64(_num_heads), num_chunks * T.int64(_C), T.int64(_val_dim)):
                with T.sblock("transpose_v"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vs = T.axis.spatial(num_chunks * T.int64(_C), s)
                    vd = T.axis.spatial(T.int64(_val_dim), d)
                    if vs < seq_len:
                        v_tp[vb, vn, vs, vd] = value[vb, vs, vn, vd]
                    else:
                        v_tp[vb, vn, vs, vd] = T.float32(0)

            for b, n, s in T.grid(batch_size, T.int64(_num_heads), num_chunks * T.int64(_C)):
                with T.sblock("transpose_beta_g"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vs = T.axis.spatial(num_chunks * T.int64(_C), s)
                    if vs < seq_len:
                        beta_tp[vb, vn, vs] = beta_in[vb, vs, vn]
                        g_tp[vb, vn, vs] = g_in[vb, vs, vn]
                    else:
                        beta_tp[vb, vn, vs] = T.float32(0)
                        g_tp[vb, vn, vs] = T.float32(0)

            # ================================================================
            # Step 3: Reshape into chunks and compute k_beta.
            #         Logical reshape: (b, n, num_chunks*C, d) -> (b, n, num_chunks, C, d)
            #         index mapping: s = c * C + t
            # ================================================================
            for b, n, c, t, d in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_key_dim)):
                with T.sblock("reshape_qk_kbeta"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vc = T.axis.spatial(num_chunks, c)
                    vt = T.axis.spatial(T.int64(_C), t)
                    vd = T.axis.spatial(T.int64(_key_dim), d)
                    out_query[vb, vn, vc, vt, vd] = q_tp[vb, vn, vc * T.int64(_C) + vt, vd]
                    out_key[vb, vn, vc, vt, vd] = k_tp[vb, vn, vc * T.int64(_C) + vt, vd]
                    k_beta_chunk[vb, vn, vc, vt, vd] = k_tp[vb, vn, vc * T.int64(_C) + vt, vd] * beta_tp[vb, vn, vc * T.int64(_C) + vt]

            for b, n, c, t, d in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_val_dim)):
                with T.sblock("reshape_v"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vc = T.axis.spatial(num_chunks, c)
                    vt = T.axis.spatial(T.int64(_C), t)
                    vd = T.axis.spatial(T.int64(_val_dim), d)
                    out_value[vb, vn, vc, vt, vd] = v_tp[vb, vn, vc * T.int64(_C) + vt, vd]

            # ================================================================
            # Step 4: Upper triangular mask (static, chunk_size x chunk_size)
            # ================================================================
            for i, j in T.grid(T.int64(_C), T.int64(_C)):
                with T.sblock("upper_tri_mask"):
                    vi = T.axis.spatial(T.int64(_C), i)
                    vj = T.axis.spatial(T.int64(_C), j)
                    out_mask[vi, vj] = T.Select(vj >= vi, 1, 0)

            # ================================================================
            # Step 5: Cumulative sum of g along chunk dim, then decay mask.
            #         g_cumsum[b,n,c,t] = sum(g[b,n,c,0..t])
            #         decay_mask[b,n,c,i,j] = exp(g_cumsum[i] - g_cumsum[j])
            #                                 if j <= i, else 0
            # ================================================================
            # Cumsum: sequential along t within each (b, n, c)
            for b, h, c in T.grid(batch_size, T.int64(_num_heads), num_chunks):
                with T.sblock("g_cumsum"):
                    vb = T.axis.spatial(batch_size, b)
                    vh = T.axis.spatial(T.int64(_num_heads), h)
                    vc = T.axis.spatial(num_chunks, c)
                    g_cumsum[vb, vh, vc, T.int64(0)] = g_tp[vb, vh, vc * T.int64(_C)]
                    for t in range(T.int64(_C) - T.int64(1)):
                        g_cumsum[vb, vh, vc, t + T.int64(1)] = g_cumsum[vb, vh, vc, t] + g_tp[vb, vh, vc * T.int64(_C) + t + T.int64(1)]

            # Copy cumsum to output
            for b, n, c, t in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C)):
                with T.sblock("g_cumsum_copy"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vc = T.axis.spatial(num_chunks, c)
                    vt = T.axis.spatial(T.int64(_C), t)
                    out_g_cumsum[vb, vn, vc, vt] = g_cumsum[vb, vn, vc, vt]

            # Decay mask: lower triangular exp(g_cumsum[i] - g_cumsum[j])
            for b, n, c, i, j in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)):
                with T.sblock("decay_mask"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vc = T.axis.spatial(num_chunks, c)
                    vi = T.axis.spatial(T.int64(_C), i)
                    vj = T.axis.spatial(T.int64(_C), j)
                    if vj <= vi:
                        out_decay_mask[vb, vn, vc, vi, vj] = T.exp(g_cumsum[vb, vn, vc, vi] - g_cumsum[vb, vn, vc, vj])
                    else:
                        out_decay_mask[vb, vn, vc, vi, vj] = T.float32(0)

            # ================================================================
            # Step 6: Attention = -(k_beta @ key^T) * decay_mask, masked by
            #         upper triangular (set diagonal and above to 0).
            #         attn[b,h,c,i,j] = sum_d(k_beta[i,d] * key[j,d]) for d in key_dim
            # ================================================================
            # Matmul: k_beta @ key^T
            for b, n, c, i, j in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)):
                with T.sblock("attn_mm_init"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vc = T.axis.spatial(num_chunks, c)
                    vi = T.axis.spatial(T.int64(_C), i)
                    vj = T.axis.spatial(T.int64(_C), j)
                    attn_raw[vb, vn, vc, vi, vj] = T.float32(0)
            for b, n, c, i, j, d in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C), T.int64(_key_dim)):
                with T.sblock("attn_mm_update"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vc = T.axis.spatial(num_chunks, c)
                    vi = T.axis.spatial(T.int64(_C), i)
                    vj = T.axis.spatial(T.int64(_C), j)
                    vd = T.axis.reduce(T.int64(_key_dim), d)
                    attn_raw[vb, vn, vc, vi, vj] = attn_raw[vb, vn, vc, vi, vj] + k_beta_chunk[vb, vn, vc, vi, vd] * out_key[vb, vn, vc, vj, vd]

            # Apply decay mask, zero upper triangle (mask[i,j]=1 when j>=i), negate
            for b, n, c, i, j in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)):
                with T.sblock("attn_mask_negate"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vc = T.axis.spatial(num_chunks, c)
                    vi = T.axis.spatial(T.int64(_C), i)
                    vj = T.axis.spatial(T.int64(_C), j)
                    if vj >= vi:
                        attn_masked[vb, vn, vc, vi, vj] = T.float32(0)
                    else:
                        attn_masked[vb, vn, vc, vi, vj] = -(attn_raw[vb, vn, vc, vi, vj] * out_decay_mask[vb, vn, vc, vi, vj])

            # ================================================================
            # Step 7: Associative prefix scan on attn_masked.
            #
            # PyTorch reference (sequential):
            #   for i in range(1, chunk_size):
            #       row = attn[..., i, :i].clone()
            #       sub = attn[..., :i, :i].clone()
            #       attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
            #
            # Row i depends on the fully-updated rows 0..i-1.
            # We parallelize over (batch, heads, chunks) and serialize over
            # the row dimension i. For each row i, column j < i:
            #   attn[i,j] += sum_{k=j+1}^{i-1} attn[i,k] * attn[k,j]
            # where attn[k,j] must already be updated (k < i).
            #
            # Outer (b,h,c) are spatial axes in a T.sblock so dlight can
            # bind them to GPU threads. The inner scan is sequential range().
            # ================================================================
            for b, h, c in T.grid(batch_size, T.int64(_num_heads), num_chunks):
                with T.sblock("prefix_scan"):
                    vb = T.axis.spatial(batch_size, b)
                    vh = T.axis.spatial(T.int64(_num_heads), h)
                    vc = T.axis.spatial(num_chunks, c)
                    # Row 0 is unchanged (no dependencies), copy it
                    for j in range(T.int64(_C)):
                        out_attn[vb, vh, vc, T.int64(0), j] = attn_masked[vb, vh, vc, T.int64(0), j]

                    # Rows 1..chunk_size-1: sequential scan
                    for i in range(T.int64(_C) - T.int64(1)):
                        # For j >= i+1 (upper triangle + diagonal): copy directly
                        for j in range(T.int64(_C) - (i + T.int64(1))):
                            out_attn[vb, vh, vc, i + T.int64(1), i + T.int64(1) + j] = attn_masked[vb, vh, vc, i + T.int64(1), i + T.int64(1) + j]

                        # For j < i+1 (lower triangle): apply prefix scan
                        for j in range(i + T.int64(1)):
                            out_attn[vb, vh, vc, i + T.int64(1), j] = attn_masked[vb, vh, vc, i + T.int64(1), j]
                            for k in range(j + T.int64(1), i + T.int64(1)):
                                out_attn[vb, vh, vc, i + T.int64(1), j] = out_attn[vb, vh, vc, i + T.int64(1), j] + attn_masked[vb, vh, vc, i + T.int64(1), k] * out_attn[vb, vh, vc, k, j]

        return chunk_gated_delta_rule
    else:
        @T.prim_func
        def chunk_gated_delta_rule_no_norm(
            var_query: T.handle,
            var_key: T.handle,
            var_value: T.handle,
            var_g: T.handle,
            var_beta: T.handle,
            # Outputs
            var_out_query: T.handle,
            var_out_key: T.handle,
            var_out_value: T.handle,
            out_mask: T.Buffer((T.int64(_C), T.int64(_C)), "int32"),
            var_out_g_cumsum: T.handle,
            var_out_decay_mask: T.handle,
            var_out_attn: T.handle,
        ):
            T.func_attr({"global_symbol": "chunk_gated_delta_rule", "tir.noalias": True})

            # --- Symbolic shape variables ---
            batch_size = T.int64()
            seq_len = T.int64()
            num_chunks = T.int64()

            # --- Input buffers ---
            query = T.match_buffer(var_query, (batch_size, seq_len, T.int64(_num_heads), T.int64(_key_dim)), _dtype)
            key = T.match_buffer(var_key, (batch_size, seq_len, T.int64(_num_heads), T.int64(_key_dim)), _dtype)
            value = T.match_buffer(var_value, (batch_size, seq_len, T.int64(_num_heads), T.int64(_val_dim)), _dtype)
            g_in = T.match_buffer(var_g, (batch_size, seq_len, T.int64(_num_heads)), _dtype)
            beta_in = T.match_buffer(var_beta, (batch_size, seq_len, T.int64(_num_heads)), _dtype)

            # --- Output buffers ---
            out_query = T.match_buffer(var_out_query, (batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_key_dim)), _dtype)
            out_key = T.match_buffer(var_out_key, (batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_key_dim)), _dtype)
            out_value = T.match_buffer(var_out_value, (batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_val_dim)), _dtype)
            out_g_cumsum = T.match_buffer(var_out_g_cumsum, (batch_size, T.int64(_num_heads), num_chunks, T.int64(_C)), _dtype)
            out_decay_mask = T.match_buffer(var_out_decay_mask, (batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)), _dtype)
            out_attn = T.match_buffer(var_out_attn, (batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)), _dtype)

            # --- Intermediate buffers ---
            q_tp = T.alloc_buffer((batch_size, T.int64(_num_heads), num_chunks * T.int64(_C), T.int64(_key_dim)), _dtype)
            k_tp = T.alloc_buffer((batch_size, T.int64(_num_heads), num_chunks * T.int64(_C), T.int64(_key_dim)), _dtype)
            v_tp = T.alloc_buffer((batch_size, T.int64(_num_heads), num_chunks * T.int64(_C), T.int64(_val_dim)), _dtype)
            beta_tp = T.alloc_buffer((batch_size, T.int64(_num_heads), num_chunks * T.int64(_C)), _dtype)
            g_tp = T.alloc_buffer((batch_size, T.int64(_num_heads), num_chunks * T.int64(_C)), _dtype)

            g_cumsum = T.alloc_buffer((batch_size, T.int64(_num_heads), num_chunks, T.int64(_C)), _dtype)
            k_beta_chunk = T.alloc_buffer((batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_key_dim)), _dtype)
            attn_raw = T.alloc_buffer((batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)), _dtype)
            attn_masked = T.alloc_buffer((batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)), _dtype)

            # ================================================================
            # Step 1: Transpose (b,s,n,d) -> (b,n,s,d), scale query, pad
            # ================================================================
            for b, n, s, d in T.grid(batch_size, T.int64(_num_heads), num_chunks * T.int64(_C), T.int64(_key_dim)):
                with T.sblock("transpose_qk"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vs = T.axis.spatial(num_chunks * T.int64(_C), s)
                    vd = T.axis.spatial(T.int64(_key_dim), d)
                    if vs < seq_len:
                        q_tp[vb, vn, vs, vd] = query[vb, vs, vn, vd] * T.float32(_scale)
                        k_tp[vb, vn, vs, vd] = key[vb, vs, vn, vd]
                    else:
                        q_tp[vb, vn, vs, vd] = T.float32(0)
                        k_tp[vb, vn, vs, vd] = T.float32(0)

            for b, n, s, d in T.grid(batch_size, T.int64(_num_heads), num_chunks * T.int64(_C), T.int64(_val_dim)):
                with T.sblock("transpose_v"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vs = T.axis.spatial(num_chunks * T.int64(_C), s)
                    vd = T.axis.spatial(T.int64(_val_dim), d)
                    if vs < seq_len:
                        v_tp[vb, vn, vs, vd] = value[vb, vs, vn, vd]
                    else:
                        v_tp[vb, vn, vs, vd] = T.float32(0)

            for b, n, s in T.grid(batch_size, T.int64(_num_heads), num_chunks * T.int64(_C)):
                with T.sblock("transpose_beta_g"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vs = T.axis.spatial(num_chunks * T.int64(_C), s)
                    if vs < seq_len:
                        beta_tp[vb, vn, vs] = beta_in[vb, vs, vn]
                        g_tp[vb, vn, vs] = g_in[vb, vs, vn]
                    else:
                        beta_tp[vb, vn, vs] = T.float32(0)
                        g_tp[vb, vn, vs] = T.float32(0)

            # ================================================================
            # Step 2: Reshape into chunks, compute k_beta
            # ================================================================
            for b, n, c, t, d in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_key_dim)):
                with T.sblock("reshape_qk_kbeta"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vc = T.axis.spatial(num_chunks, c)
                    vt = T.axis.spatial(T.int64(_C), t)
                    vd = T.axis.spatial(T.int64(_key_dim), d)
                    out_query[vb, vn, vc, vt, vd] = q_tp[vb, vn, vc * T.int64(_C) + vt, vd]
                    out_key[vb, vn, vc, vt, vd] = k_tp[vb, vn, vc * T.int64(_C) + vt, vd]
                    k_beta_chunk[vb, vn, vc, vt, vd] = k_tp[vb, vn, vc * T.int64(_C) + vt, vd] * beta_tp[vb, vn, vc * T.int64(_C) + vt]

            for b, n, c, t, d in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_val_dim)):
                with T.sblock("reshape_v"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vc = T.axis.spatial(num_chunks, c)
                    vt = T.axis.spatial(T.int64(_C), t)
                    vd = T.axis.spatial(T.int64(_val_dim), d)
                    out_value[vb, vn, vc, vt, vd] = v_tp[vb, vn, vc * T.int64(_C) + vt, vd]

            # ================================================================
            # Step 3: Upper triangular mask
            # ================================================================
            for i, j in T.grid(T.int64(_C), T.int64(_C)):
                with T.sblock("upper_tri_mask"):
                    vi = T.axis.spatial(T.int64(_C), i)
                    vj = T.axis.spatial(T.int64(_C), j)
                    out_mask[vi, vj] = T.Select(vj >= vi, 1, 0)

            # ================================================================
            # Step 4: Cumsum of g, decay mask
            # ================================================================
            for b, h, c in T.grid(batch_size, T.int64(_num_heads), num_chunks):
                with T.sblock("g_cumsum"):
                    vb = T.axis.spatial(batch_size, b)
                    vh = T.axis.spatial(T.int64(_num_heads), h)
                    vc = T.axis.spatial(num_chunks, c)
                    g_cumsum[vb, vh, vc, T.int64(0)] = g_tp[vb, vh, vc * T.int64(_C)]
                    for t in range(T.int64(_C) - T.int64(1)):
                        g_cumsum[vb, vh, vc, t + T.int64(1)] = g_cumsum[vb, vh, vc, t] + g_tp[vb, vh, vc * T.int64(_C) + t + T.int64(1)]

            for b, n, c, t in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C)):
                with T.sblock("g_cumsum_copy"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vc = T.axis.spatial(num_chunks, c)
                    vt = T.axis.spatial(T.int64(_C), t)
                    out_g_cumsum[vb, vn, vc, vt] = g_cumsum[vb, vn, vc, vt]

            for b, n, c, i, j in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)):
                with T.sblock("decay_mask"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vc = T.axis.spatial(num_chunks, c)
                    vi = T.axis.spatial(T.int64(_C), i)
                    vj = T.axis.spatial(T.int64(_C), j)
                    if vj <= vi:
                        out_decay_mask[vb, vn, vc, vi, vj] = T.exp(g_cumsum[vb, vn, vc, vi] - g_cumsum[vb, vn, vc, vj])
                    else:
                        out_decay_mask[vb, vn, vc, vi, vj] = T.float32(0)

            # ================================================================
            # Step 5: Attention matmul + mask + negate
            # ================================================================
            for b, n, c, i, j in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)):
                with T.sblock("attn_mm_init"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vc = T.axis.spatial(num_chunks, c)
                    vi = T.axis.spatial(T.int64(_C), i)
                    vj = T.axis.spatial(T.int64(_C), j)
                    attn_raw[vb, vn, vc, vi, vj] = T.float32(0)
            for b, n, c, i, j, d in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C), T.int64(_key_dim)):
                with T.sblock("attn_mm_update"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vc = T.axis.spatial(num_chunks, c)
                    vi = T.axis.spatial(T.int64(_C), i)
                    vj = T.axis.spatial(T.int64(_C), j)
                    vd = T.axis.reduce(T.int64(_key_dim), d)
                    attn_raw[vb, vn, vc, vi, vj] = attn_raw[vb, vn, vc, vi, vj] + k_beta_chunk[vb, vn, vc, vi, vd] * out_key[vb, vn, vc, vj, vd]

            for b, n, c, i, j in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)):
                with T.sblock("attn_mask_negate"):
                    vb = T.axis.spatial(batch_size, b)
                    vn = T.axis.spatial(T.int64(_num_heads), n)
                    vc = T.axis.spatial(num_chunks, c)
                    vi = T.axis.spatial(T.int64(_C), i)
                    vj = T.axis.spatial(T.int64(_C), j)
                    if vj >= vi:
                        attn_masked[vb, vn, vc, vi, vj] = T.float32(0)
                    else:
                        attn_masked[vb, vn, vc, vi, vj] = -(attn_raw[vb, vn, vc, vi, vj] * out_decay_mask[vb, vn, vc, vi, vj])

            # ================================================================
            # Step 6: Prefix scan (see L2 norm variant for detailed comments)
            # ================================================================
            for b, h, c in T.grid(batch_size, T.int64(_num_heads), num_chunks):
                with T.sblock("prefix_scan"):
                    vb = T.axis.spatial(batch_size, b)
                    vh = T.axis.spatial(T.int64(_num_heads), h)
                    vc = T.axis.spatial(num_chunks, c)
                    for j in range(T.int64(_C)):
                        out_attn[vb, vh, vc, T.int64(0), j] = attn_masked[vb, vh, vc, T.int64(0), j]

                    for i in range(T.int64(_C) - T.int64(1)):
                        for j in range(T.int64(_C) - (i + T.int64(1))):
                            out_attn[vb, vh, vc, i + T.int64(1), i + T.int64(1) + j] = attn_masked[vb, vh, vc, i + T.int64(1), i + T.int64(1) + j]

                        for j in range(i + T.int64(1)):
                            out_attn[vb, vh, vc, i + T.int64(1), j] = attn_masked[vb, vh, vc, i + T.int64(1), j]
                            for k in range(j + T.int64(1), i + T.int64(1)):
                                out_attn[vb, vh, vc, i + T.int64(1), j] = out_attn[vb, vh, vc, i + T.int64(1), j] + attn_masked[vb, vh, vc, i + T.int64(1), k] * out_attn[vb, vh, vc, k, j]

        return chunk_gated_delta_rule_no_norm


def _chunk_recurrent_gated_delta_rule():
    inputs = []
    outputs = []
    return te.create_prim_func(inputs + outputs)


#print(_gen_causal_conv1d_update(128, 128).script())
#print(_chunk_gated_delta_rule(32, 128, 16, 128, use_qk_l2norm_in_kernel=False))
#print(_chunk_gated_delta_rule(32, 128, 16, 128, use_qk_l2norm_in_kernel=True))
#

mod = tvm.IRModule(
    {
        #"conv1d_biased_act": _gen_causal_conv1d_update(128, 128, has_bias=True),
        #"conv1d_biased": _gen_causal_conv1d_update(128, 128, has_bias=True, activation=None),
        "conv1d_nobias_act": _gen_causal_conv1d_update(128, 128, has_bias=False),
        #"conv1d_nobias": _gen_causal_conv1d_update(128, 128, has_bias=False, activation=None),
        #"chunk_gated_delta_l2_norm": _chunk_gated_delta_rule(
        #           32, 128, 16, 128, use_qk_l2norm_in_kernel=True
        #       ),
        #       "chunk_gated_delta": _chunk_gated_delta_rule(
        #           32, 128, 16, 128, use_qk_l2norm_in_kernel=False
        #       ),
    }
)

mod.show()

#from tvm import dlight as dl
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
    # print(built.imports[0].inspect_source())
except Exception as e:
    print(f"BUILD FAILED: {type(e).__name__}: {e}")
