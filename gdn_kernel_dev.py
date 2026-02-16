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
    """
    Generate a TIR primfunc for chunked gated delta rule (prefill path).

    Refactored from TE/TOPI to pure TVMScript @T.prim_func to get explicit
    control over loop serialization in the prefix scan. The prefix scan has
    a sequential dependency: row i depends on the fully-computed output of
    rows 0..i-1. The prior TE-based te.extern approach with nested IRBuilder
    T.serial loops produced broken IR where the inner accumulation loop was
    compiled away to a no-op.

    Strategy for prefix scan:
      - Fuse (batch, heads, chunks) into a single T.parallel loop
      - Use T.serial for the row loop (i) and bare range loops for (j, k)
      - This mirrors the pattern used by topi.cumsum (scan.py) which TVM
        correctly lowers: T.parallel for independent dims, T.serial for
        the sequential scan dimension.

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
                q_norm_sq_sum[b, s, n] = T.float32(0)
                k_norm_sq_sum[b, s, n] = T.float32(0)
            for b, s, n, d in T.grid(batch_size, seq_len, T.int64(_num_heads), T.int64(_key_dim)):
                q_norm_sq_sum[b, s, n] = q_norm_sq_sum[b, s, n] + query[b, s, n, d] * query[b, s, n, d]
                k_norm_sq_sum[b, s, n] = k_norm_sq_sum[b, s, n] + key[b, s, n, d] * key[b, s, n, d]

            # ================================================================
            # Step 2: Transpose (b,s,n,d) -> (b,n,s,d), apply L2 norm, scale
            #         query, and pad to chunk boundary. Also transpose+pad
            #         beta and g from (b,s,n) -> (b,n,s).
            # ================================================================
            for b, n, s, d in T.grid(batch_size, T.int64(_num_heads), num_chunks * T.int64(_C), T.int64(_key_dim)):
                if s < seq_len:
                    q_tp[b, n, s, d] = query[b, s, n, d] * T.rsqrt(q_norm_sq_sum[b, s, n] + T.float32(_eps)) * T.float32(_scale)
                    k_tp[b, n, s, d] = key[b, s, n, d] * T.rsqrt(k_norm_sq_sum[b, s, n] + T.float32(_eps))
                else:
                    q_tp[b, n, s, d] = T.float32(0)
                    k_tp[b, n, s, d] = T.float32(0)

            for b, n, s, d in T.grid(batch_size, T.int64(_num_heads), num_chunks * T.int64(_C), T.int64(_val_dim)):
                if s < seq_len:
                    v_tp[b, n, s, d] = value[b, s, n, d]
                else:
                    v_tp[b, n, s, d] = T.float32(0)

            for b, n, s in T.grid(batch_size, T.int64(_num_heads), num_chunks * T.int64(_C)):
                if s < seq_len:
                    beta_tp[b, n, s] = beta_in[b, s, n]
                    g_tp[b, n, s] = g_in[b, s, n]
                else:
                    beta_tp[b, n, s] = T.float32(0)
                    g_tp[b, n, s] = T.float32(0)

            # ================================================================
            # Step 3: Reshape into chunks and compute k_beta.
            #         Logical reshape: (b, n, num_chunks*C, d) -> (b, n, num_chunks, C, d)
            #         index mapping: s = c * C + t
            # ================================================================
            for b, n, c, t, d in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_key_dim)):
                out_query[b, n, c, t, d] = q_tp[b, n, c * T.int64(_C) + t, d]
                out_key[b, n, c, t, d] = k_tp[b, n, c * T.int64(_C) + t, d]
                k_beta_chunk[b, n, c, t, d] = k_tp[b, n, c * T.int64(_C) + t, d] * beta_tp[b, n, c * T.int64(_C) + t]

            for b, n, c, t, d in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_val_dim)):
                out_value[b, n, c, t, d] = v_tp[b, n, c * T.int64(_C) + t, d]

            # ================================================================
            # Step 4: Upper triangular mask (static, chunk_size x chunk_size)
            # ================================================================
            for i, j in T.grid(T.int64(_C), T.int64(_C)):
                out_mask[i, j] = T.Select(j >= i, 1, 0)

            # ================================================================
            # Step 5: Cumulative sum of g along chunk dim, then decay mask.
            #         g_cumsum[b,n,c,t] = sum(g[b,n,c,0..t])
            #         decay_mask[b,n,c,i,j] = exp(g_cumsum[i] - g_cumsum[j])
            #                                 if j <= i, else 0
            # ================================================================
            # Cumsum: sequential along t within each (b, n, c)
            # Use the same T.parallel + T.serial pattern as topi.cumsum
            for bhc in T.parallel(batch_size * T.int64(_num_heads) * num_chunks):
                # Decompose flat index
                b: T.int64 = bhc // (T.int64(_num_heads) * num_chunks)
                h: T.int64 = bhc % (T.int64(_num_heads) * num_chunks) // num_chunks
                c: T.int64 = bhc % num_chunks

                g_cumsum[b, h, c, T.int64(0)] = g_tp[b, h, c * T.int64(_C)]
                for t in T.serial(T.int64(_C) - T.int64(1)):
                    g_cumsum[b, h, c, t + T.int64(1)] = g_cumsum[b, h, c, t] + g_tp[b, h, c * T.int64(_C) + t + T.int64(1)]

            # Copy cumsum to output
            for b, n, c, t in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C)):
                out_g_cumsum[b, n, c, t] = g_cumsum[b, n, c, t]

            # Decay mask: lower triangular exp(g_cumsum[i] - g_cumsum[j])
            for b, n, c, i, j in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)):
                if j <= i:
                    out_decay_mask[b, n, c, i, j] = T.exp(g_cumsum[b, n, c, i] - g_cumsum[b, n, c, j])
                else:
                    out_decay_mask[b, n, c, i, j] = T.float32(0)

            # ================================================================
            # Step 6: Attention = -(k_beta @ key^T) * decay_mask, masked by
            #         upper triangular (set diagonal and above to 0).
            #         attn[b,h,c,i,j] = sum_d(k_beta[i,d] * key[j,d]) for d in key_dim
            # ================================================================
            # Matmul: k_beta @ key^T
            for b, n, c, i, j in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)):
                attn_raw[b, n, c, i, j] = T.float32(0)
            for b, n, c, i, j, d in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C), T.int64(_key_dim)):
                attn_raw[b, n, c, i, j] = attn_raw[b, n, c, i, j] + k_beta_chunk[b, n, c, i, d] * out_key[b, n, c, j, d]

            # Apply decay mask, zero upper triangle (mask[i,j]=1 when j>=i), negate
            for b, n, c, i, j in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)):
                if j >= i:
                    # Upper triangle including diagonal: set to 0
                    attn_masked[b, n, c, i, j] = T.float32(0)
                else:
                    # Strictly lower triangle: -(attn * decay_mask)
                    attn_masked[b, n, c, i, j] = -(attn_raw[b, n, c, i, j] * out_decay_mask[b, n, c, i, j])

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
            # Uses T.parallel + T.serial mirroring topi.cumsum pattern.
            # ================================================================
            for bhc in T.parallel(batch_size * T.int64(_num_heads) * num_chunks):
                b: T.int64 = bhc // (T.int64(_num_heads) * num_chunks)
                h: T.int64 = bhc % (T.int64(_num_heads) * num_chunks) // num_chunks
                c: T.int64 = bhc % num_chunks

                # Row 0 is unchanged (no dependencies), copy it
                for j in range(T.int64(_C)):
                    out_attn[b, h, c, T.int64(0), j] = attn_masked[b, h, c, T.int64(0), j]

                # Rows 1..chunk_size-1: sequential scan
                for i in T.serial(T.int64(_C) - T.int64(1)):
                    row: T.int64 = i + T.int64(1)

                    # For j >= row (upper triangle + diagonal): copy directly
                    for j in range(T.int64(_C) - row):
                        out_attn[b, h, c, row, row + j] = attn_masked[b, h, c, row, row + j]

                    # For j < row (lower triangle): apply prefix scan
                    for j in range(row):
                        # Initialize with original value, then accumulate in-place
                        out_attn[b, h, c, row, j] = attn_masked[b, h, c, row, j]
                        # Accumulate: sum over k in (j+1..row-1)
                        # attn_masked[row, k] * out_attn[k, j]  (out_attn[k,j] already updated since k < row)
                        for k in range(j + T.int64(1), row):
                            out_attn[b, h, c, row, j] = out_attn[b, h, c, row, j] + attn_masked[b, h, c, row, k] * out_attn[b, h, c, k, j]

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
                if s < seq_len:
                    q_tp[b, n, s, d] = query[b, s, n, d] * T.float32(_scale)
                    k_tp[b, n, s, d] = key[b, s, n, d]
                else:
                    q_tp[b, n, s, d] = T.float32(0)
                    k_tp[b, n, s, d] = T.float32(0)

            for b, n, s, d in T.grid(batch_size, T.int64(_num_heads), num_chunks * T.int64(_C), T.int64(_val_dim)):
                if s < seq_len:
                    v_tp[b, n, s, d] = value[b, s, n, d]
                else:
                    v_tp[b, n, s, d] = T.float32(0)

            for b, n, s in T.grid(batch_size, T.int64(_num_heads), num_chunks * T.int64(_C)):
                if s < seq_len:
                    beta_tp[b, n, s] = beta_in[b, s, n]
                    g_tp[b, n, s] = g_in[b, s, n]
                else:
                    beta_tp[b, n, s] = T.float32(0)
                    g_tp[b, n, s] = T.float32(0)

            # ================================================================
            # Step 2: Reshape into chunks, compute k_beta
            # ================================================================
            for b, n, c, t, d in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_key_dim)):
                out_query[b, n, c, t, d] = q_tp[b, n, c * T.int64(_C) + t, d]
                out_key[b, n, c, t, d] = k_tp[b, n, c * T.int64(_C) + t, d]
                k_beta_chunk[b, n, c, t, d] = k_tp[b, n, c * T.int64(_C) + t, d] * beta_tp[b, n, c * T.int64(_C) + t]

            for b, n, c, t, d in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_val_dim)):
                out_value[b, n, c, t, d] = v_tp[b, n, c * T.int64(_C) + t, d]

            # ================================================================
            # Step 3: Upper triangular mask
            # ================================================================
            for i, j in T.grid(T.int64(_C), T.int64(_C)):
                out_mask[i, j] = T.Select(j >= i, 1, 0)

            # ================================================================
            # Step 4: Cumsum of g, decay mask
            # ================================================================
            for bhc in T.parallel(batch_size * T.int64(_num_heads) * num_chunks):
                b: T.int64 = bhc // (T.int64(_num_heads) * num_chunks)
                h: T.int64 = bhc % (T.int64(_num_heads) * num_chunks) // num_chunks
                c: T.int64 = bhc % num_chunks

                g_cumsum[b, h, c, T.int64(0)] = g_tp[b, h, c * T.int64(_C)]
                for t in T.serial(T.int64(_C) - T.int64(1)):
                    g_cumsum[b, h, c, t + T.int64(1)] = g_cumsum[b, h, c, t] + g_tp[b, h, c * T.int64(_C) + t + T.int64(1)]

            for b, n, c, t in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C)):
                out_g_cumsum[b, n, c, t] = g_cumsum[b, n, c, t]

            for b, n, c, i, j in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)):
                if j <= i:
                    out_decay_mask[b, n, c, i, j] = T.exp(g_cumsum[b, n, c, i] - g_cumsum[b, n, c, j])
                else:
                    out_decay_mask[b, n, c, i, j] = T.float32(0)

            # ================================================================
            # Step 5: Attention matmul + mask + negate
            # ================================================================
            for b, n, c, i, j in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)):
                attn_raw[b, n, c, i, j] = T.float32(0)
            for b, n, c, i, j, d in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C), T.int64(_key_dim)):
                attn_raw[b, n, c, i, j] = attn_raw[b, n, c, i, j] + k_beta_chunk[b, n, c, i, d] * out_key[b, n, c, j, d]

            for b, n, c, i, j in T.grid(batch_size, T.int64(_num_heads), num_chunks, T.int64(_C), T.int64(_C)):
                if j >= i:
                    attn_masked[b, n, c, i, j] = T.float32(0)
                else:
                    attn_masked[b, n, c, i, j] = -(attn_raw[b, n, c, i, j] * out_decay_mask[b, n, c, i, j])

            # ================================================================
            # Step 6: Prefix scan (see L2 norm variant for detailed comments)
            # ================================================================
            for bhc in T.parallel(batch_size * T.int64(_num_heads) * num_chunks):
                b: T.int64 = bhc // (T.int64(_num_heads) * num_chunks)
                h: T.int64 = bhc % (T.int64(_num_heads) * num_chunks) // num_chunks
                c: T.int64 = bhc % num_chunks

                for j in range(T.int64(_C)):
                    out_attn[b, h, c, T.int64(0), j] = attn_masked[b, h, c, T.int64(0), j]

                for i in T.serial(T.int64(_C) - T.int64(1)):
                    row: T.int64 = i + T.int64(1)

                    for j in range(T.int64(_C) - row):
                        out_attn[b, h, c, row, row + j] = attn_masked[b, h, c, row, row + j]

                    for j in range(row):
                        out_attn[b, h, c, row, j] = attn_masked[b, h, c, row, j]
                        for k in range(j + T.int64(1), row):
                            out_attn[b, h, c, row, j] = out_attn[b, h, c, row, j] + attn_masked[b, h, c, row, k] * out_attn[b, h, c, k, j]

        return chunk_gated_delta_rule_no_norm


def _chunk_recurrent_gated_delta_rule():
    inputs = []
    outputs = []
    return te.create_prim_func(inputs + outputs)


print(_gen_causal_conv1d_update(128, 128).script())
print(_chunk_gated_delta_rule(32, 128, 16, 128, use_qk_l2norm_in_kernel=False).script())
print(_chunk_gated_delta_rule(32, 128, 16, 128, use_qk_l2norm_in_kernel=True).script())
