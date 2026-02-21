"""
Test suite for GatedDeltaNet TVM kernels.

Tests numerical correctness against PyTorch reference implementations.
"""

import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hf_reference"))

import torch
import numpy as np
import tvm
# nd will be assigned from tvm.runtime.ndarray or similar based on TVM version

# Import TVM kernel builders
from gdn_kernel_dev import build_causal_conv1d_update, build_chunk_gated_delta_rule

# Import PyTorch reference implementations
from torch_gdn_reference import torch_causal_conv1d_update, torch_chunk_gated_delta_rule


def allclose_with_report(actual, expected, rtol=1e-4, atol=1e-4, name="output"):
    """
    Check if arrays are close and print detailed error report if not.
    
    Args:
        actual: Actual output (numpy array)
        expected: Expected output (numpy array)
        rtol: Relative tolerance
        atol: Absolute tolerance
        name: Name of the output for error reporting
    
    Returns:
        bool: True if arrays are close within tolerance
    """
    if actual.shape != expected.shape:
        print(f"✗ {name}: Shape mismatch! actual={actual.shape}, expected={expected.shape}")
        return False
    
    diff = np.abs(actual - expected)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    close = np.allclose(actual, expected, rtol=rtol, atol=atol)
    
    if close:
        print(f"✓ {name}: PASS (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
    else:
        print(f"✗ {name}: FAIL (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
        print(f"  Tolerance: rtol={rtol}, atol={atol}")
        
        # Find worst mismatch location
        worst_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"  Worst mismatch at {worst_idx}:")
        print(f"    actual={actual[worst_idx]}, expected={expected[worst_idx]}")
    
    return close


def test_causal_conv1d_update(batch_size=2, hidden_size=128, seq_len=4, state_len=3, 
                                device="cuda", dtype=torch.float32, rtol=1e-4, atol=1e-4):
    """
    Test causal_conv1d_update kernel against PyTorch reference.
    
    Args:
        batch_size: Batch size
        hidden_size: Number of hidden dimensions
        seq_len: Sequence length
        state_len: Convolution state length (kernel_size - 1)
        device: Device to run on
        dtype: Data type
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        bool: True if test passes
    """
    print(f"\n{'='*70}")
    print(f"Testing causal_conv1d_update")
    print(f"  batch_size={batch_size}, hidden_size={hidden_size}, seq_len={seq_len}, state_len={state_len}")
    print(f"  dtype={dtype}, rtol={rtol}, atol={atol}")
    print(f"{'='*70}")
    
    # Generate random inputs
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, hidden_size, seq_len, dtype=dtype, device=device)
    conv_state = torch.randn(batch_size, hidden_size, state_len, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, 1, state_len, dtype=dtype, device=device)
    bias = torch.randn(hidden_size, dtype=dtype, device=device)
    
    # PyTorch reference (make a copy of conv_state since it's modified in-place)
    conv_state_torch = conv_state.clone()
    torch_out = torch_causal_conv1d_update(
        hidden_states.clone(),
        conv_state_torch,
        weight,
        bias,
        activation="silu",
    )
    
    # Build TVM kernel
    print("\nBuilding TVM kernel...")
    tvm_kernel = build_causal_conv1d_update(
        hidden_size=hidden_size,
        state_len=state_len,
        has_bias=True,
        activation="silu",
        target="cuda",
        dtype="float32" if dtype == torch.float32 else "float16",
    )
    
    # Prepare TVM inputs (TVM uses DLPack for interop with PyTorch)
    print("Running TVM kernel...")
    ctx = tvm.cuda(0)
    
    # Create output buffers  
    tvm_out = tvm.runtime.empty((batch_size, hidden_size, seq_len), dtype="float32", device=ctx)
    next_conv_state = tvm.runtime.empty((batch_size, hidden_size, state_len), dtype="float32", device=ctx)
    
    # Convert PyTorch tensors to TVM (via DLPack)
    tvm_hidden_states = tvm.runtime.from_dlpack(hidden_states)
    tvm_conv_state = tvm.runtime.from_dlpack(conv_state.clone())
    tvm_weight = tvm.runtime.from_dlpack(weight)
    tvm_bias = tvm.runtime.from_dlpack(bias)
    
    # Run kernel
    tvm_kernel(
        tvm_hidden_states,
        tvm_conv_state,
        tvm_weight,
        tvm_bias,
        next_conv_state,
        tvm_out,
    )
    
    # Convert back to numpy for comparison
    tvm_out_np = tvm_out.numpy()
    torch_out_np = torch_out.cpu().numpy()
    
    # Check results
    print("\nComparing results...")
    passed = allclose_with_report(tvm_out_np, torch_out_np, rtol=rtol, atol=atol, name="output")
    
    if passed:
        print(f"\n{'✓'*35}")
        print("TEST PASSED")
        print(f"{'✓'*35}")
    else:
        print(f"\n{'✗'*35}")
        print("TEST FAILED")
        print(f"{'✗'*35}")
    
    return passed


def test_chunk_gated_delta_rule(batch_size=2, seq_len=128, num_v_heads=32, v_head_dim=128,
                                  num_k_heads=16, k_head_dim=128, chunk_size=64,
                                  device="cuda", dtype=torch.float32, rtol=1e-4, atol=1e-4):
    """
    Test chunk_gated_delta_rule kernel against PyTorch reference.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_v_heads: Number of value heads
        v_head_dim: Value head dimension
        num_k_heads: Number of key heads
        k_head_dim: Key head dimension
        chunk_size: Chunk size
        device: Device to run on
        dtype: Data type
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        bool: True if test passes
    """
    print(f"\n{'='*70}")
    print(f"Testing chunk_gated_delta_rule")
    print(f"  batch_size={batch_size}, seq_len={seq_len}")
    print(f"  num_v_heads={num_v_heads}, v_head_dim={v_head_dim}")
    print(f"  num_k_heads={num_k_heads}, k_head_dim={k_head_dim}")
    print(f"  chunk_size={chunk_size}, dtype={dtype}")
    print(f"  rtol={rtol}, atol={atol}")
    print(f"{'='*70}")
    
    # Generate random inputs
    torch.manual_seed(42)
    query = torch.randn(batch_size, seq_len, num_v_heads, k_head_dim, dtype=dtype, device=device)
    key = torch.randn(batch_size, seq_len, num_v_heads, k_head_dim, dtype=dtype, device=device)
    value = torch.randn(batch_size, seq_len, num_v_heads, v_head_dim, dtype=dtype, device=device)
    
    # g should be negative (decay parameter): g = -A * softplus(a + dt_bias)
    # Typical range: [-20, -0.1] for numerical stability
    g = -torch.rand(batch_size, seq_len, num_v_heads, dtype=dtype, device=device) * 5.0 - 0.1
    
    # beta should be in (0, 1) (gating parameter): beta = sigmoid(b)
    # Use sigmoid to ensure proper range
    beta = torch.sigmoid(torch.randn(batch_size, seq_len, num_v_heads, dtype=dtype, device=device))
    
    # PyTorch reference
    print("\nRunning PyTorch reference...")
    torch_out, torch_state, torch_intermediates = torch_chunk_gated_delta_rule(
        query.clone(),
        key.clone(),
        value.clone(),
        g.clone(),
        beta.clone(),
        chunk_size=chunk_size,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        capture_intermediates=True,
    )
    
    # Build TVM kernel
    print("\nBuilding TVM kernel...")
    tvm_kernel = build_chunk_gated_delta_rule(
        linear_num_value_heads=num_v_heads,
        linear_value_head_dim=v_head_dim,
        linear_num_key_heads=num_k_heads,
        linear_key_head_dim=k_head_dim,
        chunk_size=chunk_size,
        target="cuda",
        dtype="float32" if dtype == torch.float32 else "float16",
    )
    
    # Prepare TVM inputs
    print("Running TVM kernel...")
    ctx = tvm.cuda(0)
    
    # Import helper to get buffer shapes
    from gdn_kernel_dev import get_intermediate_buffer_shapes
    
    # Get all intermediate buffer shapes
    buffer_shapes = get_intermediate_buffer_shapes(
        batch_size=batch_size,
        seq_len=seq_len,
        linear_num_value_heads=num_v_heads,
        linear_value_head_dim=v_head_dim,
        linear_key_head_dim=k_head_dim,
        chunk_size=chunk_size,
    )
    
    # Create output buffers
    tvm_core_attn_out = tvm.runtime.empty((batch_size, seq_len, num_v_heads, v_head_dim), dtype="float32", device=ctx)
    tvm_state_out = tvm.runtime.empty((batch_size, num_v_heads, k_head_dim, v_head_dim), dtype="float32", device=ctx)
    
    # Allocate all intermediate buffers
    intermediate_buffers = {}
    for name, shape in buffer_shapes.items():
        intermediate_buffers[name] = tvm.runtime.empty(shape, dtype="float32", device=ctx)
    
    # Convert PyTorch tensors to TVM
    tvm_query = tvm.runtime.from_dlpack(query)
    tvm_key = tvm.runtime.from_dlpack(key)
    tvm_value = tvm.runtime.from_dlpack(value)
    tvm_g = tvm.runtime.from_dlpack(g)
    tvm_beta = tvm.runtime.from_dlpack(beta)
    
    # Initial state (zeros)
    initial_state = torch.zeros(batch_size, num_v_heads, k_head_dim, v_head_dim, dtype=dtype, device=device)
    tvm_initial_state = tvm.runtime.from_dlpack(initial_state)
    
    # Run kernel (with all intermediate buffers)
    tvm_kernel(
        tvm_query,
        tvm_key,
        tvm_value,
        tvm_g,
        tvm_beta,
        tvm_initial_state,
        tvm_core_attn_out,
        tvm_state_out,
        # Intermediate buffers in order
        intermediate_buffers["query_T"],
        intermediate_buffers["key_T"],
        intermediate_buffers["k_beta"],
        intermediate_buffers["value_T"],
        intermediate_buffers["v_beta"],
        intermediate_buffers["g_T"],
        intermediate_buffers["beta_T"],
        intermediate_buffers["query_chunked"],
        intermediate_buffers["key_chunked"],
        intermediate_buffers["value_chunked"],
        intermediate_buffers["k_beta_chunked"],
        intermediate_buffers["v_beta_chunked"],
        intermediate_buffers["g_chunked"],
        intermediate_buffers["g_cumsum"],
        intermediate_buffers["decay_mask"],
        intermediate_buffers["attn_mm_out"],
        intermediate_buffers["attn_decay_neg_mask_out"],
        intermediate_buffers["attn_associative_scan_out"],
        intermediate_buffers["attn_identity_add_out"],
        intermediate_buffers["value_attn_vbeta_matmul_out"],
        intermediate_buffers["k_beta_x_g_cumsum_exp_tmp"],
        intermediate_buffers["k_cumdecay"],
        intermediate_buffers["recurrent_state_update_attn_out"],
        intermediate_buffers["recurrent_state_update_v_buf"],
        intermediate_buffers["recurrent_state_update_attn_inter"],
        intermediate_buffers["core_attn_out_inter"],
    )
    
    # Convert back to numpy for comparison
    tvm_out_np = tvm_core_attn_out.numpy()
    torch_out_np = torch_out.cpu().numpy()
    
    # DEBUG: Check intermediate buffers that feed into kernel_22
    print(f"\n{'='*70}")
    print("DEBUG: Kernel 22 Input Buffers")
    print(f"{'='*70}")
    
    debug_buffers = [
        "attn_identity_add_out",
        "k_cumdecay",
        "key_chunked", 
        "query_chunked",
        "g_cumsum",
        "value_attn_vbeta_matmul_out",
        "recurrent_state_update_v_buf",
        "recurrent_state_update_attn_inter",
        "core_attn_out_inter"
    ]
    
    for buf_name in debug_buffers:
        if buf_name in intermediate_buffers:
            buf_np = intermediate_buffers[buf_name].numpy()
            non_zero = np.count_nonzero(buf_np)
            print(f"{buf_name:40s} shape={str(buf_np.shape):30s} "
                  f"min={buf_np.min():10.6f} max={buf_np.max():10.6f} non_zero={non_zero}")
    
    # Check recurrent state output
    state_out_np = tvm_state_out.numpy()
    print(f"\nrecurrent_state_out (final) stats:")
    print(f"  Shape: {state_out_np.shape}")
    print(f"  Min: {state_out_np.min():.6f}, Max: {state_out_np.max():.6f}")
    print(f"  Mean: {state_out_np.mean():.6f}, NaN count: {np.isnan(state_out_np).sum()}")
    print(f"  Non-zero count: {np.count_nonzero(state_out_np)}")
    
    # Check core_attn_out_inter (before unchunking/padding removal)
    core_inter_np = intermediate_buffers["core_attn_out_inter"].numpy()
    print(f"\ncore_attn_out_inter stats:")
    print(f"  Shape: {core_inter_np.shape}")
    print(f"  Min: {core_inter_np.min():.6f}, Max: {core_inter_np.max():.6f}")
    print(f"  Mean: {core_inter_np.mean():.6f}, NaN count: {np.isnan(core_inter_np).sum()}")
    print(f"  Non-zero count: {np.count_nonzero(core_inter_np)}")
    
    # Check per-chunk statistics
    print(f"\n Per-chunk analysis:")
    for chunk_idx in range(core_inter_np.shape[2]):  # Iterate over chunks
        chunk_data = core_inter_np[:, :, chunk_idx, :, :]
        print(f"  Chunk {chunk_idx}: min={chunk_data.min():.3f}, max={chunk_data.max():.3f}, non_zero={np.count_nonzero(chunk_data)}/{chunk_data.size}")
    
    print(f"\nFinal tvm_core_attn_out stats:")
    print(f"  Shape: {tvm_out_np.shape}")
    print(f"  Min: {tvm_out_np.min():.6f}, Max: {tvm_out_np.max():.6f}")
    print(f"  Mean: {tvm_out_np.mean():.6f}, NaN count: {np.isnan(tvm_out_np).sum()}")
    print(f"  Non-zero count: {np.count_nonzero(tvm_out_np)}")
    
    # Print ALL intermediate buffer statistics and compare with PyTorch
    print("\n" + "="*70)
    print("Intermediate Buffer Comparison: TVM vs PyTorch")
    print("="*70)
    print(f"{'Buffer Name':35s} {'TVM Shape':25s} {'Torch Shape':25s} {'Max Diff':15s} {'Status':10s}")
    print("-" * 130)
    
    for name in sorted(intermediate_buffers.keys()):
        tvm_buf = intermediate_buffers[name].numpy()
        
        # Check if PyTorch has equivalent intermediate
        if name in torch_intermediates:
            torch_buf = torch_intermediates[name].cpu().numpy()
            
            # Handle inter-chunk buffers - PyTorch only captures chunk 0
            # TVM shape: (batch, heads, num_chunks, chunk_size, dim)
            # PyTorch shape: (batch, heads, chunk_size, dim)
            if tvm_buf.shape != torch_buf.shape and len(tvm_buf.shape) == len(torch_buf.shape) + 1:
                # Check if this is an inter-chunk buffer (has chunk dimension)
                if name in ["recurrent_state_update_attn_out", "recurrent_state_update_v_buf", 
                           "recurrent_state_update_attn_inter", "core_attn_out_inter"]:
                    # Compare only chunk 0
                    tvm_buf_chunk0 = tvm_buf[:, :, 0]  # Take first chunk
                    if tvm_buf_chunk0.shape == torch_buf.shape:
                        diff = np.abs(tvm_buf_chunk0 - torch_buf)
                        max_diff = f"{np.max(diff):.6e}"
                        if np.allclose(tvm_buf_chunk0, torch_buf, rtol=1e-4, atol=1e-4):
                            status = "✓ MATCH (chunk 0)"
                        else:
                            status = "✗ DIFF (chunk 0)"
                        print(f"{name:35s} {str(tvm_buf.shape):25s} {str(torch_buf.shape):25s} {max_diff:15s} {status:10s}")
                        continue
            
            # Compare shapes
            if tvm_buf.shape != torch_buf.shape:
                status = "SHAPE MISMATCH"
                max_diff = "N/A"
            else:
                diff = np.abs(tvm_buf - torch_buf)
                max_diff = f"{np.max(diff):.6e}"
                
                # Check if values match
                if np.allclose(tvm_buf, torch_buf, rtol=1e-4, atol=1e-4):
                    status = "✓ MATCH"
                else:
                    status = "✗ DIFF"
                    
            print(f"{name:35s} {str(tvm_buf.shape):25s} {str(torch_buf.shape):25s} {max_diff:15s} {status:10s}")
        else:
            # No PyTorch equivalent
            has_nan = np.isnan(tvm_buf).sum() > 0
            has_inf = np.isinf(tvm_buf).sum() > 0
            flag = ""
            if has_nan:
                flag += "[NaN]"
            if has_inf:
                flag += "[Inf]"
            print(f"{name:35s} {str(tvm_buf.shape):25s} {'N/A':25s} {'N/A':15s} {flag:10s}")
    
    print("="*70 + "\n")
    
    # Check results
    print("\nComparing results...")
    passed = allclose_with_report(tvm_out_np, torch_out_np, rtol=rtol, atol=atol, name="core_attn_out")
    
    if passed:
        print(f"\n{'✓'*35}")
        print("TEST PASSED")
        print(f"{'✓'*35}")
    else:
        print(f"\n{'✗'*35}")
        print("TEST FAILED")
        print(f"{'✗'*35}")
    
    return passed


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GatedDeltaNet TVM Kernel Test Suite")
    print("="*70)
    
    results = {}
    
#    # Test 1: causal_conv1d_update
#    try:
#        results["causal_conv1d_update"] = test_causal_conv1d_update(
#            batch_size=2,
#            hidden_size=128,
#            seq_len=4,
#            state_len=3,
#            rtol=1e-4,
#            atol=1e-4,
#        )
#    except Exception as e:
#        print(f"\n✗ causal_conv1d_update crashed: {e}")
#        import traceback
#        traceback.print_exc()
#        results["causal_conv1d_update"] = False
#    
    # Test 2: chunk_gated_delta_rule
    for i in range(10):
        try:
            results[f"chunk_gated_delta_rule_{i}"] = test_chunk_gated_delta_rule(
                batch_size=2,
                seq_len=128,
                num_v_heads=32,
                v_head_dim=128,
                num_k_heads=16,
                k_head_dim=128,
                chunk_size=64,
                rtol=5e-3,
                atol=5e-3,
            )
        except Exception as e:
            print(f"\n✗ chunk_gated_delta_rule_{i} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[f"chunk_gated_delta_rule_{i}"] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)
