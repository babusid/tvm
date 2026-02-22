"""
Test suite for GatedDeltaNet TVM kernels.

Tests numerical correctness against PyTorch reference implementations.
"""

import sys
import os
import argparse

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


def test_causal_conv1d_update(
    batch_size=2,
    hidden_size=128,
    seq_len=4,
    state_len=3,
    device="cuda",
    dtype=torch.float32,
    rtol=1e-4,
    atol=1e-4,
):
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
    print(f"\n{'=' * 70}")
    print(f"Testing causal_conv1d_update")
    print(
        f"  batch_size={batch_size}, hidden_size={hidden_size}, seq_len={seq_len}, state_len={state_len}"
    )
    print(f"  dtype={dtype}, rtol={rtol}, atol={atol}")
    print(f"{'=' * 70}")

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
    tvm_kernel, _ = build_causal_conv1d_update(
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
    next_conv_state = tvm.runtime.empty(
        (batch_size, hidden_size, state_len), dtype="float32", device=ctx
    )

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
        print(f"\n{'✓' * 35}")
        print("TEST PASSED")
        print(f"{'✓' * 35}")
    else:
        print(f"\n{'✗' * 35}")
        print("TEST FAILED")
        print(f"{'✗' * 35}")

    return passed


def test_chunk_gated_delta_rule(
    batch_size=2,
    seq_len=128,
    num_v_heads=32,
    v_head_dim=128,
    num_k_heads=16,
    k_head_dim=128,
    chunk_size=64,
    device="cuda",
    dtype=torch.float32,
    rtol=1e-4,
    atol=1e-4,
    seed=42,
):
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
    print(f"\n{'=' * 70}")
    print(f"Testing chunk_gated_delta_rule")
    print(f"  batch_size={batch_size}, seq_len={seq_len}")
    print(f"  num_v_heads={num_v_heads}, v_head_dim={v_head_dim}")
    print(f"  num_k_heads={num_k_heads}, k_head_dim={k_head_dim}")
    print(f"  chunk_size={chunk_size}, dtype={dtype}")
    print(f"  rtol={rtol}, atol={atol}")
    print(f"{'=' * 70}")

    # Generate random inputs
    torch.manual_seed(seed)
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
    torch_out, torch_state = torch_chunk_gated_delta_rule(
        query.clone(),
        key.clone(),
        value.clone(),
        g.clone(),
        beta.clone(),
        chunk_size=chunk_size,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
    )

    # Build TVM kernel (force fresh build by adding unique name suffix)
    print("\nBuilding TVM kernel...")
    tvm_kernel,_ = build_chunk_gated_delta_rule(
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

    # Create output buffers (initialize to zero to avoid uninitialized memory bugs)
    tvm_core_attn_out = tvm.runtime.empty(
        (batch_size, seq_len, num_v_heads, v_head_dim), dtype="float32", device=ctx
    )
    tvm_state_out = tvm.runtime.empty(
        (batch_size, num_v_heads, k_head_dim, v_head_dim), dtype="float32", device=ctx
    )

    print(
        f"  TVM output buffer shapes: core_attn_out={tvm_core_attn_out.shape}, state_out={tvm_state_out.shape}"
    )

    # Convert PyTorch tensors to TVM
    tvm_query = tvm.runtime.from_dlpack(query.clone())
    tvm_key = tvm.runtime.from_dlpack(key.clone())
    tvm_value = tvm.runtime.from_dlpack(value.clone())
    tvm_g = tvm.runtime.from_dlpack(g.clone())
    tvm_beta = tvm.runtime.from_dlpack(beta.clone())

    # Initial state (zeros)
    initial_state = torch.zeros(
        batch_size, num_v_heads, k_head_dim, v_head_dim, dtype=dtype, device=device
    )
    tvm_initial_state = tvm.runtime.from_dlpack(initial_state.clone())

    # Run kernel
    tvm_kernel(
        tvm_query,
        tvm_key,
        tvm_value,
        tvm_g,
        tvm_beta,
        tvm_initial_state,
        tvm_core_attn_out,
        tvm_state_out,
    )

    # Convert back to numpy for comparison
    tvm_out_np = tvm_core_attn_out.numpy()
    torch_out_np = torch_out.cpu().numpy()

    # Check results
    print("\nComparing results...")
    passed = allclose_with_report(
        tvm_out_np, torch_out_np, rtol=rtol, atol=atol, name="core_attn_out"
    )

    if passed:
        print(f"\n{'✓' * 35}")
        print("TEST PASSED")
        print(f"{'✓' * 35}")
    else:
        print(f"\n{'✗' * 35}")
        print("TEST FAILED")
        print(f"{'✗' * 35}")

    return passed


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test suite for GatedDeltaNet TVM kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--func",
        choices=["causal_conv1d_update", "chunk_gated_delta_rule"],
        help="Filter tests by function name",
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for custom test (requires --seq_len and --func)"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        help="Sequence length for custom test (requires --batch_size and --func)",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        help="Override the number of iterations for tests (default: use dict values)",
    )

    args = parser.parse_args()

    # Validate arguments
    has_batch = args.batch_size is not None
    has_seq = args.seq_len is not None

    if has_batch != has_seq:
        parser.error("--batch_size and --seq_len must be specified together")

    if (has_batch or has_seq) and args.func is None:
        parser.error("--func must be specified when using --batch_size and --seq_len")

    print("\n" + "=" * 70)
    print("GatedDeltaNet TVM Kernel Test Suite")
    print("=" * 70)

    # Test configuration dictionary
    # Key: test_name (string) - format: {func}_b{batch}_s{seq_len}
    # Value: tuple of (test_func, num_iterations, kwargs_dict)
    TEST_CONFIGS = {
        # Causal Conv1D tests
        "causal_conv1d_update_b2_s4": (
            test_causal_conv1d_update,
            5,
            {
                "batch_size": 2,
                "hidden_size": 128,
                "seq_len": 4,
                "state_len": 3,
                "rtol": 1e-4,
                "atol": 1e-4,
            },
        ),
        # Chunked Gated Delta Rule tests - baseline (batch=2, seq_len=128)
        "chunk_gated_delta_rule_b2_s128": (
            test_chunk_gated_delta_rule,
            5,
            {
                "batch_size": 2,
                "seq_len": 128,
                "num_v_heads": 32,
                "v_head_dim": 128,
                "num_k_heads": 16,
                "k_head_dim": 128,
                "chunk_size": 64,
                "rtol": 5e-3,
                "atol": 5e-3,
            },
        ),
        # Non-power-of-2 batch size (3)
        "chunk_gated_delta_rule_b3_s128": (
            test_chunk_gated_delta_rule,
            5,
            {
                "batch_size": 3,
                "seq_len": 128,
                "num_v_heads": 32,
                "v_head_dim": 128,
                "num_k_heads": 16,
                "k_head_dim": 128,
                "chunk_size": 64,
                "rtol": 5e-3,
                "atol": 5e-3,
            },
        ),
        # Non-power-of-2 batch size (5)
        "chunk_gated_delta_rule_b5_s128": (
            test_chunk_gated_delta_rule,
            5,
            {
                "batch_size": 5,
                "seq_len": 128,
                "num_v_heads": 32,
                "v_head_dim": 128,
                "num_k_heads": 16,
                "k_head_dim": 128,
                "chunk_size": 64,
                "rtol": 5e-3,
                "atol": 5e-3,
            },
        ),
        # Short sequence - single chunk (64 tokens)
        "chunk_gated_delta_rule_b2_s64": (
            test_chunk_gated_delta_rule,
            5,
            {
                "batch_size": 2,
                "seq_len": 64,
                "num_v_heads": 32,
                "v_head_dim": 128,
                "num_k_heads": 16,
                "k_head_dim": 128,
                "chunk_size": 64,
                "rtol": 5e-3,
                "atol": 5e-3,
            },
        ),
        # Non-power-of-2 sequence length (100 tokens, needs padding)
        "chunk_gated_delta_rule_b2_s100": (
            test_chunk_gated_delta_rule,
            5,
            {
                "batch_size": 2,
                "seq_len": 100,
                "num_v_heads": 32,
                "v_head_dim": 128,
                "num_k_heads": 16,
                "k_head_dim": 128,
                "chunk_size": 64,
                "rtol": 5e-3,
                "atol": 5e-3,
            },
        ),
        # Non-power-of-2 sequence length (200 tokens)
        "chunk_gated_delta_rule_b2_s200": (
            test_chunk_gated_delta_rule,
            5,
            {
                "batch_size": 2,
                "seq_len": 200,
                "num_v_heads": 32,
                "v_head_dim": 128,
                "num_k_heads": 16,
                "k_head_dim": 128,
                "chunk_size": 64,
                "rtol": 5e-3,
                "atol": 5e-3,
            },
        ),
        # Longer sequence - 512 tokens (8 chunks)
        "chunk_gated_delta_rule_b2_s512": (
            test_chunk_gated_delta_rule,
            5,
            {
                "batch_size": 2,
                "seq_len": 512,
                "num_v_heads": 32,
                "v_head_dim": 128,
                "num_k_heads": 16,
                "k_head_dim": 128,
                "chunk_size": 64,
                "rtol": 5e-3,
                "atol": 5e-3,
            },
        ),
        # Non-power-of-2 + longer (333 tokens, ~5.2 chunks)
        "chunk_gated_delta_rule_b2_s333": (
            test_chunk_gated_delta_rule,
            5,
            {
                "batch_size": 2,
                "seq_len": 333,
                "num_v_heads": 32,
                "v_head_dim": 128,
                "num_k_heads": 16,
                "k_head_dim": 128,
                "chunk_size": 64,
                "rtol": 5e-3,
                "atol": 5e-3,
            },
        ),
        # Combined: non-power-of-2 batch (3) + non-power-of-2 seq (150)
        "chunk_gated_delta_rule_b3_s150": (
            test_chunk_gated_delta_rule,
            5,
            {
                "batch_size": 3,
                "seq_len": 150,
                "num_v_heads": 32,
                "v_head_dim": 128,
                "num_k_heads": 16,
                "k_head_dim": 128,
                "chunk_size": 64,
                "rtol": 5e-3,
                "atol": 5e-3,
            },
        ),
        # Edge case: batch=1 (single sequence inference)
        "chunk_gated_delta_rule_b1_s256": (
            test_chunk_gated_delta_rule,
            5,
            {
                "batch_size": 1,
                "seq_len": 256,
                "num_v_heads": 32,
                "v_head_dim": 128,
                "num_k_heads": 16,
                "k_head_dim": 128,
                "chunk_size": 64,
                "rtol": 5e-3,
                "atol": 5e-3,
            },
        ),
    }

    results = {}

    # Determine which tests to run
    if args.batch_size is not None and args.seq_len is not None:
        # Custom test with specified batch_size and seq_len
        # Use provided num_iters or default to 5
        num_iterations = args.num_iters if args.num_iters is not None else 5
        if args.func == "causal_conv1d_update":
            test_configs_to_run = {
                f"causal_conv1d_update_custom_b{args.batch_size}_s{args.seq_len}": (
                    test_causal_conv1d_update,
                    num_iterations,
                    {
                        "batch_size": args.batch_size,
                        "hidden_size": 128,
                        "seq_len": min(args.seq_len, 16),  # Cap seq_len for conv1d
                        "state_len": 3,
                        "rtol": 1e-4,
                        "atol": 1e-4,
                    },
                )
            }
        else:  # chunk_gated_delta_rule
            test_configs_to_run = {
                f"chunk_gated_delta_rule_custom_b{args.batch_size}_s{args.seq_len}": (
                    test_chunk_gated_delta_rule,
                    num_iterations,
                    {
                        "batch_size": args.batch_size,
                        "seq_len": args.seq_len,
                        "num_v_heads": 32,
                        "v_head_dim": 128,
                        "num_k_heads": 16,
                        "k_head_dim": 128,
                        "chunk_size": 64,
                        "rtol": 5e-3,
                        "atol": 5e-3,
                    },
                )
            }
    else:
        # Use default TEST_CONFIGS, filtered by --func if specified
        # Override num_iterations if --num_iters provided
        test_configs_to_run = {}
        for test_name, (test_func, num_iterations, test_kwargs) in TEST_CONFIGS.items():
            if args.func is None or args.func in test_name:
                # Override num_iterations if flag provided
                if args.num_iters is not None:
                    num_iterations = args.num_iters
                test_configs_to_run[test_name] = (test_func, num_iterations, test_kwargs)

    # Run test configurations
    for test_name, (test_func, num_iterations, test_kwargs) in test_configs_to_run.items():
        for iteration in range(num_iterations):
            test_instance_name = f"{test_name}_iter{iteration}"
            try:
                results[test_instance_name] = test_func(**test_kwargs)
            except Exception as e:
                print(f"\n✗ {test_instance_name} crashed: {e}")
                import traceback

                traceback.print_exc()
                results[test_instance_name] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
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
