"""
LLM.int8() - milestones 1, 2, and 3.

Includes:
  - FP16 GEMM baseline
  - Per-tensor absmax INT8 GEMM baseline
  - Vector-wise INT8 GEMM baseline
  - Mixed-precision decomposition (vector-wise INT8 + FP16 outliers)
  - Numerical validation, CUDA event benchmarks, and storage estimates

Run on a CUDA machine with PyTorch installed:
    python src/main.py
    python src/main.py --quick
    python src/main.py --quick --varying-scales
    python src/main.py --quick --zero-row-column
    python src/main.py --quick --outliers
    python src/main.py --self-test

The INT8 path uses torch._int_mm, an internal PyTorch API.
This performs INT8 x INT8 -> INT32 matrix multiplication.
"""

import argparse
import torch

# -----------------------------------------------------------------------------
# Milestones 1 and 2: Existing FP16 / INT8 Baselines
# -----------------------------------------------------------------------------

def quantize_absmax_per_tensor(x: torch.Tensor):
    """Return (signed INT8 tensor, FLOAT32 dequantization step)."""
    xf = x.float()
    scale = (xf.abs().amax() / 127.0).clamp_min(1e-12)
    x8 = torch.round(xf / scale).clamp(-127, 127).to(torch.int8)
    return x8.contiguous(), scale

def quantize_absmax_vectorwise(x: torch.Tensor, dim: int):
    """
    Vector-wise symmetric INT8 quantization.

    dim=1: one scale per row (activations)
    dim=0: one scale per column (weights)
    """
    xf = x.float()

    scale = (xf.abs().amax(dim=dim, keepdim=True) / 127.0).clamp_min(1e-12)

    x8 = torch.round(xf / scale).clamp(-127, 127).to(torch.int8)

    return x8.contiguous(), scale

def fp16_baseline(x16: torch.Tensor, w16: torch.Tensor):
    return x16 @ w16

def int8_from_prequantized(x8, w8, sx, sw):
    # REAL INT8 operands, INT32 accumulation.
    accum32 = torch._int_mm(x8, w8)
    return (accum32.float() * (sx * sw)).half()

def int8_baseline(x16: torch.Tensor, w16: torch.Tensor):
    """End-to-end baseline: quantize BOTH inputs, INT8 GEMM, dequantize."""
    x8, sx = quantize_absmax_per_tensor(x16)
    w8, sw = quantize_absmax_per_tensor(w16)
    return int8_from_prequantized(x8, w8, sx, sw)

def vectorwise_int8_baseline(x16: torch.Tensor, w16: torch.Tensor):
    """
    Vector-wise INT8 GEMM:
    - Activation rows quantized independently
    - Weight columns quantized independently
    - INT32 accumulation
    - Dequantization to FP16
    """

    # Quantize activation rows
    x8, sx = quantize_absmax_vectorwise(x16, dim=1)

    # Quantize weight columns
    w8, sw = quantize_absmax_vectorwise(w16, dim=0)

    # Reuse existing INT8 GEMM and dequantization
    return int8_from_prequantized(x8, w8, sx, sw)

# -----------------------------------------------------------------------------
# Milestone 3: Mixed-Precision Decomposition
# -----------------------------------------------------------------------------
def detect_outlier_features(x16: torch.Tensor, threshold: float = 6.0):
    """
    Return a boolean mask for outlier feature dimensions (columns of X).
    Feature k is an outlier if at least one activation has |X[m, k]| >= 6.
    """
    return (x16.abs() >= threshold).any(dim=0)


def decompose_features(x16: torch.Tensor, w16: torch.Tensor, outlier_mask):
    """
    Split the X @ W reduction dimension into regular and outlier features.

    Regular matrices are zero-masked rather than narrowed so that the INT8
    GEMM keeps the original K dimension and its memory alignment.

    Returns regular X/W, compact FP16 outlier X/W, and outlier indices.
    """
    outlier_indices = torch.where(outlier_mask)[0]

    x_regular = x16.masked_fill(outlier_mask.unsqueeze(0), 0)
    w_regular = w16.masked_fill(outlier_mask.unsqueeze(1), 0)

    x_outlier = x16.index_select(1, outlier_indices)
    w_outlier = w16.index_select(0, outlier_indices)

    return x_regular, w_regular, x_outlier, w_outlier, outlier_indices


def mixed_precision_llm_int8(
    x16: torch.Tensor,
    w16: torch.Tensor,
    threshold: float = 6.0,
):
    """
    Scaled-down LLM.int8() from Dettmers et al., Section 3.2 / Equation 8.

      1. Detect outlier feature dimensions from activations X.
      2. Zero out those dimensions BEFORE vector-wise quantization.
      3. Compute remaining features using vector-wise INT8 GEMM.
      4. Compute outlier features separately using FP16 GEMM.
      5. Add both results, returning an FP16 output.

    This is an unoptimized PyTorch correctness baseline, not a fused CUDA
    implementation. It dynamically quantizes both X and W on every call.
    """
    outlier_mask = detect_outlier_features(x16, threshold)
    x_regular, w_regular, x_outlier, w_outlier, outlier_indices = decompose_features(
        x16, w16, outlier_mask
    )

    # Remove outliers before quantization so they cannot distort INT8 scales.
    y_regular = vectorwise_int8_baseline(x_regular, w_regular)

    if outlier_indices.numel() == 0:
        return y_regular

    # FP16 GEMM over only the selected outlier dimensions.
    y_outlier = x_outlier @ w_outlier

    # Combine in FP32 before converting the final output to FP16.
    return (y_regular.float() + y_outlier.float()).half()

# -----------------------------------------------------------------------------
# Evaluation utilities
# -----------------------------------------------------------------------------

def cuda_time_ms(fn, warmup=10, repeat=40):
    """Average GPU elapsed time per invocation, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat

def error_metrics(estimate, reference):
    estimate = estimate.float()
    reference = reference.float()
    diff = (estimate - reference).abs()
    mae = diff.mean().item()
    # A mean relative error with protection against near-zero reference values.
    mean_relative = (diff / reference.abs().clamp_min(1e-3)).mean().item()
    return mae, mean_relative

def relative_l2_error(estimate, reference):
    estimate = estimate.float()
    reference = reference.float()

    diff = estimate - reference

    return (
        torch.linalg.vector_norm(diff)
        / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    ).item()

def validate_decomposition(x, w, outlier_mask, reference32):
    """
    Check the exact mathematical split in FP32, without quantization error:
        X @ W == X_regular @ W_regular + X_outlier @ W_outlier.
    """
    x_reg, w_reg, x_out, w_out, outlier_indices = decompose_features(x, w, outlier_mask)
    reconstructed = x_reg.float() @ w_reg.float()
    if outlier_indices.numel() > 0:
        reconstructed += x_out.float() @ w_out.float()
    torch.testing.assert_close(reconstructed, reference32, rtol=1e-3, atol=1e-2)

def benchmark_shape(
    m,
    k,
    n,
    *,
    outliers=False,
    varying_scales=False,
    zero_row_column=False,
    warmup=10,
    repeat=40
):
    torch.manual_seed(479)
    device = 'cuda'
    
    # Generate input matrices
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(k, n, device=device, dtype=torch.float16)

    # Test different row/column magnitudes
    if varying_scales:
        row_scales = torch.logspace(-1, 1, m, device=device)
        col_scales = torch.logspace(-1, 1, n, device=device)
        x = (x.float() * row_scales[:, None]).half()
        w = (w.float() * col_scales[None, :]).half()

    if outliers:
        # A handful of whole feature dimensions with high activation values.
        x[:, 0] = 40.0

    # Test zero-valued rows and columns
    if zero_row_column:
        x[0, :] = 0.0
        w[:, 0] = 0.0

    # FP32 GEMM reference from exactly the same rounded FP16 input tensors.
    reference32 = x.float() @ w.float()

    # Compute baseline outputs
    y16 = fp16_baseline(x, w)
    y8 = int8_baseline(x, w)

    # Vector-wise output
    y8_vector = vectorwise_int8_baseline(x, w)

    # Mixed output
    y8_mixed = mixed_precision_llm_int8(x, w)

    # Per-tensor INT8 validation
    x8, sx = quantize_absmax_per_tensor(x)
    w8, sw = quantize_absmax_per_tensor(w)
    accum32 = torch._int_mm(x8, w8)
    assert accum32.dtype == torch.int32
    assert x8.dtype == torch.int8 and w8.dtype == torch.int8
    assert y8.dtype == torch.float16

    # Vector-wise quantization validation
    x8_vector, sx_vector = quantize_absmax_vectorwise(x, dim=1)
    w8_vector, sw_vector = quantize_absmax_vectorwise(w, dim=0)

    # Verify scale dimensions
    assert sx_vector.shape == (m, 1)
    assert sw_vector.shape == (1, n)

    # Verify quantization types
    assert x8_vector.dtype == torch.int8
    assert w8_vector.dtype == torch.int8

    # Verify INT32 accumulation
    accum_vector = torch._int_mm(x8_vector, w8_vector)
    assert accum_vector.dtype == torch.int32

    # Verify output
    assert y8_vector.shape == reference32.shape
    assert y8_vector.dtype == torch.float16
    assert torch.isfinite(y8_vector).all()

    # Verify scales contain no NaN or infinity
    assert torch.isfinite(sx_vector).all()
    assert torch.isfinite(sw_vector).all()

    # Verify zero row/column behavior
    if zero_row_column:
        assert torch.count_nonzero(y8_vector[0, :]) == 0
        assert torch.count_nonzero(y8_vector[:, 0]) == 0

    # Verify outlier decomposition
    outlier_mask = detect_outlier_features(x)
    x_regular, w_regular, x_outlier, w_outlier, outlier_indices = (
        decompose_features(x, w, outlier_mask)
    )
    num_outliers = outlier_indices.numel()

    assert outlier_mask.shape == (k,)
    assert outlier_mask.dtype == torch.bool
    assert x_regular.shape == x.shape
    assert w_regular.shape == w.shape
    assert x_outlier.shape == (m, num_outliers)
    assert w_outlier.shape == (num_outliers, n)
    assert y8_mixed.shape == reference32.shape
    assert y8_mixed.dtype == torch.float16
    assert torch.isfinite(y8_mixed).all()

    # Selected features must be zero in the INT8 path.
    assert torch.count_nonzero(x_regular[:, outlier_mask]) == 0
    assert torch.count_nonzero(w_regular[outlier_mask, :]) == 0

    # Verify that regular + outlier channels reconstruct the FP32 GEMM.
    validate_decomposition(x, w, outlier_mask, reference32)

    if num_outliers == 0:
        # With no outliers, the method is ordinary vector-wise INT8.
        torch.testing.assert_close(y8_mixed, y8_vector, rtol=0, atol=0)

    if outliers:
        assert outlier_mask[0]

    if zero_row_column:
        assert torch.count_nonzero(y8_mixed[0, :]) == 0
        assert torch.count_nonzero(y8_mixed[:, 0]) == 0

    # Numerical accuracy
    f16_mae, f16_rel = error_metrics(y16, reference32)
    i8_mae, i8_rel = error_metrics(y8, reference32)
    
    # Vector-wise accuracy
    vector_mae, vector_rel = error_metrics(y8_vector, reference32)
    i8_rel_l2 = relative_l2_error(y8, reference32)
    vector_rel_l2 = relative_l2_error(y8_vector, reference32)

    # Accuracy after mixed-precision decomposition
    mixed_mae, mixed_rel = error_metrics(y8_mixed, reference32)
    mixed_rel_l2 = relative_l2_error(y8_mixed, reference32)

    # Different scopes: FP16 GEMM only, prequantized INT8 GEMM only,
    # and complete INT8 (both quantizations + GEMM + dequantization).
    fp16_gemm_ms = cuda_time_ms(lambda: fp16_baseline(x, w), warmup, repeat)
    int8_gemm_ms = cuda_time_ms(lambda: torch._int_mm(x8, w8), warmup, repeat)
    int8_full_ms = cuda_time_ms(lambda: int8_baseline(x, w), warmup, repeat)

    # Vector-wise INT8 GEMM only
    vector_gemm_ms = cuda_time_ms(
        lambda: torch._int_mm(x8_vector, w8_vector),
        warmup,
        repeat
    )

    # Vector-wise quantization overhead
    vector_quant_ms = cuda_time_ms(
        lambda: (quantize_absmax_vectorwise(x, dim=1), quantize_absmax_vectorwise(w, dim=0)),
        warmup,
        repeat
    )

    # Vector-wise dequantization overhead
    vector_dequant_ms = cuda_time_ms(
        lambda: (accum_vector.float() * (sx_vector * sw_vector)).half(),
        warmup,
        repeat
    )

    # Complete vector-wise pipeline
    vector_full_ms = cuda_time_ms(lambda: vectorwise_int8_baseline(x, w), warmup, repeat)

    # Mixed-precision stage timings
    mixed_detect_ms = cuda_time_ms(
        lambda: detect_outlier_features(x), warmup, repeat
    )
    mixed_decompose_ms = cuda_time_ms(
        lambda: decompose_features(x, w, outlier_mask), warmup, repeat
    )
    mixed_regular_ms = cuda_time_ms(
        lambda: vectorwise_int8_baseline(x_regular, w_regular), warmup, repeat
    )
    mixed_outlier_ms = (
        cuda_time_ms(lambda: x_outlier @ w_outlier, warmup, repeat)
        if num_outliers > 0 else 0.0
    )

    y8_regular = vectorwise_int8_baseline(x_regular, w_regular)
    y16_outlier = x_outlier @ w_outlier if num_outliers > 0 else None
    mixed_combine_ms = (
        cuda_time_ms(
            lambda: (y8_regular.float() + y16_outlier.float()).half(),
            warmup,
            repeat,
        )
        if num_outliers > 0 else 0.0
    )
    mixed_full_ms = cuda_time_ms(
        lambda: mixed_precision_llm_int8(x, w), warmup, repeat
    )

    # Weight storage
    fp16_weight_mib = w.numel() * w.element_size() / 2**20
    int8_weight_mib = w8.numel() * w8.element_size() / 2**20

    # Include per-column weight scales
    vector_weight_mib = (
        w8_vector.numel() * w8_vector.element_size()
        + sw_vector.numel() * sw_vector.element_size()
    ) / 2**20

    # Dense, zero-masked INT8 weights + vector scales + FP16 outlier rows.
    mixed_weight_mib = (
        w8_vector.numel() * w8_vector.element_size()
        + sw_vector.numel() * sw_vector.element_size()
        + w_outlier.numel() * w_outlier.element_size()
    ) / 2**20

    # Print results
    print(
        f'\nM={m:5d} K={k:5d} N={n:5d} '
        f'outliers={outliers} '
        f'varying_scales={varying_scales} '
        f'zero_row_column={zero_row_column}'
    )

    print('\nExecution Time:')
    print(f'  FP16 GEMM:             {fp16_gemm_ms:.4f} ms')
    print(f'  INT8 GEMM only:        {int8_gemm_ms:.4f} ms')
    print(f'  INT8 full pipeline:    {int8_full_ms:.4f} ms')
    print(f'  Vector-wise GEMM:      {vector_gemm_ms:.4f} ms')
    print(f'  Vector-wise full:      {vector_full_ms:.4f} ms')
    print(f'  Mixed-precision full:  {mixed_full_ms:.4f} ms')

    print('\nVector-wise Breakdown:')
    print(f'  Quantization:          {vector_quant_ms:.4f} ms')
    print(f'  INT8 GEMM:             {vector_gemm_ms:.4f} ms')
    print(f'  Dequantization:        {vector_dequant_ms:.4f} ms')

    print('\nMixed-precision Breakdown:')
    print(f'  Outlier detection:     {mixed_detect_ms:.4f} ms')
    print(f'  Decomposition:         {mixed_decompose_ms:.4f} ms')
    print(f'  Regular INT8 pipeline: {mixed_regular_ms:.4f} ms')
    print(f'  Outlier FP16 GEMM:     {mixed_outlier_ms:.4f} ms')
    print(f'  Output combination:    {mixed_combine_ms:.4f} ms')

    print('\nNumerical Accuracy (vs FP32):')
    print(f'  FP16 MAE:              {f16_mae:.6g}')
    print(f'  FP16 mean rel error:   {f16_rel:.6g}')
    print(f'  INT8 MAE:              {i8_mae:.6g}')
    print(f'  INT8 mean rel error:   {i8_rel:.6g}')
    print(f'  INT8 relative L2:       {i8_rel_l2:.6g}')
    print(f'  Vector-wise MAE:       {vector_mae:.6g}')
    print(f'  Vector-wise mean rel:  {vector_rel:.6g}')
    print(f'  Vector-wise relative L2: {vector_rel_l2:.6g}')

    print(f'  Mixed-precision MAE:   {mixed_mae:.6g}')
    print(f'  Mixed-precision mean rel: {mixed_rel:.6g}')
    print(f'  Mixed-precision relative L2: {mixed_rel_l2:.6g}')

    print('\nWeight-only Storage:')
    print(f'  FP16:                  {fp16_weight_mib:.2f} MiB')
    print(f'  INT8:                  {int8_weight_mib:.2f} MiB')
    print(
        f'  Vector-wise INT8:      {vector_weight_mib:.2f} MiB '
        '(including weight scales)'
    )

    print('\nMixed-precision Outlier Summary:')
    print(f'  Outlier dimensions:    {num_outliers}/{k} ({100.0 * num_outliers / k:.3f}%)')
    if num_outliers <= 12:
        print(f'  Outlier indices:       {outlier_indices.tolist()}')
    print(f'  Mixed weight storage:  {mixed_weight_mib:.2f} MiB (estimated)')
    print('  Storage excludes original FP16 weights and temporary buffers.')

def run_self_tests():
    """Exercise essential outlier cases using CUDA-aligned matrix shapes."""
    torch.manual_seed(479)
    m, k, n = 32, 128, 64
    device = 'cuda'

    def check_case(name, modify_inputs, expected_indices):
        x = torch.randn(m, k, device=device, dtype=torch.float16) * 0.5
        w = torch.randn(k, n, device=device, dtype=torch.float16)
        modify_inputs(x, w)

        outlier_mask = detect_outlier_features(x)
        actual_indices = torch.where(outlier_mask)[0].tolist()
        assert actual_indices == expected_indices, (
            name, actual_indices, expected_indices
        )

        reference32 = x.float() @ w.float()
        validate_decomposition(x, w, outlier_mask, reference32)

        y8_mixed = mixed_precision_llm_int8(x, w)
        assert y8_mixed.shape == (m, n)
        assert y8_mixed.dtype == torch.float16
        assert torch.isfinite(y8_mixed).all()

        if not expected_indices:
            y8_vector = vectorwise_int8_baseline(x, w)
            torch.testing.assert_close(y8_mixed, y8_vector, rtol=0, atol=0)

        if name == 'zero row and column':
            assert torch.count_nonzero(y8_mixed[0, :]) == 0
            assert torch.count_nonzero(y8_mixed[:, 0]) == 0

        print(f'  PASS: {name} (features={actual_indices})')

    check_case('no outliers', lambda x, w: None, [])
    check_case('one positive outlier', lambda x, w: x[:, 0].fill_(40.0), [0])

    def three_outliers(x, w):
        x[:, [0, 5, 10]] = 40.0

    check_case('three outliers', three_outliers, [0, 5, 10])
    check_case('negative outlier', lambda x, w: x[:, 7].fill_(-40.0), [7])

    def zero_row_column(x, w):
        x[0, :] = 0.0
        w[:, 0] = 0.0

    check_case('zero row and column', zero_row_column, [])
    check_case('all outliers', lambda x, w: x.fill_(40.0), list(range(k)))
    print('All Milestone 3 self-tests passed.')

# ============================================================
# Main / CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='run one smaller example instead of three shapes'
    )

    parser.add_argument(
        '--outliers', 
        action='store_true',
        help='inject one synthetic high-magnitude feature'
    )
    
    # Additional test configurations
    parser.add_argument(
        '--varying-scales',
        action='store_true',
        help='test different activation-row and weight-column magnitudes'
    )

    parser.add_argument(
        '--zero-row-column',
        action='store_true',
        help='test zero-valued activation rows and weight columns'
    )

    parser.add_argument('--repeat', type=int, default=40)
    parser.add_argument('--warmup', type=int, default=10)

    # Run deterministic mixed-precision tests
    parser.add_argument(
        '--self-test',
        action='store_true',
        help='run Milestone 3 validation tests instead of benchmarks'
    )
    
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark needs a CUDA GPU (run it on CSIL).')
    
    if not hasattr(torch, '_int_mm'):
        raise RuntimeError('This PyTorch build has no torch._int_mm API.')
    
    # Use full FP32 precision for the accuracy reference (not TF32).
    torch.backends.cuda.matmul.allow_tf32 = False
    
    print('PyTorch:', torch.__version__)
    print('CUDA GPU:', torch.cuda.get_device_name())
    print('CUDA:', torch.version.cuda)
    print('INT8 primitive: torch._int_mm (INT8 inputs, INT32 output)')

    if args.self_test:
        run_self_tests()
        return

    shapes = [(64, 768, 3072)] if args.quick else [
        (64, 768, 3072),
        (128, 1024, 4096),
        (256, 2048, 8192),
    ]
    for shape in shapes:
        benchmark_shape(
            *shape, 
            outliers=args.outliers,
            varying_scales=args.varying_scales,
            zero_row_column=args.zero_row_column,
            warmup=args.warmup, 
            repeat=args.repeat
        )

if __name__ == '__main__':
    main()