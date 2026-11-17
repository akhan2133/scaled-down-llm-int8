"""
LLM.int8() milestones 1 and 2:
FP16, per-tensor absmax INT8, and vector-wise INT8 GEMM baselines.

Run on a CUDA machine with PyTorch installed:
    python src/main.py
    python src/main.py --quick
    python src/main.py --quick --varying-scales
    python src/main.py --quick --zero-row-column
    python src/main.py --quick --outliers

The INT8 path uses torch._int_mm, an internal PyTorch API.
This performs INT8 x INT8 -> INT32 matrix multiplication.
"""

import argparse
import torch

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

def cuda_time_ms(fn, warmup=10, repeat=40):
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
        # Synthetic extreme feature dimension for a later milestone.
        x[:, 0] = 40.0

    # Test zero-valued rows and columns
    if zero_row_column:
        x[0, :] = 0.0
        w[:, 0] = 0.0

    # FP32 accumulation reference calculated from the SAME rounded FP16 inputs.
    # It isolates GEMM precision and INT8 quantization errors.
    reference32 = x.float() @ w.float()

    # Compute baseline outputs
    y16 = fp16_baseline(x, w)
    y8 = int8_baseline(x, w)

    # Vector-wise output
    y8_vector = vectorwise_int8_baseline(x, w)

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

    # Numerical accuracy
    f16_mae, f16_rel = error_metrics(y16, reference32)
    i8_mae, i8_rel = error_metrics(y8, reference32)
    
    # Vector-wise accuracy
    vector_mae, vector_rel = error_metrics(y8_vector, reference32)
    i8_rel_l2 = relative_l2_error(y8, reference32)
    vector_rel_l2 = relative_l2_error(y8_vector, reference32)

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

    # Weight storage
    fp16_weight_mib = w.numel() * w.element_size() / 2**20
    int8_weight_mib = w8.numel() * w8.element_size() / 2**20

    # Include per-column weight scales
    vector_weight_mib = (
        w8_vector.numel() * w8_vector.element_size()
        + sw_vector.numel() * sw_vector.element_size()
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

    print('\nVector-wise Breakdown:')
    print(f'  Quantization:          {vector_quant_ms:.4f} ms')
    print(f'  INT8 GEMM:             {vector_gemm_ms:.4f} ms')
    print(f'  Dequantization:        {vector_dequant_ms:.4f} ms')

    print('\nNumerical Accuracy (vs FP32):')
    print(f'  FP16 MAE:              {f16_mae:.6g}')
    print(f'  FP16 mean rel error:   {f16_rel:.6g}')
    print(f'  INT8 MAE:              {i8_mae:.6g}')
    print(f'  INT8 mean rel error:   {i8_rel:.6g}')
    print(f'  INT8 relative L2:       {i8_rel_l2:.6g}')
    print(f'  Vector-wise MAE:       {vector_mae:.6g}')
    print(f'  Vector-wise mean rel:  {vector_rel:.6g}')
    print(f'  Vector-wise relative L2: {vector_rel_l2:.6g}')

    print('\nWeight-only Storage:')
    print(f'  FP16:                  {fp16_weight_mib:.2f} MiB')
    print(f'  INT8:                  {int8_weight_mib:.2f} MiB')
    print(
        f'  Vector-wise INT8:      {vector_weight_mib:.2f} MiB '
        '(including weight scales)'
    )

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