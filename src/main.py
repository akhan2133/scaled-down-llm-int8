"""
LLM.int8() milestone 1: FP16 and per-tensor absmax INT8 GEMM baselines.

Run on a CUDA machine with PyTorch installed:
    python main.py
    python main.py --quick

The INT8 path uses torch._int_mm, an internal PyTorch API. This performs
INT8 x INT8 -> INT32 matrix multiplication. Check your PyTorch build if
this internal API is unavailable on your system.
"""

import argparse
import torch

def quantize_absmax_per_tensor(x: torch.Tensor):
    """Return (signed INT8 tensor, FLOAT32 dequantization step)."""
    xf = x.float()
    scale = (xf.abs().amax() / 127.0).clamp_min(1e-12)
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

def benchmark_shape(m, k, n, *, outliers=False, warmup=10, repeat=40):
    torch.manual_seed(479)
    device = 'cuda'
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(k, n, device=device, dtype=torch.float16)
    if outliers:
        # Synthetic extreme feature dimension for a later milestone.
        x[:, 0] = 40.0

    # FP32 accumulation reference calculated from the SAME rounded FP16 inputs.
    # It isolates GEMM precision and INT8 quantization errors.
    reference32 = x.float() @ w.float()
    y16 = fp16_baseline(x, w)
    y8 = int8_baseline(x, w)

    x8, sx = quantize_absmax_per_tensor(x)
    w8, sw = quantize_absmax_per_tensor(w)
    accum32 = torch._int_mm(x8, w8)
    assert accum32.dtype == torch.int32
    assert x8.dtype == torch.int8 and w8.dtype == torch.int8
    assert y8.dtype == torch.float16

    f16_mae, f16_rel = error_metrics(y16, reference32)
    i8_mae, i8_rel = error_metrics(y8, reference32)

    # Different scopes: FP16 GEMM only, prequantized INT8 GEMM only,
    # and complete INT8 (both quantizations + GEMM + dequantization).
    fp16_gemm_ms = cuda_time_ms(lambda: fp16_baseline(x, w), warmup, repeat)
    int8_gemm_ms = cuda_time_ms(lambda: torch._int_mm(x8, w8), warmup, repeat)
    int8_full_ms = cuda_time_ms(lambda: int8_baseline(x, w), warmup, repeat)

    fp16_weight_mib = w.numel() * w.element_size() / 2**20
    int8_weight_mib = w8.numel() * w8.element_size() / 2**20
    print(f'\nM={m:5d} K={k:5d} N={n:5d}  outliers={outliers}')
    print(f'  FP16 GEMM:           {fp16_gemm_ms:.4f} ms')
    print(f'  INT8 GEMM only:      {int8_gemm_ms:.4f} ms')
    print(f'  INT8 full pipeline:  {int8_full_ms:.4f} ms')
    print(f'  FP16 MAE:            {f16_mae:.6g}')
    print(f'  FP16 mean rel error: {f16_rel:.6g}')
    print(f'  INT8 MAE:            {i8_mae:.6g}')
    print(f'  INT8 mean rel error: {i8_rel:.6g}')
    print(f'  Weight-only storage: FP16={fp16_weight_mib:.2f} MiB, '
          f'INT8={int8_weight_mib:.2f} MiB (+ scaling factor)')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='run one smaller example instead of three shapes')
    parser.add_argument('--outliers', action='store_true',
                        help='inject one synthetic high-magnitude feature')
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
        benchmark_shape(*shape, outliers=args.outliers,
                        warmup=args.warmup, repeat=args.repeat)

if __name__ == '__main__':
    main()