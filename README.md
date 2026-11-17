# LLM.int8()

**Optimizing a Scaled-Down LLM.int8() Implementation for RTX 2000 GPUs**

Based on [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) (Dettmers et al., NeurIPS 2022).

## Overview

This project explores the performance, memory efficiency, and numerical accuracy of INT8 matrix multiplication for transformer inference on NVIDIA RTX 2000 Ada GPUs.

We aim to implement the two main components of LLM.int8():

- **Vector-wise quantization:** Independently quantizing activation rows and weight columns using separate scaling factors.
- **Mixed-precision decomposition:** Separating outlier feature dimensions for FP16 computation while performing the remaining matrix multiplication in INT8.

We will then investigate CUDA optimizations to reduce overhead from quantization, outlier processing, dequantization, and intermediate memory accesses.

## Current Implementation

### Python / PyTorch (`src/main.py`)

Implements the FP16, basic INT8, and vector-wise INT8 baselines:

- FP16 matrix multiplication using PyTorch.
- Per-tensor absmax INT8 quantization.
- Vector-wise absmax INT8 quantization with independent activation-row and weight-column scaling factors.
- INT8 matrix multiplication with INT32 accumulation using `torch._int_mm()`.
- Dequantization of INT32 results to FP16 using the corresponding scaling factors.
- Numerical validation against an FP32 reference using mean absolute error, mean relative error, and relative L2 error.
- CUDA event benchmarking for FP16 GEMM, INT8 GEMM, and complete INT8 pipelines.
- Separate timing measurements for vector-wise quantization, INT8 GEMM, and dequantization.
- FP16 versus INT8 weight storage comparisons, including vector-wise scaling-factor overhead.
- Synthetic input tests with varying row/column magnitudes and zero-valued rows/columns.
- Optional synthetic outlier injection for later mixed-precision decomposition experiments.

### CUDA C++ (`src/cuda_matmul_baseline.cpp`)

Contains an initial CUDA matrix multiplication prototype:

- Naive FP32 matrix multiplication using a custom CUDA kernel.
- GPU memory allocation and host-to-device/device-to-host transfers.
- CUDA event timing and basic numerical verification.

This implementation serves as an initial CUDA development prototype and does not yet implement the FP16 or INT8 baselines.

## Project Structure

```text
LLM-int8/
├── CMakeLists.txt
├── README.md
├── requirements.txt
├── setup.sh
└── src/
    ├── cuda_matmul_baseline.cpp
    └── main.py
```

## Getting Started

### Requirements

- NVIDIA GPU with CUDA support
- Python 3 and PyTorch with CUDA support
- CMake 3.24+
- C++20 and CUDA C++20 compatible compiler/toolkit

### Python / PyTorch

Install dependencies and configure the Python environment:

```bash
bash setup.sh
```

The current setup script creates a virtual environment at `/tmp/${USER}-llm-int8-venv`, installs the dependencies from `requirements.txt`, and checks CUDA availability.

Activate the environment:

```bash
source /tmp/${USER}-llm-int8-venv/bin/activate
```

Run the benchmarks:

```bash
python src/main.py
```

Additional benchmark options:

```bash
# Run a smaller benchmark
python src/main.py --quick

# Test different activation-row and weight-column magnitudes
python src/main.py --quick --varying-scales

# Test zero-valued activation rows and weight columns
python src/main.py --quick --zero-row-column

# Test with synthetic outliers
python src/main.py --outliers

# Adjust benchmarking repetitions
python src/main.py --warmup 10 --repeat 100
```

**Note:** The INT8 implementation uses PyTorch's internal `torch._int_mm()` API and requires a compatible CUDA-enabled PyTorch build.

### CUDA C++

Configure and build the CUDA implementation:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run the executable:

```bash
./build/llm_int8
```

Alternatively, build and execute using the convenience target:

```bash
cmake --build build --target run -j
```

The current CUDA program performs naive FP32 matrix multiplication on square matrices of dimension 2000 × 2000.

## Preliminary Benchmark Results

Benchmark environment:

- GPU: NVIDIA RTX 2000 Ada Generation
- PyTorch: 2.15.0+cu132
- CUDA: 13.2
- Default benchmark: 10 warmup iterations and 40 timed iterations
- Random seed: 479

### Execution Time

All times are measured in milliseconds using CUDA events.

| Matrix (M, K, N) | FP16 GEMM | INT8 GEMM | Full INT8 | Vector-wise GEMM | Full Vector-wise |
|---|---:|---:|---:|---:|---:|
| 64, 768, 3072 | 0.0242 | 0.0227 | 0.2582 | 0.0227 | 0.2653 |
| 128, 1024, 4096 | 0.0329 | 0.0571 | 0.7511 | 0.0572 | 0.7676 |
| 256, 2048, 8192 | 0.2239 | 0.4136 | 4.3688 | 0.3732 | 4.4183 |

GEMM-only measurements exclude quantization and dequantization. Full INT8 pipeline measurements include quantization of both input matrices, INT8 matrix multiplication, and dequantization to FP16.

The complete vector-wise INT8 pipeline is slower than FP16 GEMM for all tested matrix sizes.

### Vector-wise Pipeline Breakdown

| Matrix (M, K, N) | Quantization (ms) | INT8 GEMM (ms) | Dequantization (ms) |
|---|---:|---:|---:|
| 64, 768, 3072 | 0.2189 | 0.0227 | 0.0142 |
| 128, 1024, 4096 | 0.6690 | 0.0572 | 0.0257 |
| 256, 2048, 8192 | 3.8924 | 0.3732 | 0.1832 |

Quantization accounts for approximately 83–88% of the measured full-pipeline execution time across the tested matrix sizes.

These results indicate that quantization overhead is a major performance bottleneck and a potential target for future CUDA optimization.

The current implementation requantizes weights on every invocation. Prequantizing and reusing weights could reduce this overhead in a practical inference implementation.

### Numerical Accuracy

Outputs are compared against an FP32 matrix multiplication reference calculated from the same FP16 inputs.

| Matrix (M, K, N) | Per-tensor INT8 Relative L2 | Vector-wise INT8 Relative L2 |
|---|---:|---:|
| 64, 768, 3072 | 0.015593 | 0.010818 |
| 128, 1024, 4096 | 0.015884 | 0.011146 |
| 256, 2048, 8192 | 0.016500 | 0.011689 |

Vector-wise quantization reduced relative L2 error by approximately 29–31% compared with per-tensor INT8 quantization across all three matrix sizes.

### Varying-Scales Experiment

To evaluate the effect of independent scaling factors, activation rows and weight columns were assigned different magnitudes using logarithmically spaced scaling factors.

Results for matrix dimensions (64, 768, 3072):

| Metric | Per-tensor INT8 | Vector-wise INT8 |
|---|---:|---:|
| Mean absolute error | 7.56592 | 1.12593 |
| Mean relative error | 2.17849 | 0.08161 |
| Relative L2 error | 0.039327 | 0.010680 |

Vector-wise quantization reduced relative L2 error by approximately 72.8% and mean absolute error by approximately 85.1% compared with per-tensor quantization.

This demonstrates the benefit of independently quantizing activation rows and weight columns when their magnitudes differ substantially.

### Numerical Validation

The vector-wise implementation was validated using:

- Normally distributed random matrices.
- Matrices with varying activation-row and weight-column magnitudes.
- Matrices containing zero-valued activation rows and weight columns.

Validation checks confirm:

- Correct activation-row and weight-column scaling dimensions.
- INT8 quantized operands and INT32 accumulation.
- FP16 output dimensions and datatype.
- Finite scaling factors and output values.
- Correct zero-output behavior for zero-valued rows and columns.

### Weight Storage

| Matrix (M, K, N) | FP16 Weights | Per-tensor INT8 | Vector-wise INT8 |
|---|---:|---:|---:|
| 64, 768, 3072 | 4.50 MiB | 2.25 MiB | 2.26 MiB |
| 128, 1024, 4096 | 8.00 MiB | 4.00 MiB | 4.02 MiB |
| 256, 2048, 8192 | 32.00 MiB | 16.00 MiB | 16.03 MiB |

Vector-wise INT8 weight storage includes the FP32 scaling factors for each weight column.

INT8 weight storage is approximately half that of FP16 weights, with a small additional cost for scaling factors.

These values represent weight storage only and do not include activation tensors, intermediate buffers, or peak GPU memory usage.

## Development Roadmap

- [x] Implement FP16 and basic INT8 baselines (PyTorch)
- [x] Implement and validate vector-wise quantization
- [ ] Implement and validate mixed-precision decomposition
- [ ] Profile and optimize CUDA operations
- [ ] Complete performance and accuracy evaluation
- [ ] Finalize report, presentation, and code

## Reference

Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS 2022.

https://arxiv.org/abs/2208.07339