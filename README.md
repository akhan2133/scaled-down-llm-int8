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

Implements the initial FP16 and basic INT8 baselines:

- FP16 matrix multiplication using PyTorch.
- Per-tensor absmax INT8 quantization.
- INT8 matrix multiplication with INT32 accumulation using `torch._int_mm()`.
- Dequantization of INT32 results to FP16.
- Numerical validation against an FP32 reference using mean absolute and relative error.
- CUDA event benchmarking for FP16 GEMM, INT8 GEMM, and the full INT8 pipeline.
- FP16 versus INT8 weight storage comparisons.
- Optional synthetic outlier injection.

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

Results obtained on the NVIDIA RTX 2000 Ada Generation GPU using the Python/PyTorch implementation.

| Matrix (M, K, N) | FP16 GEMM (ms) | INT8 GEMM (ms) | Full INT8 (ms) |
|---|---:|---:|---:|
| 64, 768, 3072 | 0.0257 | 0.0227 | 0.2619 |
| 128, 1024, 4096 | 0.0330 | 0.0571 | 0.7500 |
| 256, 2048, 8192 | 0.2237 | 0.4171 | 4.3917 |

Basic INT8 quantization reduces weight storage by approximately 50%, excluding scaling-factor overhead. However, the full INT8 pipeline is slower than FP16 for all tested matrix sizes and introduces greater numerical error.

These preliminary results motivate further investigation into quantization precision and CUDA optimization.

## Development Roadmap

- [x] Implement FP16 and basic INT8 baselines (PyTorch)
- [ ] Implement and validate vector-wise quantization
- [ ] Implement and validate mixed-precision decomposition
- [ ] Profile and optimize CUDA operations
- [ ] Complete performance and accuracy evaluation
- [ ] Finalize report, presentation, and code

## Reference

Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS 2022.

https://arxiv.org/abs/2208.07339