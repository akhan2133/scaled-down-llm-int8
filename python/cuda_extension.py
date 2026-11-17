"""
Build and load custom CUDA kernels as a PyTorch C++ extension.

Uses PyTorch JIT extension loader to compile C++ bindings and CUDA source files
with Ninja. Compiled artifacts are cached and reused when the sources and build
configuration have not changed, and can be found at:
/home/${USER}/.cache/torch_extensions/py312_cu132/llm_int8_cuda/
"""

from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent.parent

cuda_ops = load(
    name="llm_int8_cuda",
    sources=[
        str(ROOT / "python/bindings.cpp"),
        str(ROOT / "src/kernels/gemm_naive.cu"),
    ],
    extra_include_paths=[
        str(ROOT / "include"),
    ],
    extra_cflags=[
        "-O3",
        "-std=c++20",
    ],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
    ],
    verbose=True,
)