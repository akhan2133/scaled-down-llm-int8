// Bindings to bridge PyTorch with our C++ interfaces.
//
// CUDA launcher methods should be declared in include/kernels.h, and
// interfaces for these functions (adhering to the PyTorch C++ API)
// should be defined and bound here.
//
// based on: https://apxml.com/courses/advanced-pytorch/chapter-6-custom-extensions-interoperability/custom-cuda-extensions

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <limits>

#include "kernels.h"

// Write output to tensor c
void gemm_naive(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& c
) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(),
                "All tensors must be CUDA tensors");

    TORCH_CHECK(a.device() == b.device() && a.device() == c.device(),
                "All tensors must be on the same device");

    TORCH_CHECK(a.scalar_type() == torch::kFloat32 &&
                b.scalar_type() == torch::kFloat32 &&
                c.scalar_type() == torch::kFloat32,
                "All tensors must be float32");

    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2,
                "All tensors must be 2D");

    TORCH_CHECK(a.size(0) == a.size(1),
                "Matrices must be square");

    TORCH_CHECK(a.sizes() == b.sizes() && a.sizes() == c.sizes(),
                "Matrix dimensions must match");

    TORCH_CHECK(a.is_contiguous() &&
                b.is_contiguous() &&
                c.is_contiguous(),
                "All tensors must be contiguous");

    TORCH_CHECK(a.size(0) <= std::numeric_limits<int>::max(),
                "Matrix dimension exceeds supported range");

    TORCH_CHECK(
        !a.is_alias_of(c) && !b.is_alias_of(c),
        "Output must not alias an input tensor"
    );

    // Ensure that the active CUDA device matches tensor a
    c10::cuda::CUDAGuard guard(a.device());

    auto stream = at::cuda::getCurrentCUDAStream();

    if (a.numel() > 0) {
        gemmNaiveLauncher(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            c.data_ptr<float>(),
            static_cast<int>(a.size(0)),
            stream.stream()
        );

        // Throws if the kernel launch produced an error
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// Binding function to expose C++ functions in Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_naive", &gemm_naive, "CUDA vector addition");
    // Add other bindings here as needed...
}