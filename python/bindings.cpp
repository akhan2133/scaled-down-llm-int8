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

torch::Tensor gemm_naive(
    torch::Tensor a,
    torch::Tensor b
) {
    // Input validation
    TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor b must be a CUDA tensor");
    TORCH_CHECK(a.device() == b.device(), "Inputs must be on the same device");
    TORCH_CHECK(a.is_contiguous(), "Input tensor a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Input tensor b must be contiguous");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensor a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Input tensor b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");

    TORCH_CHECK(
        a.numel() <= std::numeric_limits<int>::max(),
        "Too many elements"
    );

    // Ensure that the active CUDA device matches tensor a
    c10::cuda::CUDAGuard guard(a.device());

    // Create the output tensor on the same device as the input
    auto c = torch::empty_like(a);
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

    return c;
}

// Binding function to expose C++ functions in Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_naive", &gemm_naive, "CUDA vector addition");
    // Add other bindings here as needed...
}