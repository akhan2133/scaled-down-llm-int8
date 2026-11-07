# Test script to demonstrate use of custom CUDA kernel with PyTorch

import torch
from cuda_extension import cuda_ops

def main():
    assert torch.cuda.is_available()

    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")

    # Test multiplication
    # TODO: create an outparameter version of this, so that we can more accurately benchmark
    result_multiply = cuda_ops.gemm_naive(a, b)
    torch.testing.assert_close(result_multiply, a @ b, rtol=1e-4, atol=1e-4 )

    print("Passed!")
    print("Device:", result_multiply.device)
    print("Shape:", result_multiply.shape)

if __name__ == "__main__":
    main()