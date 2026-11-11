# Test script to demonstrate use of custom CUDA kernel with PyTorch

import torch
from cuda_extension import cuda_ops

def main():
    assert torch.cuda.is_available()

    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    c = torch.empty_like(a)

    # Test multiplication
    cuda_ops.gemm_naive(a, b, c)
    torch.testing.assert_close(c, a @ b, rtol=1e-4, atol=1e-4 )

    print("Passed!")
    print("Device:", c.device)
    print("Shape:", c.shape)

if __name__ == "__main__":
    main()