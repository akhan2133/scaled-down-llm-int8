// Declarations for all CUDA kernel launcher functions

#pragma once

void gemmNaiveLauncher(
    float *A_d,
    float *B_d,
    float *C_d,
    int matrixDim,
    cudaStream_t stream
);