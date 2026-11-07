
// Naive approach to matrix multiplication, from HW2

#include "kernels.h"

#include <iostream>

#include <cuda_runtime.h>
#include <cuda/cmath>

__global__
void gemmNaiveKernel( float *M, float *N, float *P, int width ) {
    int row = ( blockIdx.y * blockDim.y ) + threadIdx.y;
    int col = ( blockIdx.x * blockDim.x ) + threadIdx.x;

    if ( row < width && col < width ) {
        float pValue = 0.0f;
        for ( int k = 0; k < width; ++k ) {
            pValue += M[ ( row * width ) + k ] * N[ ( k * width ) + col ];
        }
        P[ ( row * width ) + col ] = pValue;
    }
}

void gemmNaiveLauncher(
    float *A_d,
    float *B_d,
    float *C_d,
    int matrixDim,
    cudaStream_t stream
) {
    // 2D block dimensions
    const dim3 DIM_BLOCK( 32, 32, 1 );

    dim3 dimGrid(
        cuda::ceil_div( matrixDim, DIM_BLOCK.x ),
        cuda::ceil_div( matrixDim, DIM_BLOCK.y ),
        1
    );

    gemmNaiveKernel<<<dimGrid, DIM_BLOCK, 0, stream>>>( A_d, B_d, C_d, matrixDim );
}
