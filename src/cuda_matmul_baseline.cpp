#include <iostream>
#include <vector>

#include <cuda/cmath>

// 2D block dimensions
const dim3 DIM_BLOCK( 32, 32, 1 );


__global__
void matrixMulKernel( float *M, float *N, float *P, int width ) {
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

int main() {
    // 1. Memory allocations in host and device memory
    const int matrixDim = 2000;
    const size_t numElements = matrixDim * matrixDim;

    // Host
    std::vector< float > A_h( numElements );
    std::vector< float > B_h( numElements );
    std::vector< float > C_h( numElements );

    // Device
    float *A_d, *B_d, *C_d;
    size_t size = static_cast< size_t >( numElements ) * sizeof( float );
    cudaError_t err = cudaMalloc( (void **) &A_d, size );
    if ( err != cudaSuccess ) {
        printf( "%s in %s at line %d \n", cudaGetErrorString( err ), __FILE__, __LINE__ );
        exit( EXIT_FAILURE );
    }

    err = cudaMalloc( (void **)&B_d, size );
    if ( err != cudaSuccess ) {
        printf( "%s in %s at line %d \n", cudaGetErrorString( err ), __FILE__, __LINE__ );
        exit( EXIT_FAILURE );
    }

    err = cudaMalloc( (void **)&C_d, size );
    if ( err != cudaSuccess ) {
        printf( "%s in %s at line %d \n", cudaGetErrorString( err ), __FILE__, __LINE__ );
        exit( EXIT_FAILURE );
    }


    // 2: Generate two input matrices
    for ( int row = 0; row < matrixDim; ++row ) {
        float value = ( row % 2 == 0 ) ? 0.005f : 0.004f;

        for ( int col = 0; col < matrixDim; ++col ) {
            int index = ( row * matrixDim ) + col;
            A_h[ index ] = value;
            B_h[ index ] = value;
        }
    }


    // 3. Copy the two input matrices from host to device memory
    err = cudaMemcpy( A_d, A_h.data(), size, cudaMemcpyHostToDevice );
    if (err != cudaSuccess) {
        std::cerr << "Copy failed: " << cudaGetErrorString(err) << "\n";
        return EXIT_FAILURE;
    }

    err = cudaMemcpy( B_d, B_h.data(), size, cudaMemcpyHostToDevice );
    if (err != cudaSuccess) {
        std::cerr << "Copy failed: " << cudaGetErrorString(err) << "\n";
        return EXIT_FAILURE;
    }

    // 4. Call the device kernel
    dim3 dimGrid(
        cuda::ceil_div( matrixDim, DIM_BLOCK.x ),
        cuda::ceil_div( matrixDim, DIM_BLOCK.y ),
        1
    );

    std::cout << "Block: ("
          << DIM_BLOCK.x << ", "
          << DIM_BLOCK.y << ", "
          << DIM_BLOCK.z << ")\n";

    std::cout << "Grid: ("
            << dimGrid.x << ", "
            << dimGrid.y << ", "
            << dimGrid.z << ")\n";

    std::cout << "Threads/block: "
            << DIM_BLOCK.x * DIM_BLOCK.y * DIM_BLOCK.z << '\n';

    // Time Kernel
    cudaStream_t stream;
    cudaStreamCreate( &stream );

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    cudaEventRecord( start, stream );
    matrixMulKernel<<<dimGrid, DIM_BLOCK, 0, stream>>>( A_d, B_d, C_d, matrixDim );
    cudaEventRecord( stop, stream );

    err = cudaGetLastError();
    if ( err != cudaSuccess ) {
        std::cerr << "Kernel Error: " << cudaGetErrorString( err ) << "\n";
    }

    err = cudaStreamSynchronize( stream );
    if ( err != cudaSuccess ) {
        std::cerr << "Kernel Error: " << cudaGetErrorString( err ) << "\n";
    }


    // 5. Copy the output matrix from device to host memory
    err = cudaMemcpy( C_h.data(), C_d, size, cudaMemcpyDeviceToHost );
    if ( err != cudaSuccess ) {
        std::cerr << "Output copy failed: " << cudaGetErrorString( err ) << "\n";
    }


    // 6. Print and verify the values of the output matrix at a few indices
    auto verifyValue = [&]( int x, int y ) {
        float expected = 0.0f;
        for ( int k = 0; k < matrixDim; ++k ) {
            expected += A_h[ ( y * matrixDim ) + k ] * B_h[ ( k * matrixDim ) + x ];
        }

        float actual = C_h[ ( y * matrixDim ) + x ];
        bool pass = std::abs(actual - expected) < 1e-5f;
        std::cout << ( pass ? "PASS" : "FAIL" )
                  << ": C(" << x << ", " << y << ")" << " = " << actual
                  << ", expected = " << expected << "\n";
        return pass;
    };
    verifyValue( 0, 0 );
    verifyValue( 0, 1 );
    verifyValue( 1, 0 );
    verifyValue( 1, 1 );
    verifyValue( 1999, 1999 );
    verifyValue( 1999, 1998 );
    verifyValue( 1998, 1999 );
    verifyValue( 1998, 1998 );


    // 7. Free all allocated memory
    cudaFree( A_d );
    cudaFree( B_d );
    cudaFree( C_d );


    // 8. Print out the time spent in step 4
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );
    std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;

    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    cudaStreamDestroy( stream );

    return 0;
}