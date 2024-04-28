// nvcc -o fft_cuda fft_cuda.cu -arch=compute_75 -code=sm_75 -use_fast_math -O3
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

define BLOCK_SIZE 4096 // Adjust this value based on the GPU

__device__ int reverseBits(int num, int log2n) {
    int reversed = 0;
    for (int i = 0; i < log2n; i++) {
        reversed = (reversed << 1) | (num & 1);
        num >>= 1;
    }
    return reversed;
}

__global__ void bitReverse(cuDoubleComplex* a, int n, int log2n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int rev = reverseBits(tid, log2n);
        if (tid < rev) {
            cuDoubleComplex temp = a[tid];
            a[tid] = a[rev];
            a[rev] = temp;
        }
    }
}


__global__ void fftKernel(cuDoubleComplex* a, int n, bool invert) {
    extern __shared__ cuDoubleComplex shared_a[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    // Load data into shared memory
    if (tid < n) {
        shared_a[local_id] = a[tid];
    }
    __syncthreads();

    int numThreads = blockDim.x * gridDim.x;
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * M_PI * (invert ? -1 : 1) / len;
        cuDoubleComplex wlen = make_cuDoubleComplex(cos(ang), sin(ang));

        for (int i = local_id; i < n; i += numThreads * len) {
            cuDoubleComplex w = make_cuDoubleComplex(1, 0);
            for (int j = 0; j < len / 2; j++) {
                cuDoubleComplex u = shared_a[i + j];
                cuDoubleComplex v = cuCmul(shared_a[i + j + len / 2], w);
                shared_a[i + j] = cuCadd(u, v);
                shared_a[i + j + len / 2] = cuCsub(u, v);
                w = cuCmul(w, wlen);
            }
        }
        __syncthreads();
    }

    // Write data back to global memory
    if (tid < n) {
        a[tid] = shared_a[local_id];
    }
}


__global__ void normalize(cuDoubleComplex* a, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        a[index] = cuCdiv(a[index], make_cuDoubleComplex(n, 0));
    }
}

void fft(cuDoubleComplex *h_a, int n, bool invert) {
    cuDoubleComplex *d_a;
    cudaMalloc(&d_a, n * sizeof(cuDoubleComplex));
    cudaMemcpy(d_a, h_a, n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    int log2n = log2((double)n);
    int blockSize = BLOCK_SIZE;
    int numBlocks = (n + blockSize - 1) / blockSize;
    size_t sharedSize = blockSize * sizeof(cuDoubleComplex);

    bitReverse<<<numBlocks, blockSize>>>(d_a, n, log2n);
    fftKernel<<<numBlocks, blockSize, sharedSize>>>(d_a, n, false);

    if (invert) {
        normalize<<<numBlocks, blockSize, sharedSize>>>(d_a, n);
    }

    cudaMemcpy(h_a, d_a, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
}

// Function to generate complex data
void generateComplexData(cuDoubleComplex *a, int n) {
    for (int i = 0; i < n; ++i) {
        double real = static_cast<double>(rand()) / RAND_MAX;
        double imag = static_cast<double>(rand()) / RAND_MAX;
        a[i] = make_cuDoubleComplex(real, imag);
    }
}

int main() {
    int sizes[] = {256, 512, 1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // Initialize cuda
    cudaFree(0);

    for (int i = 0; i < num_sizes; ++i) {
        int n = sizes[i];
        cuDoubleComplex *data = (cuDoubleComplex *)malloc(n * sizeof(cuDoubleComplex));

        // Generate data
        generateComplexData(data, n);

        // Measure start time
        struct timeval start, end;
        gettimeofday(&start, NULL);
        
        // Run FFT
        fft(data, n, false);
        
        // Measure end time
        gettimeofday(&end, NULL);
        long seconds = (end.tv_sec - start.tv_sec);
        long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

        printf("FFT size %d - Execution time: %ld microseconds\n", n, micros);

        free(data);
    }

    return 0;
}