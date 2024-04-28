// nvcc -o fft_cuda fft_cuda.cu -arch=compute_75 -code=sm_75 -use_fast_math -O3
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>

__device__ int reverseBits(int num, int log2n) {
    int reversed = 0;
    for (int i = 0; i < log2n; i++) {
        reversed = (reversed << 1) | (num & 1);
        num >>= 1;
    }
    return reversed;
}

__global__ void bitReverseCopy(cuDoubleComplex* a, cuDoubleComplex* b, int n, int log2n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int rev = reverseBits(tid, log2n);
        b[rev] = a[tid];
    }
}

__global__ void fftKernel(cuDoubleComplex* a, int n, bool invert) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * gridDim.x;;

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * 3.1415926 * (invert ? -1 : 1) / (double)len;
        cuDoubleComplex wlen = make_cuDoubleComplex(cos(ang), sin(ang));
        for (int i = tid; i < n; i += numThreads * len) {
            cuDoubleComplex w = make_cuDoubleComplex(1, 0);
            for (int j = 0; j < len>>1; j++) {
                cuDoubleComplex u = a[i + j];
                cuDoubleComplex v = cuCmul(a[i + j + len>>1], w);
                a[i + j] = cuCadd(u, v);
                a[i + j + len>>1] = cuCsub(u, v);
                w = cuCmul(w, wlen);
            }
        }
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
    dim3 block(1024);
    dim3 grid((n + block.x - 1) / block.x);

    cuDoubleComplex *d_temp;
    cudaMalloc(&d_temp, n * sizeof(cuDoubleComplex));
    bitReverseCopy<<<grid, block>>>(d_a, d_temp, n, log2n);
    cudaMemcpy(d_a, d_temp, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);

    fftKernel<<<grid, block>>>(d_a, n, invert);

    if (invert) {
        normalize<<<grid, block>>>(d_a, n);
    }

    cudaMemcpy(h_a, d_a, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_temp);
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