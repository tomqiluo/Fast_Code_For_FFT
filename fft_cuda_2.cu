#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <complex.h>
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
    extern __shared__ cuDoubleComplex temp[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * gridDim.x;

    for (int len = 2; len <= n; len <<= 1) {
        int halfLen = len / 2;
        double ang = 2 * M_PI * (invert ? -1 : 1) / len;
        cuDoubleComplex wlen = make_cuDoubleComplex(cos(ang), sin(ang));

        for (int i = tid; i < n; i += numThreads * len) {
            for (int j = 0; j < halfLen; ++j) {
                int index1 = i + j;
                int index2 = i + j + halfLen;

                // Load the data into shared memory
                temp[threadIdx.x * 2] = a[index1];
                temp[threadIdx.x * 2 + 1] = a[index2];

                __syncthreads(); // Make sure all writes to shared memory are done

                cuDoubleComplex u = temp[threadIdx.x * 2];
                cuDoubleComplex t = temp[threadIdx.x * 2 + 1];
                cuDoubleComplex w = make_cuDoubleComplex(1, 0);
                
                // Perform the twiddle factor multiplication on a single element
                cuDoubleComplex twiddled = cuCmul(w, t);

                // Save the results back into global memory
                a[index1] = cuCadd(u, twiddled);
                a[index2] = cuCsub(u, twiddled);

                // Update the twiddle factor
                if (j != 0) {
                    w = cuCmul(w, wlen);
                }
                
                __syncthreads(); // Sync before the next iteration
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

    int sharedSize = sizeof(cuDoubleComplex) * block.x * 2; // Allocate twice the block size for pair storage
    fftKernel<<<grid, block, sharedSize>>>(d_a, n, invert);

    if (invert) {
        normalize<<<grid, block>>>(d_a, n);
    }

    cudaMemcpy(h_a, d_a, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_temp);
}

void generateComplexData(cuDoubleComplex *a, int n) {
    for (int i = 0; i < n; ++i) {
        double real = static_cast<double>(rand()) / RAND_MAX;
        double imag = static_cast<double>(rand()) / RAND_MAX;
        a[i] = make_cuDoubleComplex(real, imag);
    }
}

int main() {
    int sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    cudaFree(0);  // Initialize CUDA

    for (int i = 0; i < num_sizes; ++i) {
        int n = sizes[i];
        cuDoubleComplex *data = (cuDoubleComplex *)malloc(n * sizeof(cuDoubleComplex));

        generateComplexData(data, n);

        struct timeval start, end;
        gettimeofday(&start, NULL);
        
        fft(data, n, false);
        
        gettimeofday(&end, NULL);
        long seconds = (end.tv_sec - start.tv_sec);
        long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

        printf("FFT size %d - Execution time: %ld microseconds\n", n, micros);

        free(data);
    }

    return 0;
}
