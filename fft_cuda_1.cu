#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <curand_kernel.h>

// Add the following information in comment:
// export PATH=$PATH:/usr/local/cuda-11.4/bin
// export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.4/lib64
// Compile: nvcc -o fft_cuda_1 fft_cuda_1.cu -lcufft -lcurand
// ./fft_cuda_1

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void initCurand(curandState *state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void generateComplexData(cufftDoubleComplex *a, int n, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx].x = curand_uniform_double(&states[idx]) * 2.0 - 1.0; // Real part
        a[idx].y = curand_uniform_double(&states[idx]) * 2.0 - 1.0; // Imaginary part
    }
}

int main() {
    int sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int i = 0; i < num_sizes; ++i) {
        int n = sizes[i];
        cufftDoubleComplex *data;
        curandState *states;
        cudaMalloc(&data, n * sizeof(cufftDoubleComplex));
        cudaMalloc(&states, n * sizeof(curandState));

        initCurand<<<(n + 255)/256, 256>>>(states, time(NULL));
        generateComplexData<<<(n + 255)/256, 256>>>(data, n, states);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        cufftHandle plan;
        if (cufftPlan1d(&plan, n, CUFFT_Z2Z, 1) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: Plan creation failed");
            return EXIT_FAILURE;
        }

        struct timeval start, end;
        gettimeofday(&start, NULL);

        if (cufftExecZ2Z(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: ExecZ2Z failed");
            return EXIT_FAILURE;
        }
        gpuErrchk(cudaDeviceSynchronize());

        gettimeofday(&end, NULL);
        long seconds = (end.tv_sec - start.tv_sec);
        long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

        printf("FFT size %d - Execution time: %ld microseconds\n", n, micros);

        cufftDestroy(plan);
        cudaFree(data);
        cudaFree(states);
    }

    return 0;
}