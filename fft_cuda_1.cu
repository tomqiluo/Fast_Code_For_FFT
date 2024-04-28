#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Error check macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Kernel to initialize complex data
__global__ void generateComplexData(cufftDoubleComplex *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx].x = rand() / (double)RAND_MAX; // Real part
        a[idx].y = rand() / (double)RAND_MAX; // Imaginary part
    }
}

int main() {
    int sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; ++i) {
        int n = sizes[i];
        cufftDoubleComplex *data;
        cufftHandle plan;
        cudaMalloc(&data, n * sizeof(cufftDoubleComplex));

        // Generate data on the GPU
        generateComplexData<<<(n + 255)/256, 256>>>(data, n);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // Create plan
        if (cufftPlan1d(&plan, n, CUFFT_Z2Z, 1) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: Plan creation failed");
            return EXIT_FAILURE;
        }

        // Measure start time
        struct timeval start, end;
        gettimeofday(&start, NULL);

        // Execute FFT
        if (cufftExecZ2Z(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: ExecZ2Z failed");
            return EXIT_FAILURE;
        }
        gpuErrchk(cudaDeviceSynchronize());

        // Measure end time
        gettimeofday(&end, NULL);
        long seconds = (end.tv_sec - start.tv_sec);
        long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

        printf("FFT size %d - Execution time: %ld microseconds\n", n, micros);

        cufftDestroy(plan);
        cudaFree(data);
    }

    return 0;
}