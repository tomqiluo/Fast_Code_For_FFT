#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>

// Assume ComplexData is defined as:
typedef double complex ComplexData;

// Your FFT function declaration
void fft(ComplexData *a, int n, bool invert) {
    #pragma omp parallel for
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j) {
            ComplexData temp = a[i];
            a[i] = a[j];
            a[j] = temp;
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * M_PI / len * (invert ? -1 : 1);
        ComplexData wlen = cexp(I * ang);
        
        #pragma omp parallel for
        for (int i = 0; i < n; i += len) {
            ComplexData w = 1.0;
            for (int j = 0; j < len / 2; j++) {
                ComplexData u = a[i + j];
                ComplexData v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            a[i] /= n;
    }
}

// Function to generate complex data
void generateComplexData(ComplexData *a, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = (rand() / (ComplexData)RAND_MAX) + 
               (rand() / (ComplexData)RAND_MAX) * I;
    }
}

int main() {
    int sizes[] = {256, 512, 1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; ++i) {
        int n = sizes[i];
        ComplexData *data = (ComplexData *)malloc(n * sizeof(ComplexData));

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