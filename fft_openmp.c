#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>


#define PI 3.14159265358979323846

typedef double complex ComplexData;

void fft_omp(ComplexData* restrict a, int n, bool invert) {
    int i, j, bit;
    int unroll_factor = 32;
    ComplexData w, temp, ws[unroll_factor];
    j = 0;

    #pragma omp parallel for schedule(static) private(i, j, bit, temp) shared(a, n) num_threads(omp_get_max_threads())
    for (i = 1; i < n; i++) {
        j = 0;
        bit = n >> 1;

        int temp_i = i;
        while (temp_i > 0) {
            if (temp_i & 1) {
                j |= bit;
            }
            temp_i >>= 1;
            bit >>= 1;
        }

        if (i < j) {
            temp = a[i];
            a[i] = a[j];
            a[j] = temp;
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? 1 : -1);
        ComplexData wlen = cos(ang) + sin(ang) * I;
        int half_len = len / 2;

        #pragma omp parallel for schedule(static) private(i, j, w, ws, temp) shared(a, n, len, half_len, wlen, invert) num_threads(omp_get_max_threads())
        for (i = 0; i < n; i += len) {
            w = 1;
            
            int steps = half_len / unroll_factor;

            for (j = 0; j < steps; ++j) {
                ComplexData ws[unroll_factor];
                for (int k = 0; k < unroll_factor; ++k) {
                    ws[k] = w;
                    w *= wlen;
                }

                for (int k = 0; k < unroll_factor; ++k) {
                    int idx = i + j * unroll_factor + k;
                    ComplexData u = a[idx];
                    ComplexData v = a[idx + half_len] * ws[k];
                    a[idx] = u + v;
                    a[idx + half_len] = u - v;
                }
            }

            for (j = steps * unroll_factor; j < half_len; ++j) {
                ComplexData u = a[i + j];
                ComplexData v = a[i + j + half_len] * w;
                a[i + j] = u + v;
                a[i + j + half_len] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        #pragma omp parallel for schedule(static) shared(a, n)
        for (int i = 0; i < n; i++) {
            a[i] /= n;
        }
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
        fft_omp(data, n, false);
        
        // Measure end time
        gettimeofday(&end, NULL);
        long seconds = (end.tv_sec - start.tv_sec);
        long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

        printf("FFT size %d - Execution time: %ld microseconds\n", n, micros);

        free(data);
    }

    return 0;
}