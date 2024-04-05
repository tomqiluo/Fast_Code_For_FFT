#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <complex.h>

// Assume ComplexData is defined as:
typedef double complex ComplexData;

// Your FFT function declaration
void fft(ComplexData *a, int n, bool invert);

// Function to generate complex data
void generateComplexData(ComplexData *a, int n);

int main() {
    int sizes[] = {256, 512, 1024, /* Add all other sizes here */ 8388608};
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

void generateComplexData(ComplexData *a, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = (rand() / (ComplexData)RAND_MAX) + 
               (rand() / (ComplexData)RAND_MAX) * I;
    }
}