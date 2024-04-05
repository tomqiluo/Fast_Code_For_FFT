#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <complex.h>

// Assume ComplexData is defined as:
typedef double complex ComplexData;

// Your FFT function declaration
void fft(ComplexData *a, int n, bool invert);

int reverse(int num, int log2n);

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

// Helper function to perform bit reversal
int reverse(int num, int log2n) {
    int rev = 0;
    for (int i = 0; i < log2n; i++) {
        if (num & (1 << i))
            rev |= 1 << ((log2n - 1) - i);
    }
    return rev;
}

void fft(ComplexData *a, int n, bool invert) {
    int log2n = log2(n);
    
    // Bit reversal
    for (int i = 0; i < n; ++i) {
        int rev = reverse(i, log2n);
        if (i < rev)
            swap(a[i], a[rev]);
    }
    
    // Butterfly operations
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        ComplexData wlen = cos(ang) + sin(ang) * I;
        for (int i = 0; i < n; i += len) {
            ComplexData w = 1;
            for (int j = 0; j < len / 2; ++j) {
                ComplexData u = a[i+j], v = a[i+j+len/2] * w;
                a[i+j] = u + v;
                a[i+j+len/2] = u - v;
                w *= wlen;
            }
        }
    }

    // If we are inverting the transform, scale down the numbers
    if (invert) {
        for (int i = 0; i < n; i++)
            a[i] /= n;
    }
}