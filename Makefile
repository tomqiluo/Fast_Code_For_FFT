# Makefile for compiling baseline, OpenMP implementation, and CUDA implementation of FFT

# Compiler flags
BASELINE_FLAGS = -std=c99 -lm
OPENMP_FLAGS = -std=c99 -lm -Ofast -march=native -mtune=native -ffast-math -funroll-loops
CUDA_FLAGS = -arch=compute_75 -code=sm_75 -use_fast_math -O3

# Target executable names
TARGET_CUDA = fft_cuda
TARGET_BASELINE = fft.x
TARGET_OPENMP = fft_openmp.x

# Source files
SOURCE_CUDA = fft_cuda.cu
SOURCE_BASELINE = fft.c
SOURCE_OPENMP = fft_openmp.c

# Build rule for the baseline target
baseline: $(SOURCE_BASELINE)
	gcc $(SOURCE_BASELINE) -o $(TARGET_BASELINE) $(BASELINE_FLAGS)

# Build rule for the OpenMP-optimized target
openmp: $(SOURCE_OPENMP)
	gcc $(SOURCE_OPENMP) -o $(TARGET_OPENMP) $(OPENMP_FLAGS)

# Build rule for the CUDA target
cuda: $(SOURCE_CUDA)
	nvcc -o $(TARGET_CUDA) $(SOURCE_CUDA) $(CUDA_FLAGS)

# Clean rule to remove compiled files
clean:
	rm -f $(TARGET_CUDA) $(TARGET_BASELINE) $(TARGET_OPENMP)

# Default make rule to build all targets
all: baseline openmp cuda
