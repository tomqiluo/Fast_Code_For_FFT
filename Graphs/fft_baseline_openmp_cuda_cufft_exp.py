import matplotlib.pyplot as plt

# Data from the user's FFT execution results
fft_sizes = [
    256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
    262144, 524288, 1048576, 2097152, 4194304, 8388608
]

# Standard FFT execution times in microseconds
standard_times = [
    45, 57, 123, 247, 476, 1013, 2178, 4686, 9653, 20019,
    42428, 91813, 208285, 480098, 1032068, 2142580
]

# OpenMP Accelerated FFT execution times in microseconds
openmp_times = [
    33, 17, 34, 76, 161, 303, 647, 1414, 3120, 6913,
    14825, 34033, 98781, 228900, 496092, 1057972
]

# CUDA Accelerated FFT execution times in microseconds
cuda_times = [
    228, 140, 145, 151, 162, 202, 288, 445, 768, 1122,
    1938, 3267, 5866, 12058, 24231, 48585
]

# CUDA 1 Accelerated FFT execution times in microseconds
cuda_1_times = [
    47, 34, 56, 103, 211, 111, 121, 123, 127, 352,
    538, 1118, 2249, 2521, 4875, 9382
]

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(fft_sizes, standard_times, label='Baseline FFT', marker='o')
plt.plot(fft_sizes, openmp_times, label='OpenMP FFT', marker='o')
plt.plot(fft_sizes, cuda_times, label='CUDA FFT', marker='o')
plt.plot(fft_sizes, cuda_1_times, label='cufft FFT', marker='o')

plt.xlabel('FFT Size')
plt.ylabel('Execution Time (microseconds)')
plt.title('FFT Execution Time Comparison')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()