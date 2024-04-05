import numpy as np

# FFT sizes (repeated for reference, identical for both baseline and OpenMP results)
from matplotlib import pyplot as plt

fft_sizes = np.array([
    256, 512, 1024, 2048, 4096, 8192, 16384,
    32768, 65536, 131072, 262144, 524288,
    1048576, 2097152, 4194304, 8388608
])

fft_sizes = np.array([
    256, 512, 1024, 2048, 4096, 8192, 16384,
    32768, 65536, 131072, 262144, 524288,
    1048576, 2097152, 4194304, 8388608
])

# Baseline execution times in microseconds (provided by user)
baseline_execution_times_microseconds = np.array([
    61, 83, 181, 385, 847, 1826, 3895,
    8343, 17808, 26094, 59532, 155166,
    311738, 655184, 1495509, 3067821
])

# OpenMP execution times in microseconds (provided by user)
openmp_execution_times_microseconds = np.array([
    344, 56, 84, 155, 299, 598, 1233,
    2554, 5508, 17572, 27691, 64962,
    188330, 294188, 571403, 1187229
])

# Convert execution times to milliseconds for better readability.
baseline_execution_times_milliseconds = baseline_execution_times_microseconds / 1000.0
openmp_execution_times_milliseconds = openmp_execution_times_microseconds / 1000.0

# Create the plot to compare baseline and OpenMP performances.
plt.figure(figsize=(12, 7))
plt.plot(fft_sizes, baseline_execution_times_milliseconds, marker='o', linestyle='-', label='Baseline')
plt.plot(fft_sizes, openmp_execution_times_milliseconds, marker='x', linestyle='--', label='OpenMP')
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('FFT Size')
plt.ylabel('Execution Time (ms)')
plt.title('FFT Execution Time Comparison: Baseline vs OpenMP')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.show()