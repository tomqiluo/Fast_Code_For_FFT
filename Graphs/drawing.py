import numpy as np
import matplotlib.pyplot as plt

# FFT sizes and corresponding execution times in microseconds.
fft_sizes = np.array([
    256, 512, 1024, 2048, 4096, 8192, 16384,
    32768, 65536, 131072, 262144, 524288,
    1048576, 2097152, 4194304, 8388608
])
execution_times_microseconds = np.array([
    46, 57, 123, 253, 473, 1018, 2073,
    4475, 9520, 20624, 42795, 120053,
    217277, 524856, 1191415, 2295043
])

# Convert execution times to milliseconds for better readability.
execution_times_milliseconds = execution_times_microseconds / 1000.0

# Create the plot.
plt.figure(figsize=(10, 6))
plt.plot(fft_sizes, execution_times_milliseconds, marker='o', linestyle='-')
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('FFT Size')
plt.ylabel('Execution Time (ms)')
plt.title('FFT Execution Time vs FFT Size')
plt.grid(True, which='both', linestyle='--')
plt.show()