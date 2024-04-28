import numpy as np
import matplotlib.pyplot as plt

# FFT sizes are the same for both scenarios
fft_sizes = np.array([
    256, 512, 1024, 2048, 4096, 8192, 16384,
    32768, 65536, 131072, 262144, 524288,
    1048576, 2097152, 4194304, 8388608
])

# Scenario 1 execution times in microseconds
execution_times_1 = np.array([
    46, 57, 116, 221, 478, 1674, 3572,
    7023, 9866, 25980, 49715, 111480,
    262506, 555289, 1192296, 2493408
])

# Scenario 2 execution times in microseconds
execution_times_2 = np.array([
    44, 57, 124, 282, 545, 1058, 2298,
    4477, 13757, 20871, 52792, 111460,
    262933, 591985, 1189559, 2486463
])

# Plot for Scenario 1
plt.figure(figsize=(10, 6))
plt.plot(fft_sizes, execution_times_1, marker='o', linestyle='-', color='blue', label='Scenario 1')
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('FFT Size')
plt.ylabel('Execution Time (microseconds)')
plt.title('FFT Execution Time - Scenario 1')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.show()

# Plot for Scenario 2
plt.figure(figsize=(10, 6))
plt.plot(fft_sizes, execution_times_2, marker='o', linestyle='-', color='orange', label='Scenario 2')
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('FFT Size')
plt.ylabel('Execution Time (microseconds)')
plt.title('FFT Execution Time - Scenario 2')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.show()