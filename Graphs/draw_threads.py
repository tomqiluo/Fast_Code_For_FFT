import numpy as np

# Data from the user's FFT OpenMP outputs
from matplotlib import pyplot as plt

fft_sizes = np.array([
    256, 512, 1024, 2048, 4096, 8192, 16384,
    32768, 65536, 131072, 262144, 524288,
    1048576, 2097152, 4194304, 8388608
])

fft_data = {
    1: [102, 90, 184, 430, 882, 1872, 3973, 8487, 18492, 26745, 59910, 146193, 319651, 702303, 1381777, 2980136],
    2: [1236, 66, 131, 273, 581, 1236, 2626, 5611, 12162, 23197, 52730, 118078, 248083, 574309, 963652, 2100788],
    4: [264, 58, 113, 200, 422, 814, 1700, 3624, 7872, 17122, 35021, 81175, 164598, 348147, 737872, 1527816],
    8: [344, 56, 84, 155, 299, 598, 1233, 2554, 5508, 17572, 27691, 64962, 188330, 294188, 571403, 1187229],
    16: [61409, 67150, 73229, 79291, 87008, 98300, 94590, 105296, 110440, 120099, 139346, 171681, 264245, 412034, 658371, 1192319],
    32: [122930, 137345, 141176, 164122, 175619, 181343, 198154, 210379, 212561, 218060, 255409, 299461, 356549, 466796, 710123, 1288535]
}

# Convert all execution times to milliseconds
for threads in fft_data:
    fft_data[threads] = np.array(fft_data[threads]) / 1000.0

# Plotting
plt.figure(figsize=(12, 8))

# Plot for each OMP_NUM_THREADS value
for threads, times in fft_data.items():
    plt.plot(fft_sizes, times, marker='o', linestyle='-', label=f'Threads={threads}')

plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('FFT Size')
plt.ylabel('Execution Time (ms)')
plt.title('FFT Execution Time vs FFT Size for Different Thread Counts')
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.show()