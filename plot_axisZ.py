import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Load acceleration data from file
acc_data = np.load("polar_acc_data.npy")
acc_time = np.load("polar_acc_time.npy")

acc_z = acc_data[:, 2] * 0.00981

# Calculate the time difference between the last and first timestamp
time_diff = float((acc_time[-1] - acc_time[0]) / 1e9)
print("Time: ", time_diff)


start_time = 5  # seconds
start_timestamp = acc_time[0] + start_time * 1e9
# Find the index corresponding to 5 seconds
start_index = np.argmax(acc_time >= start_timestamp)

acc_time=acc_time[start_index:]
acc_z=acc_z[start_index:]

# Create a uniform time axis
time_uniform = np.linspace(start_time, time_diff,len(acc_time))

fs=float(len(acc_z)/(time_diff-start_time))

print("frequenza: ",fs)


plt.figure(figsize=(12, 6))
plt.plot(time_uniform, acc_z, label='Z-axis', color='green')
plt.title('Acceleration along the z-axis over time')
plt.xlabel('Time (sec)')
plt.ylabel('Acceleration (m/s^2)')
plt.grid(True)
plt.legend()

# Detrend the acceleration data
acc_z_detrended = signal.detrend(acc_z)

# Define the bandpass filter parameters
lowcut = 0.1  # Low cutoff frequency (Hz)
highcut = 0.4 # High cutoff frequency (Hz)
nyquist_freq = 0.5 * fs

# Order of the filter
order = 2

# Calculate the normalized cutoff frequencies
low = lowcut / nyquist_freq
high = highcut / nyquist_freq

# Design the Butterworth bandpass filter
b, a = signal.butter(order, [low, high], btype='band')

# Apply the bandpass filter using filtfilt to avoid phase shift
acc_z_filtered = signal.filtfilt(b, a, acc_z)

acc_z_filtered_with_trend = acc_z_filtered + np.mean(acc_z)


# Plot the detrended acceleration along the z-axis 
plt.figure(figsize=(12, 6))
plt.plot(time_uniform, acc_z_filtered_with_trend, label='Z-axis', color='green')
plt.title('Filtered acceleration along the z-axis over time')
plt.xlabel('Time (sec)')
plt.ylabel('Acceleration (m/s^2)')
plt.grid(True)
plt.legend()



# Compute the FFT
fft_result = np.fft.fft(acc_z_detrended)
frequencies = np.fft.fftfreq(len(acc_z_detrended), d=1/fs)
# Take the absolute value of the FFT result to get the magnitude
fft_magnitude = np.abs(fft_result)

freq_mask = (frequencies >= 0.1) & (frequencies <= 1)
selected_frequencies = frequencies[freq_mask]
selected_fft = fft_result[freq_mask]

peak_index = np.argmax(np.abs(selected_fft))
# Trova la frequenza corrispondente al picco
peak_frequency = selected_frequencies[peak_index]
# Trova il valore del picco nello spettro di potenza
peak_value = np.abs(selected_fft[peak_index])

plt.figure(figsize=(12, 6))
plt.plot(selected_frequencies, np.abs(selected_fft), label='Asse Z')
plt.title('Acceleration along the z-axis Frequency Spectrum (0.1 Hz - 1 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.text(peak_frequency, peak_value, f'Peak frequency: {peak_frequency:.3f} Hz', fontsize=10, verticalalignment='bottom')
plt.show()