import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Load data from files
depth_data = np.load('depth_data_yolo1.npy')    

depth_addome = np.load('depth_data_yolo2.npy')

timestamps = np.load('timestamps_realsense.npy')
print(f"depth_data: {len(depth_data)}")
print(f"depth_addome: {len(depth_addome)}")

i=0
for data in depth_data:
    if data == 0:
        depth_data[i]=depth_data[i-1]
        
    i+=1
    
j=0
for data in depth_addome:
    if data == 0:
        depth_addome[j]=depth_addome[j-1]
        
    j+=1

for i in range(0, len(depth_data)):
    
    print(f"{depth_addome[i]}")

start_time = 5  # per scartare i primi 5 secondi
start_index = np.argmax(timestamps >= start_time)
print(f"start_index: {start_index}")
timestamps=timestamps[start_index:]
depth_data=depth_data[start_index:]
depth_addome=depth_addome[start_index:]

plt.figure(figsize=(12, 6))
plt.plot(timestamps, depth_data)
plt.xlabel('Time (sec)')
plt.ylabel('Depth (cm)')
plt.title('Depth Chest over Time')
plt.grid(True)

depth_data = depth_data-depth_addome

# i = 0
# for data in depth_data:
#     if data >=6:
#         depth_data[i] = depth_data[i-1]
    
#     i += 1
        
print(f"len temstamps: {len(timestamps)}")
# Plot
plt.figure(figsize=(12, 6))
plt.plot(timestamps, depth_data)
plt.xlabel('Time (sec)')
plt.ylabel('Depth (cm)')
plt.title('Depth Difference (Chest - Abdomen) over Time')
plt.grid(True)

plt.figure(figsize=(12, 6))
plt.plot(timestamps, depth_addome)
plt.xlabel('Time (sec)')
plt.ylabel('Depth (cm)')
plt.title('Depth Abdomen over Time')
plt.grid(True)

#filtrato
print(f"lunghezza {len(depth_data)}")

time_diff = float((timestamps[-1] - timestamps[0]))
print("Time: ", time_diff)

fs = float(len(depth_data) / time_diff)
print("Frequency: ", fs)

lowcut = 0.1  # Low cutoff frequency (Hz)
highcut = 0.8  # High cutoff frequency (Hz)
nyquist_frequency = 0.5 * fs
low = lowcut / nyquist_frequency
high = highcut / nyquist_frequency

order = 4

b, a = signal.butter(order, [low, high], btype='band')

depth_data_filtered = signal.filtfilt(b, a, depth_data)

depth_data_filtered_with_trend = depth_data_filtered + np.mean(depth_data)

plt.figure(figsize=(12, 6))
plt.plot(timestamps, depth_data_filtered_with_trend, color='red')
plt.title('Filtered Depth Difference (Chest - Abdomen) over time')
plt.xlabel('Time (sec)')
plt.ylabel('Depth (cm)')
plt.grid(True)

fft_result = np.fft.fft(depth_data)
frequencies = np.fft.fftfreq(len(depth_data), d=1/fs)
# Take the absolute value of the FFT result to get the magnitude
fft_magnitude = np.abs(fft_result)

freq_mask = (frequencies >= 0.1) & (frequencies <= 0.8)
selected_frequencies = frequencies[freq_mask]
selected_fft = fft_result[freq_mask]

# Trova l'indice del picco nello spettro di potenza
peak_index = np.argmax(np.abs(selected_fft))
# Trova la frequenza corrispondente al picco
peak_frequency = selected_frequencies[peak_index]
# Trova il valore del picco nello spettro di potenza
peak_value = np.abs(selected_fft[peak_index])

plt.figure(figsize=(12, 6))
plt.plot(selected_frequencies, np.abs(selected_fft), label='Asse Z')
plt.title('Depth Difference (Chest - Abdomen) Frequency Spectrum (0.1 Hz - 1 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.text(peak_frequency, peak_value, f'Peak frequency: {peak_frequency:.3f} Hz', fontsize=10, verticalalignment='bottom')

plt.show()