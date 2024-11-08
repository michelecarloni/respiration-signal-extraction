import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

# Carica i dati ECG
ecg_data = np.load("polar_ecg_data.npy")
ecg_time = np.load("polar_ecg_time.npy")

# Calcola la differenza tra l'ultimo e il primo timestamp
time_diff = float((ecg_time[-1] - ecg_time[0]) / 1e9)
print("Time: ", time_diff)

start_time = 5  # seconds
start_timestamp = ecg_time[0] + start_time * 1e9
# Find the index corresponding to 5 seconds
start_index = np.argmax(ecg_time >= start_timestamp)

ecg_time=ecg_time[start_index:]
ecg_data=ecg_data[start_index:]

fs = float(len(ecg_data) / (time_diff-start_time))
print("Frequency: ", fs)

# Calcola asse dei tempi uniforme
time_uniform = np.linspace(start_time, time_diff, len(ecg_time))

# Trova i picchi R nell'ECG
rpeaks, info = nk.ecg_peaks(ecg_data, sampling_rate=130)

# Calcola la frequenza cardiaca
ecg_rate = nk.ecg_rate(rpeaks, sampling_rate=130, desired_length=len(ecg_data))

# Calcola il segnale respiratorio EDR
edr = nk.ecg_rsp(ecg_rate, sampling_rate=130) #cighe Van get et al || chri method="charlton2016" || strona method="soni2019"

# Plot del segnale EDR
plt.figure(figsize=(12, 6))
plt.plot(time_uniform, edr)
plt.title('Edwards Respiratory Signal (EDR)')
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')
plt.grid(True)


edr_detrended = signal.detrend(edr)
# Calcola la Trasformata di Fourier del segnale EDR
fft_result = np.fft.fft(edr_detrended)
frequencies = np.fft.fftfreq(len(edr_detrended), d=1/fs)
# Take the absolute value of the FFT result to get the magnitude
fft_magnitude = np.abs(fft_result)


freq_mask = (frequencies >= 0.1) & (frequencies <= 1)
selected_frequencies = frequencies[freq_mask]
selected_fft = fft_result[freq_mask]

# Trova l'indice del picco nello spettro di potenza
peak_index = np.argmax(np.abs(selected_fft))
# Trova la frequenza corrispondente al picco
peak_frequency = selected_frequencies[peak_index]
# Trova il valore del picco nello spettro di potenza
peak_value = np.abs(selected_fft[peak_index])

# Plot dello spettro di potenza nella gamma di frequenza desiderata
plt.figure(figsize=(12, 6))
plt.plot(selected_frequencies, np.abs(selected_fft))
plt.title('Power Spectrum of EDR Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.grid(True)
plt.text(peak_frequency, peak_value, f'Peak frequency: {peak_frequency:.3f} Hz', fontsize=10, verticalalignment='bottom')
plt.show()