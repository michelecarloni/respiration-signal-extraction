import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import biosppy.signals.ecg as ecg

# Load ECG data
ecg_data = np.load("polar_ecg_data.npy")
ecg_time = np.load("polar_ecg_time.npy")

# Calculate the time difference and sampling frequency
time_diff = float((ecg_time[-1] - ecg_time[0]) / 1e9)
fs = float(len(ecg_data) / time_diff)
print("Frequency: ", fs)
print("Time: ", time_diff)

# Set start time and find the index
start_time = 5  # seconds
start_timestamp = ecg_time[0] + start_time * 1e9
start_index = np.argmax(ecg_time >= start_timestamp)

# Slice the data starting from the specified index
ecg_time = ecg_time[start_index:]
ecg_data = ecg_data[start_index:]

# Recalculate the sampling frequency
fs = float(len(ecg_data) / (time_diff - start_time))
print("Frequency: ", fs)

# Create a uniform time array
time_uniform = np.linspace(start_time, time_diff, len(ecg_time))

# Process the ECG data to detect QRS complexes
out = ecg.ecg(signal=ecg_data, sampling_rate=fs, show=False)
qrs_inds = out['rpeaks']
num_qrs = len(qrs_inds)
print("Number of QRS complexes detected:", num_qrs)

# Calculate BPM
bpm = int((num_qrs / (time_diff - start_time)) * 60)
print("bpm: ", bpm)

# Define plot limits
ylim_low = -1500
ylim_high = 4000

# Clip ECG data to defined limits
ecg_data[ecg_data < ylim_low] = ylim_low
ecg_data[ecg_data > ylim_high] = ylim_high

# Define the selection function for the SpanSelector
def onselect(xmin, xmax):
    plt.ylim(ylim_low, ylim_high)
    plt.xlim(xmin, xmax)
    plt.draw()

# Define the callback for the backspace event to reset the view
def onback(event):
    if event.key == 'backspace':
        plt.xlim(0, time_diff + 2)
        plt.ylim(ylim_low, ylim_high)
        plt.draw()

# Plot the ECG data
plt.figure(figsize=(12, 6))
plt.plot(time_uniform, ecg_data, label='ECG')
plt.plot(time_uniform[qrs_inds], ecg_data[qrs_inds], 'ro', label='QRS Complex')

# Add BPM text to the plot
plt.text(0.05, 0.95, f'BPM: {bpm}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')

# Set plot limits
plt.ylim(ylim_low, ylim_high)

# Add plot title and labels
plt.title('ECG Data with QRS Complex Detection using BioSPPy')
plt.xlabel('Time (sec)')
plt.ylabel('Voltage (mV)')
plt.legend()
plt.grid(True)

# Create the SpanSelector for zoom functionality
span = SpanSelector(plt.gca(), onselect, 'horizontal', useblit=True)

# Connect the backspace event to reset the view
plt.gcf().canvas.mpl_connect('key_press_event', onback)

# Display the plot
plt.show()
