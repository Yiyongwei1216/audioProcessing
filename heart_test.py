import numpy as np
import matplotlib.pyplot as plt
import pyaudio
from scipy.signal import butter, sosfiltfilt, stft

# Define audio stream constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 2000  # Use 2000 Hz as specified for filtering
MAX_TIME_SECONDS = 4  # Display the last 10 seconds of data
BUFFER_SIZE = RATE * MAX_TIME_SECONDS  # Total samples to maintain in buffer

# Initialize PyAudio object
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Initialize the plotting
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Setup time-domain plot
time_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)  # initialize buffer
line1, = ax1.plot(np.arange(BUFFER_SIZE), time_buffer, 'b-')
ax1.set_xlim(0, BUFFER_SIZE)  # Set X-axis limit to BUFFER_SIZE
ax1.set_xticks(np.arange(0, BUFFER_SIZE+1, RATE))  # Set x-axis ticks to match the sampling rate
ax1.set_xticklabels(np.arange(0, MAX_TIME_SECONDS+1))  # Convert to seconds
ax1.set_ylim(-0.3, 0.3)  # Adjust the y-axis range to match the images
ax1.set_title('Real-time Time Domain Waveform')
ax1.set_ylabel('Amplitude')

# Setup spectrogram plot
freqs = np.fft.rfftfreq(CHUNK, 1.0 / RATE)
idx_max_freq = np.where(freqs <= 400)[0][-1]  # index of max frequency 400 Hz
im = ax2.imshow(np.zeros((idx_max_freq + 1, int(RATE / CHUNK * MAX_TIME_SECONDS))), aspect='auto', origin='lower',
                extent=[0, MAX_TIME_SECONDS, 0, freqs[idx_max_freq]], vmin=-80, vmax=0, cmap='magma')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Frequency (Hz)')
plt.colorbar(im, ax=ax2, format='%+2.0f dB')

# Define the bandpass filter
def butterworth_bandpass(lowcut=25, highcut=400, fs=2000, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

sos = butterworth_bandpass()

def update_plots():
    raw_data = stream.read(CHUNK, exception_on_overflow=False)
    data = np.frombuffer(raw_data, dtype=np.int16)
    data = data.astype(np.float32) / np.power(2.0, 15)  # Normalize to [-1, 1] range
    
    # Apply bandpass filter
    data = sosfiltfilt(sos, data)
    
    # Update time domain buffer
    time_buffer[:-CHUNK] = time_buffer[CHUNK:]  # Shift buffer
    time_buffer[-CHUNK:] = data  # Append new data
    line1.set_ydata(time_buffer)  # Update plot

    # Compute STFT for the spectrogram
    f, t, Zxx = stft(time_buffer, fs=RATE, nperseg=256)
    Zxx_db = 20 * np.log10(np.abs(Zxx[:idx_max_freq + 1]) + 1e-6)
    im.set_data(Zxx_db)
    im.set_extent([0, MAX_TIME_SECONDS, 0, freqs[idx_max_freq]])
    fig.canvas.draw()
    fig.canvas.flush_events()

try:
    while True:
        update_plots()
except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    plt.close(fig)
    stream.stop_stream()
    stream.close()
    p.terminate()
