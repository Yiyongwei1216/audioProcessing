# Audio Processing

This script demonstrates how to record audio from a microphone using the PyAudio library in Python and save it as a WAV file. Here is a step-by-step description of what the code does:

### Importing Required Libraries:
- `pyaudio`: This library provides Python bindings for PortAudio, a cross-platform audio I/O library.
- `wave`: This library is used to read and write WAV files.

### Creating a PyAudio Instance:
```python
p = pyaudio.PyAudio()
```

### Setting Audio Parameters:
- `FORMAT`: Specifies the audio format (16-bit PCM in this case).
- `FS`: The sampling frequency (44100 Hz, which is CD quality).
- `CHANNELS`: The number of audio channels (1 for mono).
- `CHUNK`: The size of each audio chunk (1024 frames).
- `RECORD_SECOND`: Duration of the recording in seconds (3 seconds).

### Opening an Audio Stream:

```python
stream = p.open(format=FORMAT, channels=CHANNELS, rate=FS, input=True, frames_per_buffer=CHUNK)
```

### Recording Audio:
- A list `frames` is initialized to store the recorded audio data.
- `num_times` calculates the number of chunks to be recorded based on the duration and chunk size.
- A loop reads the audio data from the stream in chunks and appends it to the `frames` list.

### Starting and Closing the Stream:

```python
stream.start_stream()
stream.close()
p.terminate()
```

### Outputting Recorded Data:
- The `frames` list, which contains the recorded audio data, is printed.

### Writing Audio Data to a WAV File:
- `samplewidth` retrieves the sample size of the audio format.
- A new WAV file `output.wav` is created.
- The number of channels, sample width, and frame rate are set.
- The recorded frames are written to the WAV file.
- The WAV file is closed.

This snippet includes all necessary details about the parameters and the steps taken in the code.
