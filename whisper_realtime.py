import whisper
import pyaudio
import numpy as np


def record_audio(duration, sample_rate=16000):
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)

    print("Recording...")
    frames = []

    # Record audio for the specified duration
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(np.frombuffer(data, dtype=np.int16))

    print("Finished recording")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()


    audio_data = np.hstack(frames)

    return audio_data

# Load the Whisper model
model = whisper.load_model("medium")

# Record audio from the microphone
duration = 10  # duration in seconds
audio_data = record_audio(duration)

# Convert the audio data to float32 and resample to 16000 Hz
audio_data = audio_data.astype(np.float32) / 32768.0

# Prepare audio for Whisper (pad or trim to required length)
audio = whisper.pad_or_trim(audio_data)

# Transcribe the audio
result = model.transcribe(audio)

# Print the transcription
print("Transcription:", result["text"])
