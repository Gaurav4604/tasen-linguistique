import webrtcvad
import pyaudio
import numpy as np


vad = webrtcvad.Vad()

vad.set_mode(3)

# Initialize PyAudio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 320  # Number of audio samples per frame

audio = pyaudio.PyAudio()
mic_stream = audio.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)

try:
    while True:
        is_speech = vad.is_speech(mic_stream.read(CHUNK), RATE)
        if is_speech:
            print("Voice detected!")
        else:
            print("No voice detected.")


except KeyboardInterrupt:
    print("cancelled task")
finally:
    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()
