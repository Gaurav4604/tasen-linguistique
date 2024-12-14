import pyaudio
import numpy as np
from openwakeword.model import Model

# Initialize PyAudio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 320  # Number of audio samples per frame

audio = pyaudio.PyAudio()
mic_stream = audio.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)

# Initialize ONNX Runtime with openWakeWord
oww_model = Model(inference_framework="onnx", wakeword_models=["hey jarvis"])

print("Listening for 'hey jarvis'...")

try:
    while True:
        # Read audio data from the microphone
        audio_data = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Predict using the openWakeWord model with ONNX Runtime
        prediction = oww_model.predict(audio_data)

        # Check if the wake word was detected
        if prediction.get("hey jarvis", 0) > 0.5:
            print("Wake word 'hey jarvis' detected!")
            break
except KeyboardInterrupt:
    print("Stopped listening.")
finally:
    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()
