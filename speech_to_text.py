import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import pyaudio
import numpy as np

def initialize_device():
    """Initialize the device and torch dtype."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return device, torch_dtype

def load_model_and_processor(model_id, device, torch_dtype):
    """Load the Whisper model and processor."""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def record_audio(rate=16000, channels=1, chunk=1024, max_duration=30):
    """
    Record audio from the microphone.
    
    Args:
        rate: Sampling rate (default 16kHz).
        channels: Number of audio channels (default 1 for mono).
        chunk: Size of audio buffer (default 1024).
        max_duration: Maximum duration of recording in seconds (default 30).
    
    Returns:
        A NumPy array containing the recorded audio data.
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=channels,
                        rate=rate, input=True, frames_per_buffer=chunk)

    print(f"Recording up to {max_duration} seconds. Press Ctrl+C to stop early.")
    frames = []

    try:
        for _ in range(0, int(rate / chunk * max_duration)):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
    except KeyboardInterrupt:
        print("Recording stopped early.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    # Convert frames to NumPy array
    audio_data = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    return audio_data

def transcribe_audio(audio_data, processor, model, device, torch_dtype, sampling_rate=16000):
    """
    Transcribe audio data into text.
    
    Args:
        audio_data: NumPy array of audio data.
        processor: Whisper processor.
        model: Whisper model.
        device: Torch device.
        torch_dtype: Torch dtype.
        sampling_rate: Sampling rate of the audio (default 16kHz).
    
    Returns:
        Transcription string.
    """
    # Prepare input features
    inputs = processor(
        audio_data, sampling_rate=sampling_rate, return_tensors="pt"
    ).to(device, torch_dtype)

    # Generate transcription
    result = model.generate(inputs["input_features"])
    transcription = processor.batch_decode(result, skip_special_tokens=True)[0]
    return transcription

def speech_to_text():
    """Main function for speech-to-text."""
    # Initialize device and model
    device, torch_dtype = initialize_device()
    model_id = "openai/whisper-large-v3"
    model, processor = load_model_and_processor(model_id, device, torch_dtype)

    # Record audio
    audio_data = record_audio()

    # Transcribe audio
    transcription = transcribe_audio(audio_data, processor, model, device, torch_dtype)
    
    # Print transcription
    print("Transcription:")
    print(transcription)
    return transcription

# Run the speech-to-text pipeline
if __name__ == "__main__":
    speech_to_text()
