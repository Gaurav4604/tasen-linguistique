import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, BitsAndBytesConfig
import pyaudio
import numpy as np


def initialize_device():
    """Initialize the device and torch dtype."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return device, torch_dtype


def load_model_and_processor(model_id, device, torch_dtype):
    """Load the Whisper model and processor."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
        quantization_config=bnb_config,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def record_audio(rate=16000, channels=1, chunk=1024, max_duration=4):
    """
    Record audio from the microphone.

    Args:
        rate: Sampling rate (default 16kHz).
        channels: Number of audio channels (default 1 for mono).
        chunk: Size of audio buffer (default 1024).
        max_duration: Maximum duration of recording in seconds (default 5).

    Returns:
        A NumPy array containing the recorded audio data.
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
    )

    print("Listening...")
    frames = []
    try:
        for _ in range(0, int(rate / chunk * max_duration)):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
    except KeyboardInterrupt:
        print("Stopped listening.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    # Convert frames to NumPy array
    audio_data = (
        np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    )
    return audio_data


def transcribe_audio(
    audio_data, processor, model, device, torch_dtype, sampling_rate=16000
):
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
    inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt").to(
        device, torch_dtype
    )

    # Generate transcription
    result = model.generate(inputs["input_features"])
    transcription = processor.batch_decode(result, skip_special_tokens=True)[0]
    return transcription


def perform_action(trigger_word):
    """Define actions to perform based on the detected trigger word."""
    if trigger_word == "lights on":
        print("Action: Turning the lights on.")
    elif trigger_word == "lights off":
        print("Action: Turning the lights off.")
    elif trigger_word == "play music":
        print("Action: Playing music.")
    else:
        print("No action defined for this trigger word.")


def trigger_word_detection():
    """Main function for trigger word detection."""
    # Initialize device and model
    device, torch_dtype = initialize_device()
    model_id = "openai/whisper-large-v3"
    model, processor = load_model_and_processor(model_id, device, torch_dtype)

    # List of trigger words
    trigger_words = ["lights on", "lights off", "play music"]

    while True:
        # Record short audio clip
        audio_data = record_audio()

        # Transcribe audio
        transcription = transcribe_audio(
            audio_data, processor, model, device, torch_dtype
        )
        print(f"Transcription: {transcription}")

        if "end" in trigger_words:
            return
        # Check for trigger words
        # for trigger_word in trigger_words:
        #     if trigger_word in transcription.lower():
        #         print(f"Trigger word detected: '{trigger_word}'")
        #         perform_action(trigger_word)
        #         return  # Exit after performing an action


if __name__ == "__main__":
    trigger_word_detection()
