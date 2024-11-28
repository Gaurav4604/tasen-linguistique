import outetts

# Constants
MODEL_PATH = "OuteAI/OuteTTS-0.2-500M"  # Fixed model path
LANGUAGE = "ja"                         # Target language
SPEAKER_NAME = "male_1"                 # Fixed default speaker
OUTPUT_PATH = "output.wav"              # Fixed output file path

def text_to_speech_fixed(text, temperature=0.4, repetition_penalty=1):
    """
    Generate speech from text using the OuteTTS library with fixed configuration.

    Args:
        text (str): The text to convert to speech.
        temperature (float, optional): Controls stability (default: 0.1).
        repetition_penalty (float, optional): Controls tone consistency (default: 1.1).

    Returns:
        None: The function saves the audio file and plays it.
    """
    # Configure the model
    model_config = outetts.HFModelConfig_v1(
        model_path=MODEL_PATH,
        language=LANGUAGE,
    )

    # Initialize the interface
    interface = outetts.InterfaceHF(model_version="0.2", cfg=model_config)

    # Load the fixed speaker profile
    interface.print_default_speakers()  # Optional: List available default speakers
    speaker = interface.load_default_speaker(name=SPEAKER_NAME)

    # Generate speech
    output = interface.generate(
        text=text,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        speaker=speaker,
    )

    # Save the synthesized speech to the fixed output path
    output.save(OUTPUT_PATH)

    # Play the synthesized speech
    print(f"Generated speech saved to {OUTPUT_PATH}. Playing the audio...")
    output.play()

# Example usage
if __name__ == "__main__":
    text = "おもちゃのやり取り用のカップルがたくさん並んでいます。機械キーボード、ヘッドフォン、無線マウスが周囲に置かれています。"
    text_to_speech_fixed(text)
