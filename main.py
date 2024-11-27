from speech_to_text import speech_to_text
from text_translate_text import translate_text
from text_to_speech import text_to_speech_fixed

def main():
    """
    Orchestrates the speech-to-text, text-translation, and text-to-speech pipeline.
    """
    # Step 1: Ask the user for the translation language
    target_language = input("Enter the target language for translation (e.g., Japanese): ").strip()

    # Step 2: Record and transcribe speech
    print("Recording audio... Please speak.")
    input_text = speech_to_text()
    print(f"Transcribed Text: {input_text}")

    # Step 3: Translate the transcribed text
    print(f"Translating text to {target_language}...")
    translated_text = translate_text(
        model_name="llama3.2",
        target_language=target_language,
        text_to_translate=input_text,
    )
    print(f"Translated Text: {translated_text}")

    # Step 4: Convert the translated text into speech
    print("Converting the translated text to speech...")
    text_to_speech_fixed(translated_text)
    print("Process complete! The translated speech has been generated and played.")

if __name__ == "__main__":
    main()
