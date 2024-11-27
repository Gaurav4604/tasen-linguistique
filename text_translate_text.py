import ollama

def translate_text(model_name, target_language, text_to_translate):
    """
    Translate the provided text into the target language using the specified LLaMA model.

    Args:
        model_name: The name of the LLaMA model to use (e.g., "llama3.2").
        target_language: The target language for translation.
        text_to_translate: The text to translate.

    Returns:
        Translated text as a string.
    """
    # Define the system prompt
    system_prompt = (
        f"You are a translation expert. Your task is to translate text into {target_language}. "
        "Your responses should only contain the translated text and nothing else."
    )
    
    # Combine system prompt and user message into one array
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text_to_translate}
    ]
    
    # Send all messages together in one API call
    response = ollama.chat(model=model_name, messages=messages)
    return response.get("message").get("content").strip()

# Example usage:
if __name__ == "__main__":
    model = "llama3.2"          # Specify the model
    target_language = "Japanese"  # Specify the target language
    text = "I'm gonna talk a little bit in English. You just record for 5-10 seconds. I'm gonna tell you a little bit about what's around me right now. There's a bunch of bottles, there's my mechanical keyboard, there's my headphones and also my wireless mouse."  # Text to translate
    
    translation = translate_text(model, target_language, text)
    print("Translation:", translation)
