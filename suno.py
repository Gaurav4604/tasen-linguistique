from transformers import AutoProcessor, AutoModel
import torch
import scipy.io.wavfile

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor and model
processor = AutoProcessor.from_pretrained("suno/bark-small")
# processor = processor.to(device)
model = AutoModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)





# Voice preset configuration
voice_preset = "v2/en_speaker_6"

# Prepare input and move tensors to the correct device
inputs = processor("Hello, my dog is cute", voice_preset=voice_preset, return_tensors="pt")
# Generate speech with sampling
speech_values = model.generate(**inputs, do_sample=True)


# Ensure output is on CPU for saving
speech_values = speech_values.cpu().numpy().squeeze()


# Save audio output
scipy.io.wavfile.write("bark_out.wav", data=speech_values)