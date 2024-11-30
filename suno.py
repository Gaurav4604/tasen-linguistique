from transformers import AutoProcessor, BarkModel
import torch
import scipy.io.wavfile
import numpy as np

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor and model
processor = AutoProcessor.from_pretrained("suno/bark-small")
# processor = processor.to(device)
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to_bettertransformer().to(device)

voice_preset = "v2/en_speaker_6"

# Prepare input and move tensors to the correct device
inputs = processor("[music]Hey, this is Gaurav", voice_preset=voice_preset, return_tensors="pt")
for key, value in inputs.items():
    inputs[key] = value.to(device)  # Ensure inputs are on the same device as the model

# Generate speech with sampling
speech_values = model.generate(**inputs, do_sample=True)

# Ensure output is on CPU for saving
speech_values = speech_values.cpu().numpy().squeeze()
speech_values = (speech_values * 32767).astype(np.int16)


# Save audio output
scipy.io.wavfile.write("bark_out.wav", rate=24000, data=speech_values)