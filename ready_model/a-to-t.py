from transformers import  AutoProcessor, AutoModelForSeq2SeqLM
import torch
import torchaudio

# Load model and processor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = AutoModelForSeq2SeqLM.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

# Load and preprocess audio
audio_path = "wav_maker/hello-how-are-you-145245.mp3"
waveform, sample_rate = torchaudio.load(audio_path)

# Resample to 16kHz if needed
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# Convert stereo to mono
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0)
else:
    waveform = waveform.squeeze()

# Tokenize with a prompt (use empty string or ASR prompt)
prompt = "transcribe audio: "
inputs = processor(text=prompt, audios=waveform.numpy(), return_tensors="pt")

# Generate transcription
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

# Decode the output
decoded = processor.batch_decode(outputs, skip_special_tokens=True)
print("Transcription:", decoded[0])
