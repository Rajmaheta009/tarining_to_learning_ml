import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import librosa
import numpy as np
from gtts import gTTS
import os

# --- Configuration ---
AUDIO_FILE_FOR_INFERENCE = "wav_maker/hello-how-ghkgare-you-145245.mp3"
ASR_MODEL_PATH = "trained_models/asr_model.pth"
TEXT_GEN_MODEL_PATH = "trained_models/text_gen_model.pth"
RUN_TRAINING = True # Set to False to skip training and only run inference (if models are saved)

# Create directories if they don't exist
os.makedirs("wav_maker", exist_ok=True)
os.makedirs("trained_models", exist_ok=True)

# Step 1: Generate a sample audio file for inference if it doesn't exist
if not os.path.exists(AUDIO_FILE_FOR_INFERENCE):
    try:
        tts = gTTS("Hi how are you") # Simple phrase for inference
        tts.save(AUDIO_FILE_FOR_INFERENCE)
        print(f"Generated sample audio: {AUDIO_FILE_FOR_INFERENCE}")
    except Exception as e:
        print(f"Could not create gTTS audio (internet connection needed?): {e}")
        # As a fallback, we'll need this file for extract_mfcc later
        # If gTTS fails, the script might error out in run_full_model if this file isn't present

# Step 2: Define character set and mappings
# Extended a bit for more general text, ensure dummy data uses these
chars = " ABCDEFGHIJKLMNOPQRSTUVWXYZHI!'.?" # Added space and more common chars
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in char2idx.items()}
vocab_size = len(chars)
BLANK_IDX = char2idx.get(' ', vocab_size -1) # A blank token for CTC, often the last index or a dedicated one.

# Step 3: Utility function to convert indices to text
def indices_to_text(indices):
    # For CTC output, often need to decode (remove blanks and repeats)
    decoded_indices = []
    last_char_idx = -1
    for idx in indices:
        if idx == BLANK_IDX:
            last_char_idx = -1 # Reset so next char isn't seen as repeat of blank
            continue
        if idx == last_char_idx:
            continue
        decoded_indices.append(idx)
        last_char_idx = idx
    return ''.join([idx2char.get(i, '') for i in decoded_indices])

def text_to_indices(text, char2idx_map):
    return torch.tensor([char2idx_map.get(c.upper(), char2idx_map['?']) for c in text if c.upper() in char2idx_map])


# Step 4: Simple ASR model (audio to text indices)
class SimpleASRModel(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64, output_dim=vocab_size): # Increased hidden_dim
        super().__init__()
        # Using more layers can help
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # hidden_dim * 2 for bidirectional

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # out shape: (batch, seq_len, hidden_dim * 2)
        out = self.fc(out)
        # out shape: (batch, seq_len, output_dim)
        return F.log_softmax(out, dim=2) # CTCLoss expects log_softmax probabilities

# Step 5: Simple Text Generation Model (text indices to response)
class SimpleTextGenModel(nn.Module):
    def __init__(self, vocab_size=vocab_size, embed_dim=32, hidden_dim=64): # Increased dimensions
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=char2idx.get(' ', 0)) # Use space as padding
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len)
        x = self.embedding(x)
        # x shape: (batch, seq_len, embed_dim)
        out, hidden = self.lstm(x, hidden)
        # out shape: (batch, seq_len, hidden_dim)
        out = self.fc(out)
        # out shape: (batch, seq_len, vocab_size)
        return out, hidden # No softmax here if using CrossEntropyLoss

# Step 6: Load audio and extract MFCCs
def extract_mfcc(audio_path, n_mfcc=13, max_len=100): # Increased max_len for MFCCs
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T  # (time, features)
        if mfcc.shape[0] == 0: # Handle empty audio
            mfcc = np.zeros((max_len, n_mfcc))
        if mfcc.shape[0] < max_len:
            pad_width = max_len - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:max_len]
        return torch.tensor(mfcc).unsqueeze(0).float()  # shape: (1, max_len, n_mfcc)
    except Exception as e:
        print(f"Error extracting MFCC from {audio_path}: {e}")
        # Return a zero tensor as a fallback
        return torch.zeros((1, max_len, n_mfcc)).float()

# --- Dummy Data Generation for Training ---
def get_dummy_asr_data(num_samples=10, max_seq_len=100, feature_dim=13, vocab_map=char2idx):
    X = []
    Y_text = []
    for i in range(num_samples):
        # Simulate MFCC features
        seq_len = np.random.randint(max_seq_len // 2, max_seq_len)
        mfccs = torch.randn(1, seq_len, feature_dim)
        # Pad MFCCs to max_seq_len
        if seq_len < max_seq_len:
            mfccs = F.pad(mfccs, (0,0,0,max_seq_len-seq_len), "constant", 0)
        X.append(mfccs)

        # Create simple dummy text
        if i % 3 == 0: text = "HELLO WORLD"
        elif i % 3 == 1: text = "HI THERE"
        else: text = "TESTING"
        Y_text.append(text)

    # Convert text to indices for targets
    Y_indices = [text_to_indices(t, vocab_map) for t in Y_text]
    return X, Y_indices, Y_text # Return original text for clarity too

def get_dummy_text_gen_data(num_samples=10, max_len=15, vocab_map=char2idx):
    input_texts = []
    target_texts = []
    for i in range(num_samples):
        if i % 3 == 0:
            inp = "HOW ARE YOU"
            tgt = "I AM FINE"
        elif i % 3 == 1:
            inp = "HELLO"
            tgt = "HI THERE"
        else:
            inp = "GOOD DAY"
            tgt = "YOU TOO"
        input_texts.append(inp)
        target_texts.append(tgt)

    # Convert to indices and pad
    # For text generation, input and target might be shifted or structured for prediction
    # Here, we'll just use them as pairs for simplicity
    X_indices = []
    Y_indices = []
    padding_value = vocab_map.get(' ', 0) # Pad with space index

    for inp_text, tgt_text in zip(input_texts, target_texts):
        inp_idx = [vocab_map.get(c, vocab_map['?']) for c in inp_text]
        tgt_idx = [vocab_map.get(c, vocab_map['?']) for c in tgt_text]

        # Pad
        inp_idx_padded = inp_idx + [padding_value] * (max_len - len(inp_idx))
        tgt_idx_padded = tgt_idx + [padding_value] * (max_len - len(tgt_idx))

        X_indices.append(torch.tensor(inp_idx_padded[:max_len]))
        Y_indices.append(torch.tensor(tgt_idx_padded[:max_len]))

    return torch.stack(X_indices), torch.stack(Y_indices)


# --- Training Functions ---
def train_asr_model(model, data_X, data_Y_indices, epochs=10, lr=0.001):
    print("\n--- Training ASR Model (with dummy data) ---")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # CTCLoss expects: log_probs (T, N, C), targets (N, S) or (sum(target_lengths)),
    # input_lengths (N), target_lengths (N)
    # T: input sequence length, N: batch size, C: num classes, S: max target length
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, reduction='mean', zero_infinity=True)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(data_X)): # Batch size of 1 for simplicity here
            mfccs = data_X[i] # Shape: (1, seq_len, features)
            target_indices = data_Y_indices[i] # Shape: (target_len)

            optimizer.zero_grad()
            # Model output: (N, T, C) -> permute for CTCLoss (T, N, C)
            log_probs = model(mfccs).permute(1, 0, 2) # (seq_len, 1, vocab_size)

            input_lengths = torch.full(size=(1,), fill_value=log_probs.size(0), dtype=torch.long)
            target_lengths = torch.tensor([len(target_indices)], dtype=torch.long)

            if target_lengths.item() == 0 or input_lengths.item() < target_lengths.item() : # Skip if target is empty or too long for input
                # print(f"Skipping sample {i}: target_len={target_lengths.item()}, input_len={input_lengths.item()}")
                continue

            loss = ctc_loss(log_probs, target_indices.unsqueeze(0), input_lengths, target_lengths)

            if torch.isinf(loss) or torch.isnan(loss):
                print(f"Warning: inf or nan loss encountered at epoch {epoch+1}, sample {i}. Skipping update.")
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / len(data_X) if len(data_X) > 0 else 0
        print(f"ASR Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), ASR_MODEL_PATH)
    print(f"ASR Model saved to {ASR_MODEL_PATH}")

def train_text_gen_model(model, data_X_indices, data_Y_indices, epochs=10, lr=0.001):
    print("\n--- Training Text Generation Model (with dummy data) ---")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # CrossEntropyLoss expects (N, C, ...) and targets (N, ...)
    # Our model outputs (N, seq_len, C), targets are (N, seq_len)
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx.get(' ', 0)) # Ignore padding
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        # data_X_indices shape: (num_samples, max_len)
        # data_Y_indices shape: (num_samples, max_len)
        # For simplicity, using batch size = num_samples
        # In practice, use DataLoader for batching

        optimizer.zero_grad()
        # For text generation, the input to the model at each step could be the previous true token (teacher forcing)
        # For this simple RNN, we'll feed the whole input sequence once.
        # The target is to predict the data_Y_indices sequence given data_X_indices
        # This is more like a sequence-to-sequence task if X and Y are different.
        # If Y is X shifted by one, it's language modeling.
        # Here, X and Y are pairs, e.g., Q&A.

        # output shape: (batch_size, seq_len, vocab_size)
        output_logits, _ = model(data_X_indices)

        # Reshape for CrossEntropyLoss: (N * seq_len, vocab_size)
        # Targets: (N * seq_len)
        loss = criterion(output_logits.view(-1, vocab_size), data_Y_indices.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
        optimizer.step()
        total_loss += loss.item()

        print(f"TextGen Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), TEXT_GEN_MODEL_PATH)
    print(f"Text Generation Model saved to {TEXT_GEN_MODEL_PATH}")


# Step 7: Main function to run ASR and text generation (Modified to load weights)
def run_full_model(audio_path, asr_model_w_path, text_gen_model_w_path):
    # Load models
    asr_model = SimpleASRModel()
    text_gen_model = SimpleTextGenModel()

    # Load ASR model weights if available
    if os.path.exists(asr_model_w_path):
        try:
            asr_model.load_state_dict(torch.load(asr_model_w_path))
            print(f"Loaded ASR model weights from {asr_model_w_path}")
        except Exception as e:
            print(f"Could not load ASR model weights: {e}. Using fresh model.")
    else:
        print("No ASR model weights found. Using fresh model.")
    asr_model.eval()

    # Load Text Generation model weights if available
    if os.path.exists(text_gen_model_w_path):
        try:
            text_gen_model.load_state_dict(torch.load(text_gen_model_w_path))
            print(f"Loaded Text Gen model weights from {text_gen_model_w_path}")
        except Exception as e:
            print(f"Could not load Text Gen model weights: {e}. Using fresh model.")
    else:
        print("No Text Gen model weights found. Using fresh model.")
    text_gen_model.eval()


    # Process audio
    mfcc_input = extract_mfcc(audio_path)
    if mfcc_input.sum() == 0 and not os.path.exists(audio_path): # Check if fallback zero tensor was returned due to missing file
        print(f"Error: Cannot process audio as {audio_path} does not exist and MFCC extraction failed.")
        return "ERROR: AUDIO FILE MISSING", "ERROR: CANNOT GENERATE RESPONSE"


    # Part 1: ASR model - audio to text
    with torch.no_grad():
        asr_log_probs = asr_model(mfcc_input) # Output is (batch=1, seq_len, vocab_size)
    # For CTC, decoding is more complex than simple argmax.
    # A simple argmax approach (greedy decoding):
    predicted_indices_asr = torch.argmax(asr_log_probs, dim=2)[0].tolist()
    transcription = indices_to_text(predicted_indices_asr) # Use CTC decoding utility

    # Part 2: Text Generation model - generate response from transcription
    if transcription.strip() == "" or transcription == "ERROR: AUDIO FILE MISSING":
        generated_text = "Could not understand audio."
    else:
        # Prepare input for text generation model
        # The current text_gen_model expects a sequence and outputs a sequence.
        # For inference, we might want to generate token by token.
        # For simplicity, feed the whole transcription and take the output.
        # Max length for text gen input
        text_gen_input_max_len = 20
        input_seq_list = [char2idx.get(c.upper(), char2idx['?']) for c in transcription if c.upper() in char2idx]

        if not input_seq_list: # Handle empty transcription after filtering
             input_seq_list = [char2idx['?']]

        # Pad or truncate
        if len(input_seq_list) < text_gen_input_max_len:
            input_seq_list += [char2idx[' ']] * (text_gen_input_max_len - len(input_seq_list))
        else:
            input_seq_list = input_seq_list[:text_gen_input_max_len]

        input_indices_text_gen = torch.tensor([input_seq_list]).long()


        with torch.no_grad():
            text_gen_logits, _ = text_gen_model(input_indices_text_gen)
        # text_gen_logits shape: (batch=1, seq_len, vocab_size)
        predicted_indices_text_gen = torch.argmax(text_gen_logits, dim=2)[0].tolist()
        # For text_gen, simple join is fine if not using blanks
        generated_text = ''.join([idx2char.get(i, '') for i in predicted_indices_text_gen]).strip()

    return transcription, generated_text

# --- Main Execution ---
if __name__ == "__main__":
    if RUN_TRAINING:
        # 1. Train ASR Model
        asr_X, asr_Y_indices, asr_Y_text = get_dummy_asr_data(num_samples=50, max_seq_len=100) # More samples
        print(f"Sample ASR target text: '{asr_Y_text[0]}' -> indices: {asr_Y_indices[0]}")
        asr_model_instance = SimpleASRModel()
        train_asr_model(asr_model_instance, asr_X, asr_Y_indices, epochs=20, lr=0.001) # More epochs

        # 2. Train Text Generation Model
        # Using a slightly larger vocab size for text gen model if needed, but char2idx is global here.
        text_gen_X, text_gen_Y = get_dummy_text_gen_data(num_samples=50, max_len=15)
        print(f"Sample TextGen input indices: {text_gen_X[0]}")
        print(f"Sample TextGen target indices: {text_gen_Y[0]}")
        text_gen_model_instance = SimpleTextGenModel()
        train_text_gen_model(text_gen_model_instance, text_gen_X, text_gen_Y, epochs=30, lr=0.001) # More epochs
    else:
        print("Skipping training. Attempting to load pre-trained models for inference.")

    # Run the full model for inference
    print("\n--- Running Inference ---")
    if not os.path.exists(AUDIO_FILE_FOR_INFERENCE):
         print(f"Cannot run inference: Sample audio file {AUDIO_FILE_FOR_INFERENCE} not found and gTTS might have failed.")
    else:
        transcription_result, generated_result = run_full_model(AUDIO_FILE_FOR_INFERENCE, ASR_MODEL_PATH, TEXT_GEN_MODEL_PATH)
        print("\n--- Inference Results ---")
        print(f"Audio File Path: {AUDIO_FILE_FOR_INFERENCE}")
        print(f"Transcription from Audio: '{transcription_result}'")
        print(f"Generated Text Response: '{generated_result}'")

    print("\nNote: Models were trained on DUMMY data. Results will be basic.")
    print("For meaningful results, train with large, real datasets.")