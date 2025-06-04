import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Sample text data
text = "hello world"

# Create a set of unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mappings from characters to indices and vice versa
char_to_idx = { ch:i for i,ch in enumerate(chars) }
idx_to_char = { i:ch for i,ch in enumerate(chars) }

# Encode the entire text into indices
data = [char_to_idx[c] for c in text]

# Define the model
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Hyperparameters
embedding_dim = 10
hidden_dim = 20
learning_rate = 0.01
epochs = 100

model = SimpleLLM(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Prepare training data
inputs = torch.tensor(data[:-1], dtype=torch.long)
targets = torch.tensor(data[1:], dtype=torch.long)

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Text generation
input_char = text[0]
input_idx = torch.tensor([char_to_idx[input_char]], dtype=torch.long)

generated_text = input_char

# Generate 10 characters
for _ in range(10):
    output = model(input_idx)
    predicted_idx = torch.argmax(output, dim=1).item()
    predicted_char = idx_to_char[predicted_idx]
    generated_text += predicted_char
    input_idx = torch.tensor([predicted_idx], dtype=torch.long)

print("Generated Text:", generated_text)
