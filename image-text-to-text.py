import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Vision Transformer Encoder ---
class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_channels=3, embed_dim=256, num_layers=2, num_heads=4):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embeddings = nn.Conv2d(num_channels, embed_dim, patch_size, patch_size)
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.patch_embeddings(x)              # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)          # (B, num_patches, embed_dim)
        x = x + self.position_embeddings
        x = self.transformer_encoder(x)
        return x

# --- Text Transformer Encoder (for question) ---
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, max_seq_len=32, num_layers=2, num_heads=4):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids):
        x = self.token_embeddings(input_ids) + self.position_embeddings[:, :input_ids.size(1), :]
        x = self.transformer_encoder(x)
        return x

# --- Text Transformer Decoder (for answer generation) ---
class TextDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, max_seq_len=32, num_layers=2, num_heads=4):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, decoder_input_ids, memory):
        x = self.token_embeddings(decoder_input_ids) + self.position_embeddings[:, :decoder_input_ids.size(1), :]
        x = self.transformer_decoder(tgt=x.transpose(0,1), memory=memory.transpose(0,1))
        x = x.transpose(0,1)
        logits = self.output_proj(x)
        return logits

# --- Full VQA Model ---
class ImageQuestionAnsweringModel(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=256, max_seq_len=32):
        super().__init__()
        self.vision_encoder = VisionTransformer(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim, max_seq_len)
        self.decoder = TextDecoder(vocab_size, embed_dim, max_seq_len)

    def forward(self, images, question_ids, answer_input_ids):
        image_embeds = self.vision_encoder(images)             # (B, img_patches, embed_dim)
        question_embeds = self.text_encoder(question_ids)      # (B, q_len, embed_dim)
        fused = torch.cat([image_embeds, question_embeds], dim=1)  # (B, img_patches+q_len, embed_dim)
        logits = self.decoder(answer_input_ids, fused)          # (B, ans_len, vocab_size)
        return logits

# --- Dummy tokenizer (very basic) ---
class DummyTokenizer:
    def __init__(self):
        self.vocab = ['<pad>', '<bos>', '<eos>'] + [f'word{i}' for i in range(1, 997)]
        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        self.id2word = {i: w for i, w in enumerate(self.vocab)}

    def encode(self, text, max_len=32):
        # tokenize by splitting on space, map unknown words to random known tokens
        tokens = text.lower().split()
        ids = [self.word2id.get(t, 10) for t in tokens]
        ids = [self.word2id['<bos>']] + ids[:max_len-2] + [self.word2id['<eos>']]
        # pad if needed
        ids += [self.word2id['<pad>']] * (max_len - len(ids))
        return torch.tensor(ids).unsqueeze(0)  # batch=1

    def decode(self, token_ids):
        tokens = []
        for i in token_ids:
            w = self.id2word.get(i.item(), '<unk>')
            if w == '<eos>':
                break
            if w not in ('<bos>', '<pad>'):
                tokens.append(w)
        return ' '.join(tokens)

# --- Greedy answer generation ---
def generate_answer(model, image_tensor, question_text, tokenizer, max_len=32, device='cpu'):
    model.eval()
    question_ids = tokenizer.encode(question_text, max_len=max_len).to(device)  # (1, seq_len)

    # Prepare image batch
    image_tensor = image_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)

    # Initialize answer generation with <bos> token
    answer_ids = torch.tensor([[tokenizer.word2id['<bos>']]], device=device)

    with torch.no_grad():
        for _ in range(max_len-1):
            logits = model(image_tensor, question_ids, answer_ids)   # (1, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :]                      # (1, vocab_size)
            next_token_id = next_token_logits.argmax(-1).unsqueeze(0) # (1,1)
            answer_ids = torch.cat([answer_ids, next_token_id], dim=1)
            if next_token_id.item() == tokenizer.word2id['<eos>']:
                break

    answer = tokenizer.decode(answer_ids[0])
    return answer

# --- Example usage ---
if __name__ == "__main__":
    device = torch.device('cpu')
    model = ImageQuestionAnsweringModel(vocab_size=1000, embed_dim=256, max_seq_len=32).to(device)
    tokenizer = DummyTokenizer()

    # Dummy random image tensor (simulate your input image)
    dummy_image = torch.randn(3,224, 224)

    # Your question about the image
    question = "What color is the car?"

    # Generate answer (random since model is untrained)
    answer = generate_answer(model, dummy_image , question, tokenizer, max_len=32, device=device)

    print(f"Question: {question}")
    print(f"Answer: {answer}")
