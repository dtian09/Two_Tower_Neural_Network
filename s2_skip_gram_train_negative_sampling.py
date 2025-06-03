'''
This script implements a Skip-gram model using PyTorch.
input: filtered_corpus.pt (output of filter_corpus.py)
       word_to_id.pt
       id_to_word.pt
output: skip_gram_model.pt (model weights)
'''
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random
import wandb
from tqdm import tqdm

# Initialize Weights & Biases
wandb.init(project="skipgram-sgns", 
          entity="dtian", 
          config={
              "window_size": 2,
              "embedding_dim": 200,
              "epochs": 30,
              "learning_rate": 0.01,
              "batch_size": 128,
              "training_percentage": 0.7,
              "patience": 3
          })

# Load config
config = wandb.config
window_size = config.window_size
embedding_dim = config.embedding_dim
epochs = config.epochs
learning_rate = config.learning_rate
batch_size = config.batch_size
training_percentage = config.training_percentage
patience = config.patience
# Load data
words = torch.load('filtered_corpus.pt')
word2idx = torch.load('word_to_id.pt')
idx2word = torch.load('id_to_word.pt')

# Select subset of vocab (or full)
vocab_size = 'all vocab'
random.seed(42)

if vocab_size == 'all vocab':
    selected_words = set(word2idx.keys())
    vocab_size = len(selected_words)
else:
    selected_words = set(random.sample(list(word2idx.keys()), vocab_size))
filtered_corpus = [word for word in words if word in selected_words]
filtered_vocab = sorted(set(filtered_corpus))
word2idx = {word: idx for idx, word in enumerate(filtered_vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
print('vocab size:', len(selected_words))

# Generate Skip-gram training pairs
def generate_skipgram_data(words, corpus, window_size):
    pairs = []
    for i in range(len(words)):
        target = words[i]
        target_idx = word2idx[target]
        for j in range(max(0, i - window_size), min(len(corpus), i + window_size + 1)):
            if i != j:
                context_idx = word2idx[corpus[j]]
                pairs.append((target_idx, context_idx))
    return pairs

print('generating skip-gram data')
data_pairs = generate_skipgram_data(list(selected_words), filtered_corpus, window_size)
print('skip-gram data is generated')
print('number of training pairs before split:', len(data_pairs))

# Shuffle and split into train/val sets
random.shuffle(data_pairs)
split_index = int(training_percentage * len(data_pairs))
train_pairs = data_pairs[:split_index]
val_pairs = data_pairs[split_index:]

print('Training pairs:', len(train_pairs))
print('Validation pairs:', len(val_pairs))

# Negative Sampler
class NegativeSampler:
    def __init__(self, vocab_size, word_freqs, num_negatives=5):
        self.vocab_size = vocab_size
        self.num_negatives = num_negatives
        freq = torch.tensor([word_freqs.get(idx2word[i], 1) for i in range(vocab_size)], dtype=torch.float)
        self.weights = freq.pow(0.75)
        self.weights /= self.weights.sum()

    def sample(self, batch_size):
        return torch.multinomial(self.weights, batch_size * self.num_negatives, replacement=True).view(batch_size, self.num_negatives)

# Compute word frequencies
word_freqs = defaultdict(int)
for word in filtered_corpus:
    word_freqs[word] += 1

# Skip-gram model with negative sampling
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNegSampling, self).__init__()
        self.target_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target, context, negatives):
        target_emb = self.target_embedding(target)                  # [B, D]
        context_emb = self.context_embedding(context)               # [B, D]
        neg_emb = self.context_embedding(negatives)                 # [B, N, D]
        pos_score = torch.sum(target_emb * context_emb, dim=1)      # [B]
        neg_score = torch.bmm(neg_emb.neg(), target_emb.unsqueeze(2)).squeeze()  # [B, N]
        return pos_score, neg_score

# Training utilities
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkipGramNegSampling(vocab_size, embedding_dim).to(device)
sampler = NegativeSampler(vocab_size, word_freqs)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def batchify(pairs, batch_size):
    for i in range(0, len(pairs), batch_size):
        yield pairs[i:i + batch_size]

# Training loop with early stopping and progress bar
print('training the skip-gram model with negative sampling')

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    total_loss = 0
    random.shuffle(train_pairs)

    print(f"\nEpoch {epoch+1}/{epochs}")
    for batch in tqdm(batchify(train_pairs, batch_size), desc="Training", leave=False):
        target_indices = torch.tensor([pair[0] for pair in batch], dtype=torch.long).to(device)
        context_indices = torch.tensor([pair[1] for pair in batch], dtype=torch.long).to(device)
        negative_indices = sampler.sample(len(batch)).to(device)

        pos_score, neg_score = model(target_indices, context_indices, negative_indices)

        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)

        loss_pos = criterion(pos_score, pos_labels)
        loss_neg = criterion(neg_score, neg_labels)

        loss = loss_pos + loss_neg.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (len(train_pairs) // batch_size)
    avg_loss = round(avg_loss, 2)
    # --- Validation ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(batchify(val_pairs, batch_size), desc="Validating", leave=False):
            target_indices = torch.tensor([pair[0] for pair in batch], dtype=torch.long).to(device)
            context_indices = torch.tensor([pair[1] for pair in batch], dtype=torch.long).to(device)
            negative_indices = sampler.sample(len(batch)).to(device)

            pos_score, neg_score = model(target_indices, context_indices, negative_indices)

            pos_labels = torch.ones_like(pos_score)
            neg_labels = torch.zeros_like(neg_score)

            loss_pos = criterion(pos_score, pos_labels)
            loss_neg = criterion(neg_score, neg_labels)

            loss = loss_pos + loss_neg.mean()
            val_loss += loss.item()

    avg_val_loss = val_loss / (len(val_pairs) // batch_size)
    avg_val_loss = round(avg_val_loss, 2)
    
    print(f"Train Loss: {avg_loss}, Val Loss: {avg_val_loss}")
    
    wandb.log({
      "epoch": epoch + 1,
      "train_loss": avg_loss,
      "val_loss": avg_val_loss
    })

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "skip_gram_model.pt")
        print("Model saved to skip_gram_model.pt (improved validation loss)")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
            break

    model.train()

#save only embeddings to dictionary 
#embeddings = {}
#device = torch.device("cpu")
#model.to(device)
#for word in word2idx:
#    idx = torch.tensor([word2idx[word]])
#    emb = model.target_embedding(idx).detach().numpy()
#    embeddings[int(idx)] = emb

# Save the dictionary to a .pt file
#torch.save(embeddings, 'skip_gram_negative_sampling_embeddings.pt')
