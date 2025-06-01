''' 
This script implements a Skip-gram model using PyTorch.
input: djb_cbow_dataset_corpus.pt (output of djb_word2vec_tokenise.py)
       djb_cbow_dataset_word_to_idx.pt
       djb_cbow_dataset_idx_to_word.pt
output: skip_gram_embeddings.pt file (dictionary with key= word id, value=embedding
'''
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random
import wandb
import tokenizer as tk

wandb.init(project="skipgram-sgns", config={
    "window_size": 2,
    "embedding_dim": 200,#100,
    "epochs": 30, #20, #100,
    "learning_rate": 0.01,
    "batch_size": 128,
    #"dropout_rate": 0.3 ,
    "training_percentage": 0.7
})

config = wandb.config
window_size = config.window_size
embedding_dim = config.embedding_dim
epochs = config.epochs
learning_rate = config.learning_rate
batch_size = config.batch_size
training_percentage = config.training_percentage
#dropout_rate = config.dropout_rate

#words = torch.load('djb_cbow_dataset_corpus.pt')
#word2idx = torch.load('djb_cbow_dataset_word_to_idx.pt')
#idx2word = torch.load('djb_cbow_dataset_idx_to_word.pt')

words = torch.load('filtered_corpus.pt')
word2idx = torch.load('word_to_id.pt')
idx2word = torch.load('id_to_word.pt')

#select a subset of vocabulary
#vocab_size=10000
vocab_size='all vocab'
random.seed(42)

if vocab_size == 'all vocab':
    selected_words = set(word2idx.keys())#unique words of vocab size
    vocab_size = len(selected_words)
else:#Randomly sample a subset of words
    selected_words = set(random.sample(list(word2idx.keys()), vocab_size))#unique words of vocab size
filtered_corpus = [word for word in words if word in selected_words]#corpus of vocab size that include sequence of words and repeats of words
# New vocabulary based only on selected words
filtered_vocab = sorted(set(filtered_corpus))
word2idx = {word: idx for idx, word in enumerate(filtered_vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
print('vocab size:', len(selected_words))

# Generate Skip-gram training pairs
def generate_skipgram_data(words, corpus, window_size):
    pairs = []
    for i in range(0,len(words)):
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

# Shuffle and split into training data and testing data
random.shuffle(data_pairs)
split_index = int(training_percentage * len(data_pairs))
train_pairs = data_pairs[:split_index]
test_pairs = data_pairs[split_index:]

print('Training pairs:', len(train_pairs))
print('Test pairs:', len(test_pairs))

class NegativeSampler:
    def __init__(self, vocab_size, word_freqs, num_negatives=5):
        self.vocab_size = vocab_size
        self.num_negatives = num_negatives
        freq = torch.tensor([word_freqs.get(idx2word[i], 1) for i in range(vocab_size)], dtype=torch.float)
        self.weights = freq.pow(0.75)
        self.weights /= self.weights.sum()

    def sample(self, batch_size):
        return torch.multinomial(self.weights, batch_size * self.num_negatives, replacement=True).view(batch_size, self.num_negatives)

# Use actual word frequencies to bias negative sampling
word_freqs = defaultdict(int)
for word in filtered_corpus:
    word_freqs[word] += 1

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
'''
# Update model to compute dot products for positive and negative pairs
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout_rate=0.3):  # You can change 0.3 to any value
        super(SkipGramNegSampling, self).__init__()
        self.target_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, target, context, negatives):
        target_emb = self.dropout(self.target_embedding(target))         # [B, D]
        context_emb = self.dropout(self.context_embedding(context))      # [B, D]
        neg_emb = self.dropout(self.context_embedding(negatives))        # [B, N, D]

        pos_score = torch.sum(target_emb * context_emb, dim=1)           # [B]
        neg_score = torch.bmm(neg_emb.neg(), target_emb.unsqueeze(2)).squeeze()  # [B, N]

        return pos_score, neg_score
'''
'''
class SkipGramNegSampling(nn.Module):#with hidden layer
    def __init__(self, vocab_size, embedding_dim, hidden_dim=128):
        super(SkipGramNegSampling, self).__init__()
        self.target_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.context_embedding = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, target, context, negatives):
        target_emb = self.target_embedding(target)        # [B, D]
        target_hidden = torch.relu(self.hidden(target_emb))  # [B, H]

        context_emb = self.context_embedding(context)     # [B, H]
        neg_emb = self.context_embedding(negatives)       # [B, N, H]

        # Positive score
        pos_score = torch.sum(target_hidden * context_emb, dim=1)  # [B]

        # Negative scores
        neg_score = torch.bmm(neg_emb.neg(), target_hidden.unsqueeze(2)).squeeze(2)  # [B, N]

        return pos_score, neg_score
'''
# Initialize model and sampler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkipGramNegSampling(vocab_size, embedding_dim).to(device)
sampler = NegativeSampler(vocab_size, word_freqs)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=1e-5)

def batchify(pairs, batch_size):
    for i in range(0, len(pairs), batch_size):
        yield pairs[i:i + batch_size]

print('training the skip-gram model with negative sampling')

# Training loop using negative sampling
for epoch in range(epochs):
    total_loss = 0
    random.shuffle(train_pairs)

    for batch in batchify(train_pairs, batch_size):
        target_indices = torch.tensor([pair[0] for pair in batch], dtype=torch.long).to(device)
        context_indices = torch.tensor([pair[1] for pair in batch], dtype=torch.long).to(device)
        negative_indices = sampler.sample(len(batch)).to(device)  # [B, N]

        pos_score, neg_score = model(target_indices, context_indices, negative_indices)  # [B], [B, N]

        # Positive labels are 1, negative labels are 0
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
    wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})
    # --- Evaluation on test set ---
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in batchify(test_pairs, batch_size):
            target_indices = torch.tensor([pair[0] for pair in batch], dtype=torch.long).to(device)
            context_indices = torch.tensor([pair[1] for pair in batch], dtype=torch.long).to(device)
            negative_indices = sampler.sample(len(batch)).to(device)

            pos_score, neg_score = model(target_indices, context_indices, negative_indices)

            pos_labels = torch.ones_like(pos_score)
            neg_labels = torch.zeros_like(neg_score)

            loss_pos = criterion(pos_score, pos_labels)
            loss_neg = criterion(neg_score, neg_labels)

            loss = loss_pos + loss_neg.mean()
            test_loss += loss.item()

    avg_test_loss = test_loss / (len(test_pairs) // batch_size)
    wandb.log({"epoch": epoch + 1, "test_loss": avg_test_loss})
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
    model.train()  # Back to training mode

torch.save(model.state_dict(), "skip_gram_model.pt")
print("Model saved to skip_gram_model.pt")

# save embeddings to dictionary 
#embeddings = {}
#device = torch.device("cpu")
#model.to(device)
#for word in word2idx:
#    idx = torch.tensor([word2idx[word]])
#    emb = model.target_embedding(idx).detach().numpy()
#    embeddings[int(idx)] = emb

# Save the dictionary to a .pt file
#torch.save(embeddings, 'skip_gram_negative_sampling_embeddings.pt')