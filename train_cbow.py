#!/usr/bin/env python
# coding: utf-8
from collections import Counter
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import tqdm
import wandb
import datetime
import torch.nn.functional as F
import torch

'''
def generate_cbow_data(corpus):
   data = []
   # start from index 2 and end 2 positions before the last word
   # this ensures we always have 2 words before and after the target word
   # for a 5-len sliding window

   for i in range(2, len(corpus) - 2):
       # Get the context words
       # 'i' is the index of the target word
       # [i-2:i] gets the two words before the target word
       # [i+1:i+3] gets the two words after the target word
       context_words = corpus[i-2:i] + corpus[i+1:i+3]
      
       # Get the target word
       target_word = corpus[i]

       # Append the tuple to the data list
       data.append((context_words, target_word))
   return data
'''
def generate_cbow_data(corpus):
    data = []
    # Precompute slices for context words
    for i in range(2, len(corpus) - 2):
        # Use slicing to get context words and target word
        context_words = [corpus[i - 2], corpus[i - 1], corpus[i + 1], corpus[i + 2]]
        target_word = corpus[i]
        data.append((context_words, target_word))
    return data

class CBOWDataset(Dataset):
   def __init__(self, data, word_to_id):
       self.data = data
       self.word_to_id = word_to_id


   # overriding the __len__ method to tell PyTorch how many samples you have
   def __len__(self):
       return len(self.data)


   # overriding the __getitem__ method
   # to tell PyTorch how to retrieve a specific sample and convert it to the format your model expects
   def __getitem__(self, idx):
       context, target = self.data[idx]
       context_ids = torch.tensor([self.word_to_id[word] for word in context], dtype=torch.long)
       target_id = torch.tensor(self.word_to_id[target], dtype=torch.long)
       return context_ids, target_id

class CBOW(torch.nn.Module):
   def __init__(self, vocab_size, embedding_dim):
       super(CBOW, self).__init__()
       self.embedding = nn.Embedding(vocab_size, embedding_dim)
       self.linear = nn.Linear(embedding_dim, vocab_size)

   def forward(self, inputs):
       embed = self.embedding(inputs)
       embed = embed.mean(dim=1)
       out = self.linear(embed)
       probs = F.log_softmax(out, dim=1)
       return probs

corpus = torch.load('filtered_corpus.pt')
word_to_id = torch.load('word_to_id.pt')
id_to_word = torch.load('id_to_word.pt')

vocab_size = len(word_to_id)
print("Vocabulary size:", vocab_size)
cbow_data = generate_cbow_data(corpus)
print("CBOW training data generated")

# Initialize settings
torch.manual_seed(42)

if torch.cuda.is_available():
   print(f"GPU: {torch.cuda.get_device_name(0)}")
   print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
   # Enable cuDNN auto-tuner
   torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timestamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

dataset = CBOWDataset(cbow_data, word_to_id)
train_size = 1 #whole dataset
train_size = int(train_size * len(dataset))
test_size = len(dataset) - train_size
if test_size > 0:
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    test_loader = DataLoader(
                    test_dataset,
                    batch_size=wandb.config.batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=wandb.config.num_workers
                    )
else:#test_size = 0
    train_dataset = dataset

# Initialize wandb with your configuration
wandb.init(
   project="CBOW_MS_MARCO",
   config={
       # Model parameters
       "embedding_dim": 200,
       "vocab_size": vocab_size,
      
       # Training parameters
       "batch_size": 128,
       "learning_rate": 0.001,
       "num_epochs": 5,
       "train_split": 0.7,
      
       # Optimizer parameters
       "weight_decay": 1e-5,
      
       # DataLoader parameters
       "num_workers": 4,
      
       # Architecture details
       "model_type": "CBOW",
       "context_size": 4  # 2 words before + 2 words after
   }
)

# Then use the config values throughout your code
EMBEDDING_DIM = wandb.config.embedding_dim
BATCH_SIZE = wandb.config.batch_size
LEARNING_RATE = wandb.config.learning_rate
NUM_EPOCHS = wandb.config.num_epochs
TRAIN_SPLIT = wandb.config.train_split

# Create data loaders with GPU pinning
train_loader = DataLoader(
   train_dataset,
   batch_size=wandb.config.batch_size,
   shuffle=True,
   pin_memory=True,  # Enable pinning for faster GPU transfer
   num_workers=wandb.config.num_workers     # Use multiple workers for data loading
)

model = CBOW(
   vocab_size=wandb.config.vocab_size,
   embedding_dim=wandb.config.embedding_dim)


model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                            lr=wandb.config.learning_rate,
                            weight_decay=wandb.config.weight_decay)


# simplified training loop
# Training Loop
for epoch in range(wandb.config.num_epochs):
    model.train()
    total_train_loss = 0
    progress = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)
    for inputs, targets in progress:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        
        # Update the progress bar with the current loss
        progress.set_postfix(loss=loss.item())

        wandb.log({'batch_loss': loss.item()})

    # Calculate average training loss
    avg_train_loss = total_train_loss / len(train_loader)

    if test_size > 0:# Evaluate on test set
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)    
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'embedding_dim': wandb.config.embedding_dim,
            'batch_size': wandb.config.batch_size
        })
    else:
        # Log epoch metrics
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'embedding_dim': wandb.config.embedding_dim,
            'batch_size': wandb.config.batch_size
        })

    # Print epoch summary
    if test_size > 0:
        print(f'Epoch {epoch+1}/{wandb.config.num_epochs}: Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
    else:
        print(f'Epoch {epoch+1}/{wandb.config.num_epochs}: Train Loss: {avg_train_loss:.4f}')
    # Save model checkpoint after every epoch
    #checkpoint_path = f"cbow_epoch{epoch+1}_{timestamp}.pt"
    #torch.save(model.state_dict(), checkpoint_path)
    #model_artifact = wandb.Artifact('model-weights', type='model')
    #model_artifact.add_file(checkpoint_path)
    #wandb.log_artifact(model_artifact)

embedding_weights = model.embedding.weight.data.cpu()
embedding_path = f"embeddings_cbow.pt"
torch.save(embedding_weights, embedding_path)
embedding_artifact = wandb.Artifact('embeddings', type='embeddings')
embedding_artifact.add_file(embedding_path)
wandb.log_artifact(embedding_artifact)

# Finish Wandb
wandb.finish()

'''
def get_similar_words(query_word, word_to_ix, ix_to_word, embeddings, top_k=5):
    if query_word not in word_to_ix:
        print(f"'{query_word}' not in vocabulary.")
        return []

    query_idx = word_to_ix[query_word]
    query_embedding = embeddings[query_idx]

    # Compute cosine similarity with all embeddings
    similarities = F.cosine_similarity(query_embedding.unsqueeze(0), embeddings)

    # Get top_k most similar (excluding the query word itself)
    top_indices = similarities.argsort(descending=True)[1:top_k+1]
    similar_words = [(ix_to_word[idx.item()], similarities[idx].item()) for idx in top_indices]

    return similar_words

# Normalize embeddings
embedding_weights = model.embedding.weight.data  # shape: [vocab_size, embedding_dim]
norms = embedding_weights.norm(dim=1, keepdim=True)
normalized_embeddings = embedding_weights / norms

# Test the function
word = "castle"
similar = get_similar_words(word, word_to_id, id_to_word, normalized_embeddings, top_k=10)
print(f"Words similar to {word}:")
for word, score in similar:
    print(f"{word} ({score:.4f})")
'''