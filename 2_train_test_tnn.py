import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import wandb
import numpy as np
import re
from tqdm import tqdm
# Enable performance optimizations for consistent input shapes
torch.backends.cudnn.benchmark = True

# Init W&B
wandb.init(
    project="two-tower-ms_marco",
    entity="dtian",
    config={
        "hidden_dim": 128,
        "batch_size": 1000, #1200, #1000, 128
        "epochs": 20,
        "max_query_len": 10, #20,
        "max_passage_len": 100, #200,
        "margin": 0.2,
        "lr": 1e-3,
        "patience": 5
    }
)
config = wandb.config

# Text to embedding
def text_to_embedding(text, word_to_id, embedding_matrix, max_len, unk_id=0):
    # remove punctuation and non alphabetic characters
    remove_punctuation = re.sub(r'[^\w\s]', '', text)
    lower_case_words = remove_punctuation.lower()
    tokens = lower_case_words.split()#split by whitespace, tab and newline
    ids = [word_to_id.get(tok, unk_id) for tok in tokens]
    ids = ids[:max_len] + [unk_id] * max(0, max_len - len(ids))
    embeddings = torch.stack([embedding_matrix[i] for i in ids])
    return embeddings

# Dataset class
class TripletDataset(Dataset):
    def __init__(self, dataset, word_to_id, embedding_matrix,
                 max_query_len=20, max_passage_len=200, negative_sampling=False, seed=42):
        self.data = []
        self.embedding_matrix = embedding_matrix
        self.word_to_id = word_to_id
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len
        self.negative_sampling = negative_sampling
        rng = np.random.default_rng(seed)

        total_pos, total_neg, total_used = 0, 0, 0

        for item in dataset:
            query = item["query"]
            passages = item["passages"]

            mask = np.array(passages['is_selected'], dtype=bool)
            ps = np.array(passages['passage_text'])

            pos = ps[mask]
            neg = ps[~mask]

            total_pos += len(pos)
            total_neg += len(neg)

            if len(pos) == 0 or len(neg) == 0:
                continue  # skip if no valid triplets

            # Negative sampling: sample from neg to match pos count
            if self.negative_sampling and len(neg) > len(pos):
                neg = rng.choice(neg, size=len(pos), replace=False)

            # Create all combinations
            for p in pos:
                for n in neg:
                    self.data.append((query, p, n))
                    total_used += 1

        print(f"Total positive passages: {total_pos}")
        print(f"Total negative passages: {total_neg}")
        print(f"Total triplets created: {total_used}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, pos, neg = self.data[idx]
        q_embed = text_to_embedding(query, self.word_to_id, self.embedding_matrix, self.max_query_len)
        p_pos = text_to_embedding(pos, self.word_to_id, self.embedding_matrix, self.max_passage_len)
        p_neg = text_to_embedding(neg, self.word_to_id, self.embedding_matrix, self.max_passage_len)
        return q_embed, p_pos, p_neg

def collate_batch(batch):
    q, p, n = zip(*batch)
    return torch.stack(q), torch.stack(p), torch.stack(n)

class SimpleRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, token_embedding, hidden_state):
        combined = torch.cat((token_embedding, hidden_state), dim=-1)
        hidden_state = torch.tanh(self.linear(combined))
        prediction = self.predictor(hidden_state)
        return hidden_state, prediction

class TwoTowerTripletRNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.query_encoder = SimpleRNNCell(embed_dim, hidden_dim)
        self.passage_encoder = SimpleRNNCell(embed_dim, hidden_dim)

    def encode_sequence(self, embeddings, rnn_cell):
        batch_size, seq_len, _ = embeddings.shape
        hidden_state = torch.zeros(batch_size, rnn_cell.predictor.out_features, device=embeddings.device)
        for t in range(seq_len):
            token_embedding = embeddings[:, t, :]
            hidden_state, _ = rnn_cell(token_embedding, hidden_state)
        return hidden_state #encoding of the input sequence

    def forward(self, query_embs, pos_embs, neg_embs):
        q_vec = self.encode_sequence(query_embs, self.query_encoder)
        p_pos = self.encode_sequence(pos_embs, self.passage_encoder)
        p_neg = self.encode_sequence(neg_embs, self.passage_encoder)
        return q_vec, p_pos, p_neg
'''
class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, embeddings):
        # embeddings: (batch_size, seq_len, input_dim)
        _, hidden = self.rnn(embeddings)  # hidden: (1, batch_size, hidden_dim)
        return hidden.squeeze(0)  # (batch_size, hidden_dim)

class TwoTowerTripletRNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.query_encoder = GRUEncoder(embed_dim, hidden_dim)
        self.passage_encoder = GRUEncoder(embed_dim, hidden_dim)

    def forward(self, query_embs, pos_embs, neg_embs):
        q_vec = self.query_encoder(query_embs)
        p_pos = self.passage_encoder(pos_embs)
        p_neg = self.passage_encoder(neg_embs)
        return q_vec, p_pos, p_neg
'''
def triplet_loss(q_vec, pos_vec, neg_vec, margin=0.2):
    sim_pos = F.cosine_similarity(q_vec, pos_vec, dim=-1)
    sim_neg = F.cosine_similarity(q_vec, neg_vec, dim=-1)
    return F.relu(margin - sim_pos + sim_neg).mean()

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for query, pos, neg in loader:
            query, pos, neg = query.to(device, non_blocking=True), pos.to(device, non_blocking=True), neg.to(device, non_blocking=True)
            q_vec, p_pos, p_neg = model(query, pos, neg)
            sim_pos = F.cosine_similarity(q_vec, p_pos, dim=-1)
            sim_neg = F.cosine_similarity(q_vec, p_neg, dim=-1)
            correct += (sim_pos > sim_neg).sum().item()
            total += query.size(0)
    return correct / total if total > 0 else 0

def train_with_progress_bar(model, train_loader, optimizer, device, epoch, margin):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
    for query, pos, neg in progress_bar:
        query, pos, neg = query.to(device, non_blocking=True), pos.to(device, non_blocking=True), neg.to(device, non_blocking=True)
        q_vec, p_pos, p_neg = model(query, pos, neg)
        loss = triplet_loss(q_vec, p_pos, p_neg, margin=margin)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        avg_loss = total_loss / (progress_bar.n + 1)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    return total_loss / len(train_loader)

def compute_validation_loss(model, val_loader, device, margin):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for query, pos, neg in val_loader:
            query, pos, neg = query.to(device, non_blocking=True), pos.to(device, non_blocking=True), neg.to(device, non_blocking=True)
            q_vec, p_pos, p_neg = model(query, pos, neg)
            loss = F.relu(margin - F.cosine_similarity(q_vec, p_pos, dim=-1) + F.cosine_similarity(q_vec, p_neg, dim=-1)).mean()
            total_loss += loss.item()
    return total_loss / len(val_loader)

def train_validate_test():
    # Load dataset and embeddings
    from huggingface_hub import hf_hub_download
    import torch

    repo_id = "dtian09/MS_MARCO"
    skip_gram_path = hf_hub_download(repo_id=repo_id, filename="skip_gram_model.pt",  repo_type="dataset" )
    word_to_id_path = hf_hub_download(repo_id=repo_id, filename="word_to_id.pt",  repo_type="dataset" )

    state_dict = torch.load(skip_gram_path, map_location="cpu")
    word_to_id = torch.load(word_to_id_path)
    embedding_matrix = state_dict["target_embedding.weight"]
    embed_dim = embedding_matrix.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset splits
    train_data = load_dataset("microsoft/ms_marco", "v1.1", split="train")
    val_data = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
    print("Train data")
    train_dataset = TripletDataset(train_data, word_to_id, embedding_matrix, config.max_query_len, config.max_passage_len, negative_sampling=True)
    print("Validation data")
    val_dataset = TripletDataset(val_data, word_to_id, embedding_matrix, config.max_query_len, config.max_passage_len)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch, pin_memory=True)
    
    model = TwoTowerTripletRNN(embed_dim=embed_dim, hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_acc = 0
    patience = config.patience
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        train_loss = train_with_progress_bar(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            margin=config.margin
        )

        val_acc = evaluate(model, val_loader, device)
        val_loss = compute_validation_loss(model, val_loader, device, margin=config.margin)

        wandb.log({
                   "epoch": epoch + 1,
                   "train_loss": train_loss,
                   "val_loss": val_loss,
                   "val_accuracy": val_acc
                  })

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, "best_two_tower_model.pt")
            wandb.save("best_two_tower_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    # Load best model for final evaluation
    print("evaluate model on test set")
    model=torch.load("best_two_tower_model.pt")
    model.eval() #switch to inference mode
    test_data = load_dataset("microsoft/ms_marco", "v1.1", split="test")
    test_dataset = TripletDataset(test_data, word_to_id, embedding_matrix, config.max_query_len, config.max_passage_len)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch, pin_memory=True)
    test_acc = evaluate(model, test_loader, device)
    wandb.log({"test_accuracy": test_acc})
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    train_validate_test()
    
