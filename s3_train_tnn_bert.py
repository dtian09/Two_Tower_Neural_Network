'''
Train a two-tower network with 2 BERT encoders
Fine-tune BERT using LoRA on MS MARCO  
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import wandb
import numpy as np
import re
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import hf_hub_download

# Enable performance optimizations for consistent input shapes
torch.backends.cudnn.benchmark = True

# Init W&B
wandb.init(
    project="two-tower-ms_marco-lora",
    entity="dtian",
    config={
        "batch_size": 8,
        "epochs": 10,
        "max_query_len": 20,
        "max_passage_len": 200,
        "margin": 0.2,
        "lr": 2e-5,
        "patience": 3,
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": ["query", "key", "value"],
        "lora_bias": "none"
    }
)
config = wandb.config

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataset
class TripletDataset(Dataset):
    def __init__(self, dataset, negative_sampling=False, seed=42):
        self.data = []
        rng = np.random.default_rng(seed)

        for item in dataset:
            query = item["query"]
            passages = item["passages"]
            mask = np.array(passages['is_selected'], dtype=bool)
            ps = np.array(passages['passage_text'])

            pos = ps[mask]
            neg = ps[~mask]

            if len(pos) == 0 or len(neg) == 0:
                continue

            if negative_sampling and len(neg) > len(pos):
                neg = rng.choice(neg, size=len(pos), replace=False)

            for p in pos:
                for n in neg:
                    self.data.append((query, p, n))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_batch(batch):
    q, p, n = zip(*batch)
    query_tokens = tokenizer(list(q), padding=True, truncation=True, max_length=config.max_query_len, return_tensors="pt")
    pos_tokens = tokenizer(list(p), padding=True, truncation=True, max_length=config.max_passage_len, return_tensors="pt")
    neg_tokens = tokenizer(list(n), padding=True, truncation=True, max_length=config.max_passage_len, return_tensors="pt")
    return query_tokens, pos_tokens, neg_tokens

# LoRA-wrapped Two-Tower model
class TwoTowerBERTLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            task_type=TaskType.FEATURE_EXTRACTION
        )
        base_q = BertModel.from_pretrained("bert-base-uncased")
        base_p = BertModel.from_pretrained("bert-base-uncased")
        self.query_encoder = get_peft_model(base_q, lora_cfg)
        self.passage_encoder = get_peft_model(base_p, lora_cfg)

    def mean_pool(self, outputs, attention_mask):
        token_embs = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
        return (token_embs * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

    def forward(self, q_inputs, pos_inputs, neg_inputs):
        q_out = self.query_encoder(**q_inputs)
        p_pos_out = self.passage_encoder(**pos_inputs)
        p_neg_out = self.passage_encoder(**neg_inputs)

        q_vec = self.mean_pool(q_out, q_inputs['attention_mask'])
        p_pos = self.mean_pool(p_pos_out, pos_inputs['attention_mask'])
        p_neg = self.mean_pool(p_neg_out, neg_inputs['attention_mask'])
        return q_vec, p_pos, p_neg


def triplet_loss(q_vec, pos_vec, neg_vec, margin=0.2):
    q_vec = F.normalize(q_vec, p=2, dim=-1)
    pos_vec = F.normalize(pos_vec, p=2, dim=-1)
    neg_vec = F.normalize(neg_vec, p=2, dim=-1)

    sim_pos = F.cosine_similarity(q_vec, pos_vec, dim=-1)
    sim_neg = F.cosine_similarity(q_vec, neg_vec, dim=-1)
    return F.relu(margin - sim_pos + sim_neg).mean()

def train_with_progress_bar(model, loader, optimizer, scheduler, device, epoch, margin):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
    for q_inputs, pos_inputs, neg_inputs in progress_bar:
        q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
        pos_inputs = {k: v.to(device) for k, v in pos_inputs.items()}
        neg_inputs = {k: v.to(device) for k, v in neg_inputs.items()}

        q_vec, p_pos, p_neg = model(q_inputs, pos_inputs, neg_inputs)
        loss = triplet_loss(q_vec, p_pos, p_neg, margin)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss / (progress_bar.n + 1)
        progress_bar.set_postfix(loss=f"{avg_loss:.2f}")

    return total_loss / len(loader)

def compute_validation_loss(model, val_loader, device, margin):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for q_inputs, pos_inputs, neg_inputs in val_loader:
            q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
            pos_inputs = {k: v.to(device) for k, v in pos_inputs.items()}
            neg_inputs = {k: v.to(device) for k, v in neg_inputs.items()}
            q_vec, p_pos, p_neg = model(q_inputs, pos_inputs, neg_inputs)
            loss = triplet_loss(q_vec, p_pos, p_neg, margin)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def train_validate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_dataset("microsoft/ms_marco", "v1.1", split="train")
    val_data = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
    train_dataset = TripletDataset(train_data, negative_sampling=True)
    val_dataset = TripletDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)

    model = TwoTowerBERTLoRA().to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.epochs):
        train_loss = round(train_with_progress_bar(model, train_loader, optimizer, scheduler, device, epoch, config.margin), 2)
        val_loss = round(compute_validation_loss(model, val_loader, device, config.margin), 2)

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[INFO] Epoch {epoch+1}: Train Loss = {train_loss:.2f}, Val Loss = {val_loss:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_two_tower_lora.pt")
            wandb.save("best_two_tower_lora.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    train_validate()