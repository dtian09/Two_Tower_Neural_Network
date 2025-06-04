import os
os.environ["WANDB_MODE"] = "disabled"
import uuid
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizer
from chromadb import PersistentClient
from s3_train_tnn_bert import TwoTowerBERTLoRA

# === Settings ===
COLLECTION_NAME = "ms_marco_passages_lora"
N_PASSAGES = 'all'
BATCH_SIZE = 32
MAX_BATCH_SIZE = 5000
PERSIST_DIR = "./chroma_db"
MODEL_PATH = "best_two_tower_lora.pt"

# === Load Tokenizer ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# === Load Checkpoint (state_dict + config) ===
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
lora_config = checkpoint["config"] if "config" in checkpoint else {
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target_modules": ["query", "key", "value"],
    "lora_bias": "none"
}

# === Initialize Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoTowerBERTLoRA(lora_config).to(device)
model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
model.eval()

# === Load Dataset ===
print("Loading MS MARCO dataset...")
if N_PASSAGES == 'all':
    dataset = load_dataset("ms_marco", "v1.1", split="test")
else:
    dataset = load_dataset("ms_marco", "v1.1", split="test").select(range(N_PASSAGES))

# === Extract Passage Texts ===
passagesL = []
for item in dataset:
    passages = item["passages"]
    ps = passages["passage_text"]
    passagesL.extend(ps)

# === Tokenize and Encode Passages ===
print("Tokenizing passages...")
encoded_passagesL = []
with torch.no_grad():
    for i in tqdm(range(0, len(passagesL), BATCH_SIZE)):
        batch_passages = passagesL[i:i+BATCH_SIZE]
        tokens = tokenizer(batch_passages, padding=True, truncation=True, max_length=200, return_tensors="pt").to(device)
        embeddings = model.passage_encoder(**tokens).pooler_output
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        encoded_passagesL.extend(embeddings.cpu())

# === Store in ChromaDB ===
client = PersistentClient(path=PERSIST_DIR)
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    client.delete_collection(name=COLLECTION_NAME)
collection = client.create_collection(name=COLLECTION_NAME)

print("Adding to ChromaDB...")
def chunked(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]

for doc_batch, emb_batch in zip(
    chunked(passagesL, MAX_BATCH_SIZE),
    chunked(encoded_passagesL, MAX_BATCH_SIZE)
):
    collection.add(
        documents=doc_batch,
        embeddings=[e.numpy() for e in emb_batch],
        ids=[str(uuid.uuid4()) for _ in doc_batch],
        metadatas=[{"source": "ms_marco"} for _ in doc_batch]
    )

print(f"Collection '{COLLECTION_NAME}' created with {len(passagesL)} passages.")
print("ChromaDB is ready for querying!")
