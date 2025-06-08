'''
This script stores MS MARCO passages (embedding vectors) in the ChromaDB vector database.
It loads the MS MARCO dataset, encodes the passages using an pretrained dual-encoder model,
and stores the embeddings in ChromaDB.
'''
import os
os.environ["WANDB_MODE"] = "disabled"
import chromadb
from chromadb.config import Settings
from datasets import load_dataset
#from sentence_transformers import SentenceTransformer
import uuid
import torch
from s3_train_test_tnn import text_to_embedding, TwoTowerTripletRNN, SimpleRNNCell
from huggingface_hub import hf_hub_download
from chromadb import PersistentClient
from huggingface_hub import login

HF_TOKEN = os.getenv("HF_TOKEN") #running command: HF_TOKEN=hf_token python script.py

# === Settings ===
COLLECTION_NAME = "ms_marco_passages"
N_PASSAGES = 'all' #1000
MAX_BATCH_SIZE = 5000  # Stay safely under ChromaDB's max
PERSIST_DIR = "./chroma_db"  # <-- Add this line for persistence

print("Loading MS MARCO dataset...")
if N_PASSAGES == 'all':
  dataset = load_dataset("ms_marco", "v1.1", split="test")
else: 
  dataset = load_dataset("ms_marco", "v1.1", split="test").select(range(N_PASSAGES))
passagesL = []
for item in dataset:
    passages = item["passages"]
    ps = passages['passage_text']
    passagesL.extend(ps)

#tokenize the passages and look up embeddings for each passage from the embedding matrix

repo_id = "dtian09/MS_MARCO"
skip_gram_path = hf_hub_download(repo_id=repo_id, filename="skip_gram_model.pt",  repo_type="dataset" )
word_to_id_path = hf_hub_download(repo_id=repo_id, filename="word_to_id.pt",  repo_type="dataset" )
state_dict = torch.load(skip_gram_path, map_location="cpu")
word_to_id = torch.load(word_to_id_path)
embedding_matrix = state_dict["target_embedding.weight"]
embed_dim = embedding_matrix.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('best_two_tower_model.pt').to(device)
model.eval()

from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

BATCH_SIZE = 32
encoded_passagesL = []

# Precompute token embeddings
print("Tokenizing passages...")
all_embeddings = [
    text_to_embedding(p, word_to_id, embedding_matrix, unk_id=0)
    for p in tqdm(passagesL)
]

print("Encoding in batches...")
with torch.no_grad():
    for i in range(0, len(all_embeddings), BATCH_SIZE):
        batch = all_embeddings[i:i + BATCH_SIZE]

        # Pad batch to uniform length
        padded_batch = pad_sequence(batch, batch_first=True).to(device)  # [B, max_seq_len, D]

        # Encode with RNN
        encodings = model.encode_sequence(padded_batch, model.passage_encoder)  # [B, hidden_dim]

        encoded_passagesL.extend(encodings.cpu())  # Store as CPU tensors (or .numpy())

PERSIST_DIR = "./chroma_db"

# === Initialize ChromaDB with Persistence ===
client = PersistentClient(path=PERSIST_DIR)

# Create or clear collection
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    client.delete_collection(name=COLLECTION_NAME)
collection = client.create_collection(name=COLLECTION_NAME)

# === Add to ChromaDB in batches ===
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
        embeddings=[e.squeeze(0).detach().cpu().numpy() for e in emb_batch],
        ids=[str(uuid.uuid4()) for _ in doc_batch],
        metadatas=[{"source": "ms_marco"} for _ in doc_batch]
   )

print(f"Collection '{COLLECTION_NAME}' created with {len(passagesL)} passages.")
print("ChromaDB is ready for querying!")
# Note: You can now run the search engine script to query this collection.
