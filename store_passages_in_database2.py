
'''
This script creates a ChromaDB collection from the MS MARCO passage dataset.
It loads the dataset, encodes the passages using a SentenceTransformer model,
and stores the embeddings in ChromaDB.
'''
import chromadb
from chromadb.config import Settings
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import uuid

# === Settings ===
COLLECTION_NAME = "ms_marco_passages"
N_PASSAGES = 1000  # Load more if you want
MAX_BATCH_SIZE = 5000  # Stay safely under ChromaDB's max

# === Load MS MARCO Dataset ===
print("Loading MS MARCO dataset...")
dataset = load_dataset("ms_marco", "v1.1", split="train").select(range(N_PASSAGES))
texts = []
for item in dataset:
    passages = item["passages"]
    ps = passages['passage_text']
    texts.extend(ps)

# === Load SentenceTransformer Encoder ===
print("Encoding passages...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# === Initialize ChromaDB ===
client = chromadb.Client(Settings())

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
    chunked(texts, MAX_BATCH_SIZE),
    chunked(embeddings.tolist(), MAX_BATCH_SIZE)
):
    collection.add(
        documents=doc_batch,
        embeddings=emb_batch,
        ids=[str(uuid.uuid4()) for _ in doc_batch],
        metadatas=[{"source": "ms_marco"} for _ in doc_batch]
    )

print(f"Collection '{COLLECTION_NAME}' created with {len(texts)} passages.")
