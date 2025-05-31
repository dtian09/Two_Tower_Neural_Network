
import torch
from transformers import AutoTokenizer, AutoModel
from chromadb import PersistentClient
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

client = PersistentClient(path="./chroma_db")
collections = client.list_collections()
print([col.name for col in collections])

# === CONFIG ===
QUERY_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "ms_marco_passages"
TOP_K = 10

# === Load Encoder ===
tokenizer = AutoTokenizer.from_pretrained(QUERY_ENCODER_NAME)
model = AutoModel.from_pretrained(QUERY_ENCODER_NAME)
model.eval()

# === Get Query from CLI ===
query = input("Enter your query: ")

# === Encode Query ===
with torch.no_grad():
    encoded_input = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    model_output = model(**encoded_input)
    query_embedding = model_output.last_hidden_state.mean(dim=1).squeeze().numpy()

# === Connect to ChromaDB ===
collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

# === Query ChromaDB ===
# You must use query_embeddings if you already have them
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=TOP_K
)

# === Rank by Cosine Similarity ===
docs = results["documents"][0]
metadatas = results["metadatas"][0]
ids = results["ids"][0]
embeddings = results["embeddings"][0]

scores = cosine_similarity([query_embedding], embeddings)[0]
ranked_indices = np.argsort(scores)[::-1]  # Descending

print("\nTop Results:")
for idx in ranked_indices:
    print(f"\nRank {idx+1}")
    print(f"Score: {scores[idx]:.4f}")
    print(f"Doc ID: {ids[idx]}")
    print(f"Text: {docs[idx]}")
    print(f"Metadata: {metadatas[idx]}")
