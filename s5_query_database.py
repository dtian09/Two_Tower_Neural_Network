'''
Query the ChromaDB database to retrieve the top-k most relevant passages to a query
using Approximate Nearest Neighbor (ANN) search based on cosine similarity.

The query is encoded with the trained Two-Tower BERT query encoder, normalized, 
and passed directly to ChromaDB for ANN retrieval.
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="
os.environ["ABSL_LOGGING"] = "0"
from absl import logging
logging.set_verbosity(logging.ERROR)
import torch
from transformers import BertTokenizer
from chromadb import PersistentClient
from s3_train_tnn_bert import TwoTowerBERTLoRA
from peft import LoraConfig, TaskType
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utilities import download_from_huggingface

# === CONFIG ===
CHROMA_COLLECTION_NAME = "ms_marco_passages_lora"
TOP_K = 3
PERSIST_DIR = "./chroma_db"
query_length=30 #20 #number of tokens

# === Load Tokenizer and Checkpoint ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MODEL_PATH = download_from_huggingface(repo_id = "dtian09/MS_MARCO",
                                       model_or_data_pt = "best_two_tower_lora.pt")
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# === Load LoRA config from checkpoint or default ===
if "config" in checkpoint:
    print("Loaded LoRA config from checkpoint.")
    lora_config = checkpoint["config"]
else:
    print("Using default LoRA config.")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "key", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

# === Initialize Two-Tower Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoTowerBERTLoRA(lora_config).to(device)
model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
model.eval()

search_strategy = input('input search strategy (ann or exact):')

# === Connect to ChromaDB and perform ANN cosine search ===
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

while True:
  query = input("Enter your query: ")

  # === Encode and Normalize Query ===
  with torch.no_grad():
      tokens = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=query_length).to(device)
      outputs = model.query_encoder(**tokens)
      token_embs = outputs.last_hidden_state  # (1, T, H)
      mask = tokens["attention_mask"].unsqueeze(-1)  # (1, T, 1)
      masked_embs = token_embs * mask  # zero out padded tokens
      sum_embs = masked_embs.sum(dim=1)  # (1, H)
      lengths = mask.sum(dim=1)  # (1, 1)
      avg_embs = sum_embs / lengths  # average pooling
      q_vec = torch.nn.functional.normalize(avg_embs, p=2, dim=-1)
      query_embedding = q_vec.squeeze(0).cpu().numpy()
  
  if search_strategy == 'ann': ###Approximate Nearest Neighbor Search
      print("approximate nearest neighbour search")
      results = collection.query( #approximate nearest neighbour search using cosine similiarity score (default)  
          query_embeddings=[query_embedding],
          n_results=TOP_K,
          include=["documents", "metadatas", "distances"]  # distances = 1 - cosine_similarity
      )

      # === Convert ChromaDB distances to cosine similarity ===
      cosine_scores = [1 - d for d in results["distances"][0]]

      # === Display Results ===
      docs = results["documents"][0]
      #metadatas = results["metadatas"][0]
      #ids = results["ids"][0]

      print("\nTop Results (ANN with cosine similarity):")
      for i in range(len(docs)):
          print(f"\nRank {i + 1}")
          print(f"Cosine Similarity: {cosine_scores[i]:.4f}")
          #print(f"Doc ID: {ids[i]}")
          print(f"Text: {docs[i]}")
          #print(f"Metadata: {metadatas[i]}")
  else: #exact top k search
      print("exact top k search.")
      # Retrieve all documents and embeddings
      print("Loading all documents and embeddings from ChromaDB...")
      all_results = collection.get(include=["documents", "embeddings", "metadatas"])

      docs = all_results["documents"]
      metadatas = all_results["metadatas"]
      embeddings = np.array(all_results["embeddings"])

      # === Compute Cosine Similarity Manually ===
      scores = cosine_similarity([query_embedding], embeddings)[0]
      topk_indices = np.argsort(scores)[::-1][:TOP_K]

      # === Display Top-k Results ===
      print("\nTop Results (Exact Search using Cosine Similarity):")
      for rank, idx in enumerate(topk_indices, 1):
          print(f"\nRank {rank}")
          print(f"Cosine Similarity: {scores[idx]:.4f}")
          print(f"Text: {docs[idx]}")
          print(f"Metadata: {metadatas[idx]}")
      