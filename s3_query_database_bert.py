'''
Query the ChromaDB database to retrieve the top-k most relevant passages to a query
using Approximate Nearest Neighbor (ANN) search based on cosine similarity.

The query is encoded with the trained Two-Tower BERT query encoder, normalized, 
and passed directly to ChromaDB for ANN retrieval.

input: chroma_db
       query
       TwoTowerBERTLoRA
output: the most relevant passages using cosine similarity
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="
os.environ["ABSL_LOGGING"] = "0"
os.environ["WANDB_MODE"] = "disabled"
from absl import logging
logging.set_verbosity(logging.ERROR)
import torch
from transformers import BertTokenizer
from chromadb import PersistentClient
from s1_train_tnn_bert import TwoTowerBERTLoRA
from peft import LoraConfig, TaskType
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utilities import download_from_huggingface

class ChromaDBQueryEngine:
    def __init__(self, 
                 chroma_collection_name="ms_marco_passages_lora", 
                 top_k=3, 
                 persist_dir="./chroma_db", 
                 query_length=20):
        self.CHROMA_COLLECTION_NAME = chroma_collection_name
        self.TOP_K = top_k
        self.PERSIST_DIR = persist_dir
        self.query_length = query_length
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model, self.lora_config = self._load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client = PersistentClient(path=self.PERSIST_DIR)
        self.collection = self.client.get_collection(name=self.CHROMA_COLLECTION_NAME)

    def _load_model(self):
        MODEL_PATH = download_from_huggingface(repo_id = "dtian09/MS_MARCO",
                                               model_or_data_pt = "best_two_tower_lora_average_pool.pt")
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
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
        model = TwoTowerBERTLoRA(lora_config)
        model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
        model.eval()
        return model, lora_config

    def encode_query(self, query):
        with torch.no_grad():
            tokens = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=self.query_length).to(self.device)
            outputs = self.model.query_encoder(**tokens)
            token_embs = outputs.last_hidden_state  # (1, T, H)
            mask = tokens["attention_mask"].unsqueeze(-1)  # (1, T, 1)
            masked_embs = token_embs * mask  # zero out padded tokens
            sum_embs = masked_embs.sum(dim=1)  # (1, H)
            lengths = mask.sum(dim=1)  # (1, 1)
            avg_embs = sum_embs / lengths  # average pooling
            q_vec = torch.nn.functional.normalize(avg_embs, p=2, dim=-1)
            query_embedding = q_vec.squeeze(0).cpu().numpy()
        return query_embedding

    def ann_search(self, query_embedding):
        print("approximate nearest neighbour search")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.TOP_K,
            include=["documents", "metadatas", "distances"]
        )
        cosine_scores = [1 - d for d in results["distances"][0]]
        docs = results["documents"][0]
        return [(docs[i], cosine_scores[i]) for i in range(len(docs))]

    def exact_search(self, query_embedding):
        print("exact top k search.")
        print("Loading all documents and embeddings from ChromaDB...")
        all_results = self.collection.get(include=["documents", "embeddings", "metadatas"])
        docs = all_results["documents"]
        metadatas = all_results["metadatas"]
        embeddings = np.array(all_results["embeddings"])
        scores = cosine_similarity([query_embedding], embeddings)[0]
        topk_indices = np.argsort(scores)[::-1][:self.TOP_K]
        return [(docs[idx], scores[idx], metadatas[idx]) for idx in topk_indices]

    def interactive_query(self):
        search_strategy = input('input search strategy (ann or exact):')
        results_list = []
        while True:
            query = input("Enter your query: ")
            query_embedding = self.encode_query(query)
            if search_strategy == 'ann':
                results = self.ann_search(query_embedding)
                print("\nTop Results (ANN with cosine similarity):")
                for i, (doc, score) in enumerate(results):
                    print(f"\nRank {i + 1}")
                    print(f"Cosine Similarity: {score:.4f}")
                    print(f"Text: {doc}")
                results_list.append(results)
            else:
                results = self.exact_search(query_embedding)
                print("\nTop Results (Exact Search using Cosine Similarity):")
                for rank, (doc, score, metadata) in enumerate(results, 1):
                    print(f"\nRank {rank}")
                    print(f"Cosine Similarity: {score:.4f}")
                    print(f"Text: {doc}")
                    print(f"Metadata: {metadata}")
                results_list.append(results)
            # Return results for the current query
            return results

def main():
    engine = ChromaDBQueryEngine()
    engine.interactive_query()

if __name__ == "__main__":
    main()
