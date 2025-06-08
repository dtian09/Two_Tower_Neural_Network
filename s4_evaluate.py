'''
evaluate performance of the retrieval system (database) using performance metrics: recall@k, precision@k, MRR
The queries should match the passages in the database.
e.g. if the database contains the passages of the first 1000 queries, the input queries should be the first 1000 queries.
The database is queried using ANN search based on cosine similarity.

input: chroma_db
       query
       TwoTowerBERTLoRA
output: performance metrics


'''
import torch
from transformers import BertTokenizer
from datasets import load_dataset
from chromadb import PersistentClient
from s3_train_tnn_bert import TwoTowerBERTLoRA
from peft import LoraConfig, TaskType
from utilities import download_from_huggingface
import numpy as np
from tqdm import tqdm

class RetrievalEvaluator:
    def __init__(self, 
                 collection_name="ms_marco_passages_lora",
                 persist_dir="./chroma_db",
                 top_k=10,
                 query_length=20,
                 n_eval_queries=1000):
        self.COLLECTION_NAME = collection_name
        self.PERSIST_DIR = persist_dir
        self.TOP_K = top_k
        self.QUERY_LENGTH = query_length
        self.N_EVAL_QUERIES = n_eval_queries
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = self._load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.client = PersistentClient(path=self.PERSIST_DIR)
        self.collection = self.client.get_collection(name=self.COLLECTION_NAME)

    def _load_model(self):
        MODEL_PATH = download_from_huggingface(repo_id = "dtian09/MS_MARCO",
                                               model_or_data_pt = "best_two_tower_lora_average_pool.pt")
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        if "config" in checkpoint:
            lora_config = checkpoint["config"]
        else:
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
        return model

    @staticmethod
    def recall_at_k(relevant, retrieved, k):
        return int(any(doc in relevant for doc in retrieved[:k]))

    @staticmethod
    def precision_at_k(relevant, retrieved, k):
        retrieved_k = retrieved[:k]
        return sum(doc in relevant for doc in retrieved_k) / k

    @staticmethod
    def mrr_at_k(relevant, retrieved, k):
        for rank, doc in enumerate(retrieved[:k], 1):
            if doc in relevant:
                return 1.0 / rank
        return 0.0

    def evaluate(self):
        # Use only the first 1000 queries of the test set to match the database
        test_data = load_dataset("ms_marco", "v1.1", split="test").select(range(self.N_EVAL_QUERIES))
        recall_scores = []
        precision_scores = []
        mrr_scores = []
        for item in tqdm(test_data, desc="Evaluating"):
            query = item["query"]
            passages = item["passages"]
            relevant_passages = set(np.array(passages["passage_text"])[np.array(passages["is_selected"], dtype=bool)])
            if not relevant_passages:
                continue
            with torch.no_grad():
                tokens = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=self.QUERY_LENGTH).to(self.device)
                outputs = self.model.query_encoder(**tokens)
                token_embs = outputs.last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                sum_embs = (token_embs * mask).sum(dim=1)
                lengths = mask.sum(dim=1)
                avg_embs = sum_embs / lengths
                q_vec = torch.nn.functional.normalize(avg_embs, p=2, dim=-1)
                query_embedding = q_vec.squeeze(0).cpu().numpy()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.TOP_K,
                include=["documents"]
            )
            retrieved_docs = results["documents"][0]
            recall_scores.append(self.recall_at_k(relevant_passages, retrieved_docs, self.TOP_K))
            precision_scores.append(self.precision_at_k(relevant_passages, retrieved_docs, self.TOP_K))
            mrr_scores.append(self.mrr_at_k(relevant_passages, retrieved_docs, self.TOP_K))
        print(f"Recall@{self.TOP_K}: {np.mean(recall_scores):.4f}")
        print(f"Precision@{self.TOP_K}: {np.mean(precision_scores):.4f}")
        print(f"MRR@{self.TOP_K}: {np.mean(mrr_scores):.4f}")

def main():
    evaluator = RetrievalEvaluator()
    evaluator.evaluate()

if __name__ == "__main__":
    main()