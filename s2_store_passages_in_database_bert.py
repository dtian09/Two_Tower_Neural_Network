'''
build a retrieval system (ChromaDB embeddings database) which contains MS MARCO passages encoded using the trained dual-encoder network
input: MS MARCO passages,
       TwoTowerBERTLoRA (passage encoder)
output: a ChromaDB database (chroma_db)
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="
os.environ["ABSL_LOGGING"] = "0"
os.environ["WANDB_MODE"] = "disabled"
from absl import logging
logging.set_verbosity(logging.ERROR)
import uuid
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizer
from chromadb import PersistentClient
from peft import LoraConfig, TaskType
from utilities import download_from_huggingface
from s1_train_tnn_bert import TwoTowerBERTLoRA

class PassageDatabaseBuilder:
    def __init__(self, 
                 collection_name="ms_marco_passages_lora",
                 n_queries=1000,
                 batch_size=8,
                 max_batch_size=5000,
                 persist_dir="./chroma_db"):
        self.COLLECTION_NAME = collection_name
        self.N_QUERIES = n_queries
        self.BATCH_SIZE = batch_size
        self.MAX_BATCH_SIZE = max_batch_size
        self.PERSIST_DIR = persist_dir
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model, self.lora_config = self._load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client = PersistentClient(path=self.PERSIST_DIR)

    def _load_model(self):
        MODEL_PATH = download_from_huggingface(repo_id = "dtian09/MS_MARCO",
                                               model_or_data_pt = "best_two_tower_lora_average_pool.pt")
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        if "config" in checkpoint:
            print('load lora_config from checkpoint.')
            lora_config = checkpoint["config"]
        else:
            print('load a specified lora_config.')
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["query", "key", "value"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
        model = TwoTowerBERTLoRA(lora_config)
        model.to(model.device)
        model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
        model.eval()
        return model, lora_config

    def load_passages(self):
        print("Loading MS MARCO dataset...")
        if self.N_QUERIES == 'all':
            dataset = load_dataset("ms_marco", "v1.1", split="test")
        else:
            dataset = load_dataset("ms_marco", "v1.1", split="test").select(range(self.N_QUERIES))
        passagesL = []
        for item in dataset:
            passages = item["passages"]
            ps = passages["passage_text"]
            passagesL.extend(ps)
        return passagesL

    def encode_passages(self, passagesL):
        print("Tokenizing passages...")
        encoded_passagesL = []
        with torch.no_grad():
            for i in tqdm(range(0, len(passagesL), self.BATCH_SIZE)):
                batch_passages = passagesL[i:i+self.BATCH_SIZE]
                tokens = self.tokenizer(batch_passages, padding=True, truncation=True, max_length=200, return_tensors="pt").to(self.device)
                outputs = self.model.passage_encoder(**tokens)
                token_embs = outputs.last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                embeddings = (token_embs * mask).sum(dim=1) / mask.sum(dim=1)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
                encoded_passagesL.extend(embeddings.cpu())
        return encoded_passagesL

    def store_in_chromadb(self, passagesL, encoded_passagesL):
        if self.COLLECTION_NAME in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(name=self.COLLECTION_NAME)
        collection = self.client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print("Adding to ChromaDB...")
        def chunked(data, size):
            for i in range(0, len(data), size):
                yield data[i:i + size]
        for doc_batch, emb_batch in zip(
            chunked(passagesL, self.MAX_BATCH_SIZE),
            chunked(encoded_passagesL, self.MAX_BATCH_SIZE)
        ):
            collection.add(
                documents=doc_batch,
                embeddings=[e.numpy() for e in emb_batch],
                ids=[str(uuid.uuid4()) for _ in doc_batch],
                metadatas=[{"source": "ms_marco"} for _ in doc_batch]
            )
        print(f"Collection '{self.COLLECTION_NAME}' created with {len(passagesL)} passages.")
        print("ChromaDB is ready for querying!")

    def build(self):
        passagesL = self.load_passages()
        encoded_passagesL = self.encode_passages(passagesL)
        self.store_in_chromadb(passagesL, encoded_passagesL)


def main():
    builder = PassageDatabaseBuilder(n_queries=10)
    builder.build()

if __name__ == "__main__":
    main()
