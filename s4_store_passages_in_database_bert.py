import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="
os.environ["ABSL_LOGGING"] = "0"
#os.environ["WANDB_MODE"] = "disabled"
from absl import logging
logging.set_verbosity(logging.ERROR)
import uuid
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizer
from chromadb import PersistentClient
import torch.nn as nn
from transformers import BertModel
from peft import get_peft_model, LoraConfig, TaskType
from utilities import download_from_huggingface
from s3_train_tnn_bert import TwoTowerBERTLoRA

if __name__ == "__main__":
    # === Settings ===
    COLLECTION_NAME = "ms_marco_passages_lora"
    N_QUERIES = 100 #1000 #100 #'all' (each row of MS MARCO corresponds to a query)
    BATCH_SIZE = 8 #batch size of tokenization
    MAX_BATCH_SIZE = 5000 #batch size of chunking passages
    PERSIST_DIR = "./chroma_db"
    
    # === Load Tokenizer ===
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # === Load Checkpoint (state_dict + config) ===
    MODEL_PATH = download_from_huggingface(repo_id = "dtian09/MS_MARCO",
                                       model_or_data_pt = "best_two_tower_lora.pt")
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


    # === Initialize Model ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TwoTowerBERTLoRA(lora_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    model.eval()

    # === Load Dataset ===
    print("Loading MS MARCO dataset...")
    if N_QUERIES == 'all':
        dataset = load_dataset("ms_marco", "v1.1", split="train")
    else:
        dataset = load_dataset("ms_marco", "v1.1", split="train").select(range(N_QUERIES))

    # === Extract Passage Texts ===
    passagesL = []
    for item in dataset:
        passages = item["passages"]
        ps = passages["passage_text"]
        passagesL.extend(ps)
        #debug
        #print(item["query"])

    # === Tokenize and Encode Passages ===
    print("Tokenizing passages...")
    encoded_passagesL = []
    with torch.no_grad():
        for i in tqdm(range(0, len(passagesL), BATCH_SIZE)):
            batch_passages = passagesL[i:i+BATCH_SIZE]
            tokens = tokenizer(batch_passages, padding=True, truncation=True, max_length=200, return_tensors="pt").to(device)
            outputs = model.passage_encoder(**tokens)
            token_embs = outputs.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            embeddings = (token_embs * mask).sum(dim=1) / mask.sum(dim=1)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            encoded_passagesL.extend(embeddings.cpu())

    # === Store in ChromaDB ===
    client = PersistentClient(path=PERSIST_DIR)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(name=COLLECTION_NAME)
    
    collection = client.create_collection(
                                          name="ms_marco_passages_lora",
                                          metadata={"hnsw:space": "cosine"} #use cosine similarity metric for search 
                                          )

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
