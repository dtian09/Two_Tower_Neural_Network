# Two-Tower (Dual-Encoder) Retrieval System

This repository provides a full pipeline for building a retrieval-augmented system using a two-tower (dual-encoder) neural network architecture, trained and evaluated on the MS MARCO dataset. The system leverages ChromaDB for efficient vector storage and retrieval, and supports both classic and modern (LangChain-based) RAG pipelines.

## Features
- **Train** a two-tower BERT-based neural network with LoRA on MS MARCO.
- **Encode** and store passage embeddings in a persistent ChromaDB vector database.
- **Query** the database for relevant passages using a trained query encoder.
- **RAG** (Retrieval-Augmented Generation) pipelines for question answering, including a LangChain-based implementation.

## Workflow
1. **Train the Dual-Encoder Model**
   - Run `s1_train_tnn_bert.py` to train the two-tower BERT model with LoRA on MS MARCO.
2. **Build the Passage Database**
   - Run `s2_store_passages_in_database_bert.py` to encode MS MARCO passages and store them in a ChromaDB vector database.
3. **Query the Database**
   - Use `s3_query_database_bert.py` to interactively query the ChromaDB database for relevant passages.
4. **RAG Pipelines**
   - `s4_rag.py`: A RAG pipeline using the ChromaDB database and a transformer.
   - `s4_rag2.py`: A RAG pipeline using LangChain, ChromaDB database, and a transformer.

## Setup
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # For LangChain RAG (s4_rag2.py):
   pip install langchain langchain-community chromadb transformers sentence-transformers
   ```
2. **(Optional) Set up Weights & Biases**
   - For experiment tracking, set your `WANDB_API_KEY` as an environment variable.

## Usage
- **Train the model:**
  ```bash
  python s1_train_tnn_bert.py
  ```
- **Build the passage database:**
  ```bash
  python s2_store_passages_in_database_bert.py
  ```
- **Query the database:**
  ```bash
  python s3_query_database_bert.py
  ```
- **Run a RAG pipeline:**
  ```bash
  python s4_rag.py
  # or
  python s4_rag2.py
  ```

## Notes
- The default models and paths can be changed in the scripts as needed.
- Ensure you have enough disk space for ChromaDB and MS MARCO data.

## References
- [MS MARCO Dataset](https://microsoft.github.io/msmarco/)
- [ChromaDB](https://www.trychroma.com/)
- [LangChain](https://python.langchain.com/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)


