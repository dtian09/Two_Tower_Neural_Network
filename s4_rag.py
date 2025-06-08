'''
RAG (Retrieval-Augmented Generation) pipeline using ChromaDB, TwoTowerBERTLoRA retriever, and a transformer generator.
'''
import torch
from transformers import BertTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from s3_query_database_bert import ChromaDBQueryEngine

default_generator_model = "facebook/bart-large-cnn"  # You can change to any seq2seq model

class RAGPipeline:
    def __init__(self, 
                 chroma_collection_name="ms_marco_passages_lora",
                 persist_dir="./chroma_db",
                 top_k=3,
                 query_length=20,
                 generator_model_name=default_generator_model,
                 device=None):
        self.retriever = ChromaDBQueryEngine(
            chroma_collection_name=chroma_collection_name,
            top_k=top_k,
            persist_dir=persist_dir,
            query_length=query_length
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name).to(self.device)

    def retrieve(self, query, search_strategy="ann"):
        query_embedding = self.retriever.encode_query(query)
        if search_strategy == "ann":
            results = self.retriever.ann_search(query_embedding)
            passages = [doc for doc, _ in results]
        else:
            results = self.retriever.exact_search(query_embedding)
            passages = [doc for doc, _, _ in results]
        return passages

    def generate(self, query, passages, max_length=128):
        # Concatenate retrieved passages and query for generation
        context = "\n".join(passages)
        input_text = f"question: {query} context: {context}"
        inputs = self.generator_tokenizer([input_text], return_tensors="pt", truncation=True, padding=True).to(self.device)
        summary_ids = self.generator.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        return self.generator_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def rag(self, query, search_strategy="ann", max_length=128):
        passages = self.retrieve(query, search_strategy=search_strategy)
        answer = self.generate(query, passages, max_length=max_length)
        return answer, passages


def main():
    rag = RAGPipeline()
    print("RAG pipeline ready. Type your question (Ctrl+C to exit):")
    while True:
        try:
            query = input("\nEnter your question: ")
            answer, passages = rag.rag(query)
            print("\nRetrieved Passages:")
            for i, p in enumerate(passages, 1):
                print(f"[{i}] {p}")
            print("\nGenerated Answer:")
            print(answer)
        except KeyboardInterrupt:
            print("\nExiting RAG pipeline.")
            break

if __name__ == "__main__":
    main()

