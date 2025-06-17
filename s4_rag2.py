'''
RAG (Retrieval-Augmented Generation) pipeline using ChromaDB, TwoTowerBERTLoRA retriever, and a transformer generator, implemented with LangChain (OOP version).
'''
import torch
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline

CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "ms_marco_passages_lora"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL_NAME = "facebook/bart-large-cnn"

class LangChainRAG:
    def __init__(self,
                 chroma_persist_dir=CHROMA_PERSIST_DIR,
                 chroma_collection_name=CHROMA_COLLECTION_NAME,
                 embedding_model_name=EMBEDDING_MODEL_NAME,
                 generator_model_name=GENERATOR_MODEL_NAME,
                 top_k=3):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vectordb = Chroma(
            persist_directory=chroma_persist_dir,
            collection_name=chroma_collection_name,
            embedding_function=self.embeddings
        )
        hf_pipe = pipeline(
            "text2text-generation",
            model=generator_model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipe)
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following context to answer the question.\nContext: {context}\nQuestion: {question}\nAnswer:"""
        )
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(search_kwargs={"k": top_k}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def answer(self, query):
        return self.rag_chain({"query": query})

    def interactive(self):
        print("LangChain RAG pipeline ready. Type your question (Ctrl+C to exit):")
        while True:
            try:
                query = input("\nEnter your question: ")
                result = self.answer(query)
                print("\nRetrieved Passages:")
                for i, doc in enumerate(result['source_documents'], 1):
                    print(f"[{i}] {doc.page_content}")
                print("\nGenerated Answer:")
                print(result['result'])
            except KeyboardInterrupt:
                print("\nExiting LangChain RAG pipeline.")
                break

def main():
    rag = LangChainRAG()
    rag.interactive()

if __name__ == "__main__":
    main()
