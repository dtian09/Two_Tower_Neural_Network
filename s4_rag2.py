'''
RAG (Retrieval-Augmented Generation) pipeline using ChromaDB, TwoTowerBERTLoRA retriever, and a transformer generator, implemented with LangChain.
'''
import torch
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline

# You may need to adjust these paths and model names as needed
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "ms_marco_passages_lora"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # or your own
GENERATOR_MODEL_NAME = "facebook/bart-large-cnn"

def build_langchain_rag():
    # Load embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    # Load ChromaDB as a vectorstore
    vectordb = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings
    )
    # Set up the generator LLM using HuggingFace pipeline
    hf_pipe = pipeline("text2text-generation", model=GENERATOR_MODEL_NAME, device=0 if torch.cuda.is_available() else -1)
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    # Optionally, customize the prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Use the following context to answer the question.\nContext: {context}\nQuestion: {question}\nAnswer:"""
    )
    # Build the RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return rag_chain

def main():
    rag_chain = build_langchain_rag()
    print("LangChain RAG pipeline ready. Type your question (Ctrl+C to exit):")
    while True:
        try:
            query = input("\nEnter your question: ")
            result = rag_chain({"query": query})
            print("\nRetrieved Passages:")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"[{i}] {doc.page_content}")
            print("\nGenerated Answer:")
            print(result['result'])
        except KeyboardInterrupt:
            print("\nExiting LangChain RAG pipeline.")
            break

if __name__ == "__main__":
    main()
