'''
RAG (Retrieval-Augmented Generation) pipeline using LangChain with a custom retriever (ChromaDBQueryEngine) and a transformer generator.
'''
import torch
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from s3_query_database_bert import ChromaDBQueryEngine

class ChromaDBLangChainRetriever(BaseRetriever):
    def __init__(self, chroma_engine, search_strategy="ann"):
        self.chroma_engine = chroma_engine
        self.search_strategy = search_strategy

    def get_relevant_documents(self, query):
        passages = self.chroma_engine.retrieve(query, search_strategy=self.search_strategy)
        # Wrap passages as LangChain Documents
        return [Document(page_content=p) for p in passages]

    async def aget_relevant_documents(self, query):
        return self.get_relevant_documents(query)

class LangChainRAG:
    def __init__(self,
                 chroma_collection_name="ms_marco_passages_lora",
                 persist_dir="./chroma_db",
                 top_k=3,
                 query_length=20,
                 generator_model_name="facebook/bart-large-cnn",
                 device=None):
        self.chroma_engine = ChromaDBQueryEngine(
            chroma_collection_name=chroma_collection_name,
            top_k=top_k,
            persist_dir=persist_dir,
            query_length=query_length
        )
        self.retriever = ChromaDBLangChainRetriever(self.chroma_engine)
        hf_pipe = pipeline(
            "text2text-generation",
            model=generator_model_name,
            device=0 if (device or torch.cuda.is_available()) else -1
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipe)
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following context to answer the question.\nContext: {context}\nQuestion: {question}\nAnswer:"""
        )
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def answer(self, query):
        return self.rag_chain({"query": query})

    def interactive(self):
        print("LangChain RAG pipeline (custom retriever) ready. Type your question (Ctrl+C to exit):")
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
