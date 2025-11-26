from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


import os

def build_vectorstore():
    pdf_path = "references/comprehensive-clinical-nephrology.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="chroma_db"
    )
    vectordb.persist()
    print("✅ RAG index built and saved to chroma_db")

if __name__ == "__main__":
    build_vectorstore()