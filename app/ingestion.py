import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

UPLOAD_DIR = Path("uploads")
CHROMA_DIR = Path("chroma_db")
UPLOAD_DIR.mkdir(exist_ok=True)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


def save_uploaded_file(file_bytes: bytes, filename: str) -> Path:
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    return file_path

def load_and_chunk_pdf(file_path: Path) -> list:
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap = 200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def store_in_chroma(chunks: list, collection_name:str = "finsight") -> Chroma:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory = str(CHROMA_DIR),
        collection_name = collection_name
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore

def ingest_pdf(file_bytes: bytes, filename: str) -> dict:
    file_path = save_uploaded_file(file_bytes, filename)
    chunks = load_and_chunk_pdf(file_path)
    store_in_chroma(chunks)
    return{
        "filename" : filename,
        "chunks" : len(chunks),
        "status" : "success"
    }


