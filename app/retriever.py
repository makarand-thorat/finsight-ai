import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = Path("chroma_db")

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def get_vectorstore(collection_name: str = "finsight") -> Chroma:
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=collection_name
    )
    return vectorstore


def retrieve_context(query: str, k: int = 4) -> list:
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return results

def format_context(docs: list) -> str:
    context = "\n\n".join([
        f"Source: Page {doc.metadata.get('page', 'unknown')}\n{doc.page_content}"
        for doc in docs
    ])
    return context