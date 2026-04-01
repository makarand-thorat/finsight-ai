import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from app.retriever import retrieve_context, format_context
from app.model_router import generate_answer

load_dotenv()

class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    model_used: str
    sources: list

def retrieve_node(state: RAGState) -> RAGState:
    print(f"Retrieving context for: {state['question']}")
    docs = retrieve_context(state["question"], k=4)
    context = format_context(docs)
    sources = [
        {
            "page": doc.metadata.get("page", "unknown"),
            "source": doc.metadata.get("source", "unknown"),
            "content_preview": doc.page_content[:200]
        }
        for doc in docs
    ]
    return {
        **state,
        "context": context,
        "sources": sources
    }

def generate_node(state: RAGState) -> RAGState:
    print(f"Generating answer using {state['model_used']}...")
    result = generate_answer(
        query=state["question"],
        context=state["context"],
        model_name=state["model_used"]
    )
    return {
        **state,
        "answer": result["answer"]
    }

def build_rag_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

def run_rag_pipeline(question: str, model_name: str = "gemini") -> dict:
    graph = build_rag_graph()

    initial_state = RAGState(
        question=question,
        context="",
        answer="",
        model_used=model_name,
        sources=[]
    )

    result = graph.invoke(initial_state)

    return {
        "question": result["question"],
        "answer": result["answer"],
        "model_used": result["model_used"],
        "sources": result["sources"],
        "context": result["context"]
    }