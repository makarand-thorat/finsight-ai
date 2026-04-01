import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

SYSTEM_PROMPT = """You are a financial document analyst. 
You answer questions strictly based on the provided context from financial documents.
If the answer is not in the context, say "I cannot find this information in the provided document."
Never make up information. Always cite the page number when possible."""

def get_model(model_name: str = "gemini"):
    if model_name == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.1
        )
    else:
        raise ValueError(f"Model {model_name} not configured. Only gemini is supported.")

def generate_answer(query: str, context: str, model_name: str = "gemini") -> dict:
    model = get_model(model_name)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""Context from financial document:
{context}

Question: {query}

Answer based strictly on the context above:""")
    ]

    response = model.invoke(messages)

    return {
        "question": query,
        "answer": response.content,
        "model_used": model_name,
        "context_used": context
    }