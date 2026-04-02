import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

RESULTS_DIR = Path("eval_results")
RESULTS_DIR.mkdir(exists_ok =True)

def get_ragas_llm():
        return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1
    )

def get_ragas_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

def evaluate_response(
    question: str,
    answer: str,
    context: str,
    ground_truth: str = None
) -> dict:
    context_list = [c.strip() for c in context.split("\n\n") if c.strip()]

    data = {
        "question": [question],
        "answer" : [answer],
        "context" : [context_list],
        "ground_truth" : [ground_truth if ground_truth else answer]

    }
    dataset = Dataset.from_dict(data)
    llm=get_ragas_llm()
    embeddings = get_ragas_embeddings()
    metrics = [faithfulness,answer_relevancy,context_recall]

    result = evaluate(
        dataset = dataset,
        metrics = metrics,
        lm= llm,
        embeddings = embeddings
    )

    scores =  {
        "faithfulness" : round(float(result["faithfulness"]),3),
        "answer_relevancy" : round(float(result["answer_relevancy"]),3),
        "context_recall" : round(float(result["context_recall"]),3),
        "average" : round(
            (float(result["faithfulness"]) + 
            float(result["answer_relevancy"]) + 
            float(result["context_recall"])) / 3,3
        )
    }

    return scores

def save_eval_result(
    question: str,
    answer: str,
    model_used: str,
    scores: dict
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RESULTS_DIR / f"eval_{timestamp}.json"

    result = {
        "timestamp": timestamp,
        "model_used": model_used,
        "question": question,
        "answer": answer,
        "scores": scores
    }

    with open(filename, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Eval result saved to {filename}")
    return str(filename)

def load_all_results() -> list:
    results = []
    for file in sorted(RESULTS_DIR.glob("*.json")):
        with open(file, "r") as f:
            results.append(json.load(f))
    return results


def get_average_scores() -> dict:
    results = load_all_results()

    if not results:
        return {}

    total = {
        "faithfulness": 0,
        "answer_relevancy": 0,
        "context_recall": 0
    }

    for r in results:
        for metric in total:
            total[metric] += r["scores"].get(metric, 0)

    count = len(results)
    return {
        metric: round(total[metric] / count, 3)
        for metric in total
    }