import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import langchain
langchain.debug = True
load_dotenv()

RESULTS_DIR = Path("eval_results")
RESULTS_DIR.mkdir(exist_ok=True)


def get_ragas_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
        max_output_tokens=8192
    )
    return LangchainLLMWrapper(llm)


def get_ragas_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    return LangchainEmbeddingsWrapper(embeddings)

def _extract_score(value) -> float:
    """Safely extract a float score from ragas result (may be list or float)."""
    if isinstance(value, list):
        valid = [v for v in value if v is not None and not (isinstance(v, float) and v != v)]  # filter NaN
        return float(valid[0]) if valid else 0.0
    return float(value) if value is not None else 0.0

def evaluate_response(
    question: str,
    answer: str,
    context: str,
    ground_truth: str = None
) -> dict:
    try:
        context_list = [c.strip() for c in context.split("\n\n") if c.strip()]

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [context_list],
            "ground_truth": [ground_truth if ground_truth else answer]
        }

        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_recall],
            llm=get_ragas_llm(),
            embeddings=get_ragas_embeddings()
        )

        scores = {
            "faithfulness": round(_extract_score(result["faithfulness"]), 3),
            "answer_relevancy": round(_extract_score(result["answer_relevancy"]), 3),
            "context_recall": round(_extract_score(result["context_recall"]), 3),
        }
        scores["average"] = round(
        sum(scores[k] for k in ["faithfulness", "answer_relevancy", "context_recall"]) / 3, 3)

        return scores

    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "average": 0.0,
            "error": str(e)
        }


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