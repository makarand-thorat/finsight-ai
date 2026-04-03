import os
import re
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

RESULTS_DIR = Path("eval_results")
RESULTS_DIR.mkdir(exist_ok=True)


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
        max_output_tokens=512
    )


def extract_score(response_content) -> float:
    if isinstance(response_content, list):
        response_content = response_content[0].get("text", "0.0")
    match = re.search(r"-?[\d.]+", str(response_content))
    score = float(match.group()) if match else 0.0
    return round(min(1.0, max(0.0, score)), 3)


def evaluate_faithfulness(answer: str, context: str) -> float:
    llm = get_llm()
    prompt = f"""Rate the faithfulness of this answer on a scale of 0 to 1.
Faithfulness means the answer is grounded in the context and contains no made up information.

Context: {context[:1000]}
Answer: {answer}

Reply with ONLY a decimal number between 0 and 1. Example: 0.85"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return extract_score(response.content)


def evaluate_relevancy(question: str, answer: str) -> float:
    llm = get_llm()
    prompt = f"""Rate the relevancy of this answer on a scale of 0 to 1.
Relevancy means the answer directly addresses the question asked.

Question: {question}
Answer: {answer}

Reply with ONLY a decimal number between 0 and 1. Example: 0.85"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return extract_score(response.content)


def evaluate_response(
    question: str,
    answer: str,
    context: str,
    ground_truth: str = None
) -> dict:
    try:
        print("Running faithfulness evaluation...")
        faithfulness_score = evaluate_faithfulness(answer, context)
        print(f"Faithfulness: {faithfulness_score}")

        print("Running relevancy evaluation...")
        relevancy_score = evaluate_relevancy(question, answer)
        print(f"Relevancy: {relevancy_score}")

        scores = {
            "faithfulness": faithfulness_score,
            "answer_relevancy": relevancy_score,
            "average": round(
                (faithfulness_score + relevancy_score) / 2, 3
            )
        }

        return scores

    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
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
    }

    for r in results:
        for metric in total:
            total[metric] += r["scores"].get(metric, 0)

    count = len(results)
    return {
        metric: round(total[metric] / count, 3)
        for metric in total
    }