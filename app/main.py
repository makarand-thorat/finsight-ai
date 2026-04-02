import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from app.ingestion import ingest_pdf
from app.rag_pipeline import run_rag_pipeline
from app.evaluator import evaluate_response, save_eval_result, load_all_results, get_average_scores

load_dotenv()

app = FastAPI(
    title="FinSight AI",
    description="Production RAG evaluation platform for financial documents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    model_name: str = "gemini"
    evaluate: bool = True

class QuestionResponse(BaseModel):
    question: str
    answer: str
    model_used: str
    sources: list
    scores: dict = {}

@app.get("/")
def root():
    return {
        "app": "FinSight AI",
        "status": "running",
        "version": "1.0.0"
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    try:
        file_bytes = await file.read()
        result = ingest_pdf(file_bytes, file.filename)
        return {
            "message": f"Successfully ingested {file.filename}",
            "chunks_created": result["chunks"],
            "status": result["status"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        rag_result = run_rag_pipeline(
            question=request.question,
            model_name=request.model_name
        )

        scores = {}
        if request.evaluate:
            scores = evaluate_response(
                question=request.question,
                answer=rag_result["answer"],
                context=rag_result["context"]
            )
            save_eval_result(
                question=request.question,
                answer=rag_result["answer"],
                model_used=request.model_name,
                scores=scores
            )

        return QuestionResponse(
            question=rag_result["question"],
            answer=rag_result["answer"],
            model_used=rag_result["model_used"],
            sources=rag_result["sources"],
            scores=scores
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
def get_results():
    results = load_all_results()
    return {
        "total_evaluations": len(results),
        "results": results
    }

@app.get("/scores")
def get_scores():
    scores = get_average_scores()
    return {
        "average_scores": scores,
        "message": "Scores averaged across all evaluations"
    }