import pytest
from app.evaluator import evaluate_faithfulness, evaluate_relevancy, extract_score


def test_extract_score_from_string():
    assert extract_score("0.85") == 0.85


def test_extract_score_from_list():
    assert extract_score([{"text": "0.9"}]) == 0.9


def test_extract_score_clamped_high():
    assert extract_score("1.5") == 1.0


def test_extract_score_clamped_low():
    assert extract_score("-0.5") == 0.0


def test_extract_score_no_number():
    assert extract_score("no number here") == 0.0


def test_faithfulness_returns_float():
    score = evaluate_faithfulness(
        answer="Total revenue was 5.2 billion euros.",
        context="FinCorp reported total revenue of 5.2 billion euros in 2024."
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_relevancy_returns_float():
    score = evaluate_relevancy(
        question="What was the total revenue?",
        answer="Total revenue was 5.2 billion euros."
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0