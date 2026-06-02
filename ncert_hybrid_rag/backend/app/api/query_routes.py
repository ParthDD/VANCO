from fastapi import APIRouter, HTTPException, Query
from ..rag.answer_generator import AnswerGenerator

router = APIRouter(prefix="/query", tags=["Query"])

generator = AnswerGenerator()


@router.get("/ask")
def ask_question(question: str = Query(...), top_k: int = 5):
    try:
        return generator.generate_answer(question=question, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))