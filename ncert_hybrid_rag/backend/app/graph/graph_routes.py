from fastapi import APIRouter, HTTPException, Query
from .graph_builder import GraphBuilder

router = APIRouter(prefix="/graph", tags=["Graph"])

graph_builder = GraphBuilder()


@router.post("/build")
def build_graph():
    try:
        return graph_builder.build_and_save()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query")
def query_graph(concept: str = Query(...)):
    try:
        return graph_builder.query_graph(concept)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))