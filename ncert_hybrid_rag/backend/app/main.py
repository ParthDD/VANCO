from fastapi import FastAPI
from .api.ingest_routes import router as ingest_router
from .api.query_routes import router as query_router
from .retrieval.retrieval_routes import router as retrieval_router
from .graph.graph_routes import router as graph_router

app = FastAPI(title="NCERT Hybrid RAG API")

app.include_router(ingest_router)
app.include_router(retrieval_router)
app.include_router(graph_router)
app.include_router(query_router)


@app.get("/")
def root():
    return {"message": "NCERT Hybrid RAG backend is running"}


@app.get("/health")
def health():
    return {"status": "ok"}