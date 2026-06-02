from fastapi import APIRouter, HTTPException, Query
from ..core.config import CHUNKS_FILE_PATH, FAISS_DIR, WHOOSH_DIR
from .embedding_index import EmbeddingIndexer
from .keyword_index import KeywordIndexer
from .hybrid_retrieval import hybrid_search, graph_search

router = APIRouter(prefix="/retrieve", tags=["Retrieval"])

FAISS_INDEX_PATH = FAISS_DIR / "chunks.index"
FAISS_METADATA_PATH = FAISS_DIR / "chunks_metadata.pkl"


@router.post("/build-vector-index")
def build_vector_index():
    try:
        indexer = EmbeddingIndexer()
        return indexer.build_index(CHUNKS_FILE_PATH, FAISS_INDEX_PATH, FAISS_METADATA_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/build-keyword-index")
def build_keyword_index():
    try:
        indexer = KeywordIndexer()
        return indexer.build_index(CHUNKS_FILE_PATH, WHOOSH_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/semantic-search")
def semantic_search(query: str = Query(...), top_k: int = 5):
    try:
        indexer = EmbeddingIndexer()
        return {
            "query": query,
            "results": indexer.search(query, FAISS_INDEX_PATH, FAISS_METADATA_PATH, top_k)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/keyword-search")
def keyword_search(query: str = Query(...), top_k: int = 5):
    try:
        indexer = KeywordIndexer()
        return {
            "query": query,
            "results": indexer.search(query, WHOOSH_DIR, top_k)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph-search")
def graph_search_route(query: str = Query(...)):
    try:
        return {
            "query": query,
            "results": graph_search(query)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hybrid-search")
def hybrid_search_route(query: str = Query(...), top_k: int = 5):
    try:
        return hybrid_search(query=query, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))