from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PARSED_DATA_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"
INDEXES_DIR = DATA_DIR / "indexes"
FAISS_DIR = INDEXES_DIR / "faiss"
WHOOSH_DIR = INDEXES_DIR / "whoosh"
GRAPH_DIR = DATA_DIR / "graph"

PDF_FILE_NAME = "ncert_class12_physics_part1.pdf"
PDF_FILE_PATH = RAW_DATA_DIR / PDF_FILE_NAME
CHUNKS_FILE_PATH = CHUNKS_DIR / "chunks.json"
GRAPH_FILE_PATH = GRAPH_DIR / "knowledge_graph.json"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")