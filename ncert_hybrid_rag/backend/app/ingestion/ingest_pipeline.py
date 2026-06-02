from pathlib import Path
from .parser import PDFParser
from .chunker import Chunker
from ..core.config import PDF_FILE_PATH, PARSED_DATA_DIR


CHUNKS_DIR = PARSED_DATA_DIR.parent / "chunks"
PARSED_JSON_PATH = PARSED_DATA_DIR / "parsed_pages.json"
CHUNKS_JSON_PATH = CHUNKS_DIR / "chunks.json"


def run_ingestion_pipeline():
    parser = PDFParser(PDF_FILE_PATH)
    parsed_pages = parser.parse_pdf()
    parser.save_parsed_output(PARSED_JSON_PATH)

    chunker = Chunker(min_chars=400, max_chars=1200)
    chunks = chunker.chunk_parsed_pages(parsed_pages)
    chunker.save_chunks(chunks, CHUNKS_JSON_PATH)

    return {
        "parsed_pages_path": str(PARSED_JSON_PATH),
        "chunks_path": str(CHUNKS_JSON_PATH),
        "total_pages": len(parsed_pages),
        "total_chunks": len(chunks)
    }