from fastapi import APIRouter, HTTPException
from ..core.config import PDF_FILE_PATH
from ..ingestion.pdf_loader import PDFLoader
from ..ingestion.ingest_pipeline import run_ingestion_pipeline

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.get("/pdf-info")
def get_pdf_info():
    try:
        loader = PDFLoader(PDF_FILE_PATH)
        return loader.get_document_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/page/{page_number}")
def get_page_text(page_number: int):
    try:
        loader = PDFLoader(PDF_FILE_PATH)
        return loader.extract_page_text(page_number)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preview")
def preview_pages(start_page: int = 1, end_page: int = 3):
    try:
        loader = PDFLoader(PDF_FILE_PATH)
        return {"pages": loader.extract_page_range(start_page, end_page)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-pipeline")
def run_pipeline():
    try:
        return run_ingestion_pipeline()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))