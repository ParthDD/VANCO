from backend.app.core.config import PDF_FILE_PATH
from backend.app.ingestion.pdf_loader import PDFLoader

loader = PDFLoader(PDF_FILE_PATH)

print("Checking PDF...")
print(loader.get_document_info())

print("\nPreview first page:\n")
page_1 = loader.extract_page_text(1)
print(page_1["text"][:2000])