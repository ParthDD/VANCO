import fitz
from pathlib import Path


class PDFLoader:
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path

    def exists(self) -> bool:
        return self.pdf_path.exists()

    def get_document_info(self) -> dict:
        if not self.exists():
            raise FileNotFoundError(f"PDF not found at: {self.pdf_path}")

        with fitz.open(self.pdf_path) as doc:
            metadata = doc.metadata or {}
            return {
                "file_name": self.pdf_path.name,
                "file_path": str(self.pdf_path),
                "page_count": doc.page_count,
                "metadata": metadata,
            }

    def extract_page_text(self, page_number: int) -> dict:
        if not self.exists():
            raise FileNotFoundError(f"PDF not found at: {self.pdf_path}")

        with fitz.open(self.pdf_path) as doc:
            if page_number < 1 or page_number > doc.page_count:
                raise ValueError(f"Page number must be between 1 and {doc.page_count}")

            page = doc[page_number - 1]
            text = page.get_text(sort=True)

            return {
                "page_number": page_number,
                "text": text.strip(),
                "char_count": len(text.strip()),
            }

    def extract_page_range(self, start_page: int = 1, end_page: int = 3) -> list:
        if not self.exists():
            raise FileNotFoundError(f"PDF not found at: {self.pdf_path}")

        results = []

        with fitz.open(self.pdf_path) as doc:
            total_pages = doc.page_count
            start_page = max(1, start_page)
            end_page = min(end_page, total_pages)

            for page_num in range(start_page, end_page + 1):
                page = doc[page_num - 1]
                text = page.get_text(sort=True)

                results.append({
                    "page_number": page_num,
                    "text": text.strip(),
                    "char_count": len(text.strip()),
                })

        return results