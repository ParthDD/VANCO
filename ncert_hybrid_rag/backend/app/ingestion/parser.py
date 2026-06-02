import fitz
import json
from pathlib import Path


class PDFParser:
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path

    def parse_pdf(self) -> list:
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at: {self.pdf_path}")

        parsed_pages = []

        with fitz.open(self.pdf_path) as doc:
            for page_index, page in enumerate(doc):
                blocks_data = page.get_text("dict")["blocks"]
                text_blocks = []

                for block in blocks_data:
                    if block.get("type") != 0:
                        continue

                    block_lines = []
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                        if line_text.strip():
                            block_lines.append(line_text.strip())

                    block_text = "\n".join(block_lines).strip()

                    if block_text:
                        text_blocks.append({
                            "block_number": block.get("number"),
                            "bbox": block.get("bbox"),
                            "text": block_text
                        })

                full_text = "\n\n".join([b["text"] for b in text_blocks]).strip()

                parsed_pages.append({
                    "page_number": page_index + 1,
                    "text": full_text,
                    "blocks": text_blocks
                })

        return parsed_pages

    def save_parsed_output(self, output_path: Path) -> Path:
        parsed_pages = self.parse_pdf()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed_pages, f, ensure_ascii=False, indent=2)

        return output_path