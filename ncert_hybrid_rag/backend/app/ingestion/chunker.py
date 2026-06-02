import json
from pathlib import Path


class Chunker:
    def __init__(self, min_chars: int = 400, max_chars: int = 1200):
        self.min_chars = min_chars
        self.max_chars = max_chars

    def chunk_parsed_pages(self, parsed_pages: list) -> list:
        chunks = []

        for page in parsed_pages:
            page_number = page["page_number"]
            blocks = page["blocks"]

            current_text = []
            current_len = 0
            chunk_id = 1

            for block in blocks:
                block_text = block["text"].strip()
                if not block_text:
                    continue

                if current_len + len(block_text) > self.max_chars and current_text:
                    combined_text = "\n\n".join(current_text).strip()
                    chunks.append({
                        "chunk_id": f"page_{page_number}_chunk_{chunk_id}",
                        "page_number": page_number,
                        "text": combined_text,
                        "char_count": len(combined_text)
                    })
                    chunk_id += 1
                    current_text = [block_text]
                    current_len = len(block_text)
                else:
                    current_text.append(block_text)
                    current_len += len(block_text)

            if current_text:
                combined_text = "\n\n".join(current_text).strip()
                chunks.append({
                    "chunk_id": f"page_{page_number}_chunk_{chunk_id}",
                    "page_number": page_number,
                    "text": combined_text,
                    "char_count": len(combined_text)
                })

        return chunks

    def save_chunks(self, chunks: list, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        return output_path