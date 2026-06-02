import json
from pathlib import Path
from whoosh import index
from whoosh.fields import Schema, ID, TEXT, NUMERIC
from whoosh.qparser import MultifieldParser


class KeywordIndexer:
    def __init__(self):
        self.schema = Schema(
            chunk_id=ID(stored=True),
            page_number=NUMERIC(stored=True),
            text=TEXT(stored=True)
        )

    def load_chunks(self, chunks_path: Path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_index(self, chunks_path: Path, index_dir: Path):
        chunks = self.load_chunks(chunks_path)

        index_dir.mkdir(parents=True, exist_ok=True)

        if not index.exists_in(index_dir):
            ix = index.create_in(index_dir, self.schema)
        else:
            ix = index.open_dir(index_dir)

        writer = ix.writer()

        for chunk in chunks:
            writer.add_document(
                chunk_id=chunk["chunk_id"],
                page_number=chunk["page_number"],
                text=chunk["text"]
            )

        writer.commit()

        return {
            "total_chunks": len(chunks),
            "whoosh_index_dir": str(index_dir)
        }

    def search(self, query: str, index_dir: Path, top_k: int = 5):
        ix = index.open_dir(index_dir)

        with ix.searcher() as searcher:
            parser = MultifieldParser(["text"], schema=ix.schema)
            parsed_query = parser.parse(query)
            results = searcher.search(parsed_query, limit=top_k)

            output = []
            for rank, result in enumerate(results):
                output.append({
                    "rank": rank + 1,
                    "score": float(result.score),
                    "chunk_id": result["chunk_id"],
                    "page_number": result["page_number"],
                    "text": result["text"]
                })

        return output