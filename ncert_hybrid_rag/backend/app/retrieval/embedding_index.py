import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path


class EmbeddingIndexer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def load_chunks(self, chunks_path: Path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_index(self, chunks_path: Path, index_output_path: Path, metadata_output_path: Path):
        chunks = self.load_chunks(chunks_path)
        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        index_output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_output_path))

        with open(metadata_output_path, "wb") as f:
            pickle.dump(chunks, f)

        return {
            "total_chunks": len(chunks),
            "embedding_dimension": dimension,
            "faiss_index_path": str(index_output_path),
            "metadata_path": str(metadata_output_path)
        }

    def search(self, query: str, index_path: Path, metadata_path: Path, top_k: int = 5):
        index = faiss.read_index(str(index_path))

        with open(metadata_path, "rb") as f:
            chunks = pickle.load(f)

        query_embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = index.search(query_embedding, top_k)

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            chunk = chunks[idx]
            results.append({
                "rank": rank + 1,
                "score": float(distances[0][rank]),
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "text": chunk["text"]
            })

        return results