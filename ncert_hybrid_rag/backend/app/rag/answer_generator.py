import requests
from ..core.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from ..retrieval.hybrid_retrieval import hybrid_search


class AnswerGenerator:
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL.rstrip("/")
        self.model = OLLAMA_MODEL

    def build_context(self, hybrid_result: dict, max_chunks: int = 5):
        hybrid_chunks = hybrid_result.get("hybrid_results", [])[:max_chunks]

        context_blocks = []
        citations = []

        for idx, chunk in enumerate(hybrid_chunks, start=1):
            page_number = chunk.get("page_number")
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text", "").strip()
            sources = ", ".join(chunk.get("sources", []))

            context_blocks.append(
                f"[Source {idx}] Page: {page_number}, Chunk ID: {chunk_id}, Retrieval Sources: {sources}\n{text}"
            )

            citations.append({
                "source_number": idx,
                "page_number": page_number,
                "chunk_id": chunk_id,
                "sources": chunk.get("sources", [])
            })

        return "\n\n".join(context_blocks), citations

    def call_ollama(self, prompt: str):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    def generate_answer(self, question: str, top_k: int = 5):
        retrieval_output = hybrid_search(question, top_k=top_k)
        context_text, citations = self.build_context(retrieval_output, max_chunks=top_k)

        if not context_text.strip():
            return {
                "question": question,
                "answer": "This information is not available in the source document.",
                "citations": [],
                "retrieval": retrieval_output
            }

        prompt = f"""
You are a grounded NCERT Physics assistant.

Answer ONLY from the provided retrieved evidence.
Do not use outside knowledge.
If the answer is not supported by the evidence, say exactly:
"This information is not available in the source document."

When possible, mention the source page numbers in the answer.

Question:
{question}

Retrieved Evidence:
{context_text}

Instructions:
1. Answer only using the retrieved evidence.
2. Keep the answer concise and accurate.
3. If supported, include page references like (Page 22).
4. If the evidence is insufficient, say exactly: This information is not available in the source document.
"""

        try:
            answer_text = self.call_ollama(prompt)
            if not answer_text:
                answer_text = "This information is not available in the source document."
        except Exception:
            answer_text = "This information is not available in the source document."

        return {
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "retrieval": retrieval_output
        }