from .embedding_index import EmbeddingIndexer
from .keyword_index import KeywordIndexer
from ..graph.graph_builder import GraphBuilder
from ..core.config import FAISS_DIR, WHOOSH_DIR

FAISS_INDEX_PATH = FAISS_DIR / "chunks.index"
FAISS_METADATA_PATH = FAISS_DIR / "chunks_metadata.pkl"


def normalize_semantic_scores(results):
    if not results:
        return results

    distances = [r["score"] for r in results]
    max_d = max(distances)
    min_d = min(distances)

    if max_d == min_d:
        for r in results:
            r["norm_score"] = 1.0
        return results

    for r in results:
        r["norm_score"] = 1 - ((r["score"] - min_d) / (max_d - min_d))
    return results


def normalize_keyword_scores(results):
    if not results:
        return results

    scores = [r["score"] for r in results]
    max_s = max(scores)
    min_s = min(scores)

    if max_s == min_s:
        for r in results:
            r["norm_score"] = 1.0
        return results

    for r in results:
        r["norm_score"] = (r["score"] - min_s) / (max_s - min_s)
    return results


def extract_graph_chunks(graph_result):
    connected_nodes = graph_result.get("connected_nodes", [])
    connected_edges = graph_result.get("connected_edges", [])

    chunk_nodes = {
        node["id"]: node
        for node in connected_nodes
        if node.get("type") == "Chunk"
    }

    page_lookup = {
        node["id"]: node.get("metadata", {}).get("page_number")
        for node in connected_nodes
        if node.get("type") == "Page"
    }

    graph_chunks = []

    for chunk_id, chunk_node in chunk_nodes.items():
        page_number = chunk_node.get("metadata", {}).get("page_number")
        text = chunk_node.get("metadata", {}).get("text", "")

        if page_number is None:
            for edge in connected_edges:
                if edge["source"] == chunk_id and edge["target"].startswith("page_"):
                    page_number = page_lookup.get(edge["target"])
                elif edge["target"] == chunk_id and edge["source"].startswith("page_"):
                    page_number = page_lookup.get(edge["source"])

        graph_chunks.append({
            "chunk_id": chunk_id.replace("chunk_", "", 1),
            "page_number": page_number,
            "text": text,
            "graph_score": 1.0
        })

    return graph_chunks


def reciprocal_rank_fusion(semantic_results, keyword_results, graph_results, k=60):
    fused = {}

    def ensure_item(key, item):
        if key not in fused:
            fused[key] = {
                "chunk_id": item["chunk_id"],
                "page_number": item.get("page_number"),
                "text": item.get("text", ""),
                "semantic_score": None,
                "semantic_norm_score": None,
                "keyword_score": None,
                "keyword_norm_score": None,
                "graph_score": None,
                "fusion_score": 0.0,
                "sources": []
            }

    for i, item in enumerate(semantic_results):
        key = item["chunk_id"]
        ensure_item(key, item)
        fused[key]["semantic_score"] = item["score"]
        fused[key]["semantic_norm_score"] = item.get("norm_score")
        fused[key]["fusion_score"] += 1 / (k + i + 1)
        fused[key]["sources"].append("semantic")

    for i, item in enumerate(keyword_results):
        key = item["chunk_id"]
        ensure_item(key, item)
        fused[key]["keyword_score"] = item["score"]
        fused[key]["keyword_norm_score"] = item.get("norm_score")
        fused[key]["fusion_score"] += 1 / (k + i + 1)
        fused[key]["sources"].append("keyword")

    for i, item in enumerate(graph_results):
        key = item["chunk_id"]
        ensure_item(key, item)
        fused[key]["graph_score"] = item.get("graph_score", 1.0)
        fused[key]["fusion_score"] += 1 / (k + i + 1)
        fused[key]["sources"].append("graph")

    merged = list(fused.values())
    merged.sort(key=lambda x: x["fusion_score"], reverse=True)

    for rank, item in enumerate(merged, start=1):
        item["rank"] = rank
        item["sources"] = sorted(list(set(item["sources"])))

    return merged


def graph_search(query: str):
    graph_builder = GraphBuilder()
    try:
        graph_result = graph_builder.query_graph(query)
        return extract_graph_chunks(graph_result)
    finally:
        graph_builder.close()


def hybrid_search(query: str, top_k: int = 5):
    emb = EmbeddingIndexer()
    kw = KeywordIndexer()

    semantic_results = emb.search(
        query=query,
        index_path=FAISS_INDEX_PATH,
        metadata_path=FAISS_METADATA_PATH,
        top_k=10
    )

    keyword_results = kw.search(
        query=query,
        index_dir=WHOOSH_DIR,
        top_k=10
    )

    graph_results = graph_search(query)

    semantic_results = normalize_semantic_scores(semantic_results)
    keyword_results = normalize_keyword_scores(keyword_results)

    fused_results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        graph_results
    )

    return {
        "query": query,
        "semantic_results": semantic_results,
        "keyword_results": keyword_results,
        "graph_results": graph_results,
        "hybrid_results": fused_results[:top_k]
    }