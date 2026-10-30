"""Hybrid retrieval: dense (embeddings) + sparse (BM25) -> RRF fusion -> cross-encoder rerank.

Every query is filtered by the caller's role level, so restricted documents never leave the index.
"""
import json

from app import config
from app.rag.textutils import rrf, tokenize


class HybridRetriever:
    def __init__(self):
        self.reload()

    def reload(self):
        self.chunks = json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))
        self.by_id = {c["id"]: c for c in self.chunks}
        self._bm25 = {}
        self._col = None

    def _collection(self):
        if self._col is None:
            import chromadb
            self._col = chromadb.PersistentClient(path=str(config.CHROMA_DIR)).get_collection("hr_policies")
        return self._col

    def _bm25_for(self, level: int):
        if level not in self._bm25:
            from rank_bm25 import BM25Okapi
            pool = [c for c in self.chunks if c["level"] <= level]
            self._bm25[level] = (pool, BM25Okapi([tokenize(c["heading"] + " " + c["text"]) for c in pool]))
        return self._bm25[level]

    def dense(self, query: str, level: int) -> list[str]:
        from app.rag.models import get_embedder
        vec = get_embedder().encode([query], normalize_embeddings=True).tolist()
        res = self._collection().query(query_embeddings=vec, n_results=config.TOP_K_DENSE,
                                       where={"level": {"$lte": level}})
        return res["ids"][0]

    def sparse(self, query: str, level: int) -> list[str]:
        pool, bm25 = self._bm25_for(level)
        scores = bm25.get_scores(tokenize(query))
        order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)[:config.TOP_K_BM25]
        return [pool[i]["id"] for i in order if scores[i] > 0]

    def search(self, query: str, level: int = 1, k: int | None = None) -> list[dict]:
        k = k or config.TOP_K_FINAL
        fused = rrf([self.dense(query, level), self.sparse(query, level)])[:config.TOP_K_FUSED]
        candidates = [self.by_id[i] for i in fused]
        if not candidates:
            return []
        from app.rag.models import get_reranker
        scores = get_reranker().predict([(query, c["heading"] + ". " + c["text"]) for c in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:k]
        return [{**c, "rerank_score": float(s)} for c, s in ranked]
