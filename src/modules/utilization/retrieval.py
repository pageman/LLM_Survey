"""Retrieval modules extracted from DPR/BM25 donor notebooks."""

from __future__ import annotations

from collections import Counter

import numpy as np

from src.core import compute_retrieval_metrics, hashed_bow_embedding, softmax


def contrastive_loss(query_emb: np.ndarray, positive_emb: np.ndarray, negative_embs: list[np.ndarray]) -> float:
    pos_score = float(np.dot(query_emb, positive_emb))
    neg_scores = [float(np.dot(query_emb, neg_emb)) for neg_emb in negative_embs]
    all_scores = np.array([pos_score] + neg_scores, dtype=float)
    probs = softmax(all_scores, axis=-1)
    return float(-np.log(probs[0] + 1e-8))


class DenseRetriever:
    """Deterministic dense retriever with hashed embeddings."""

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim

    def encode_text(self, text: str) -> np.ndarray:
        return hashed_bow_embedding(text, self.embedding_dim)

    def encode_corpus(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.encode_text(text) for text in texts])

    def retrieve(self, query: str, documents: list[str], k: int = 3) -> tuple[np.ndarray, np.ndarray]:
        query_embedding = self.encode_text(query)
        document_embeddings = self.encode_corpus(documents)
        similarities = document_embeddings @ query_embedding
        top_indices = np.argsort(similarities)[::-1][:k]
        top_scores = similarities[top_indices]
        return top_indices, top_scores


class SimpleBM25Retriever:
    """Simplified BM25 retriever adapted from the donor notebook."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_freqs: dict[str, int] = {}

        all_words = []
        for document in documents:
            words = set(document.lower().split())
            all_words.extend(document.lower().split())
            for word in words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        self.avg_doc_len = len(all_words) / max(len(documents), 1)
        self.N = len(documents)

    def score(self, query: str, doc_index: int) -> float:
        query_words = query.lower().split()
        document_words = self.documents[doc_index].lower().split()
        document_len = len(document_words)
        tf = Counter(document_words)

        score = 0.0
        for word in query_words:
            if word not in tf:
                continue
            df = self.doc_freqs.get(word, 0)
            idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            freq = tf[word]
            norm = 1.0 - self.b + self.b * (document_len / max(self.avg_doc_len, 1e-8))
            tf_component = (freq * (self.k1 + 1.0)) / (freq + self.k1 * norm)
            score += float(idf * tf_component)
        return score

    def retrieve(self, query: str, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
        scores = np.array([self.score(query, idx) for idx in range(len(self.documents))], dtype=float)
        top_indices = np.argsort(scores)[::-1][:k]
        return top_indices, scores[top_indices]


class HybridRetriever:
    """Combine dense and BM25 scores with simple normalization."""

    def __init__(self, documents: list[str], embedding_dim: int = 64, alpha: float = 0.5):
        self.documents = documents
        self.alpha = alpha
        self.dense = DenseRetriever(embedding_dim=embedding_dim)
        self.bm25 = SimpleBM25Retriever(documents)

    def retrieve(self, query: str, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
        dense_indices, dense_scores = self.dense.retrieve(query, self.documents, k=len(self.documents))
        bm25_indices, bm25_scores = self.bm25.retrieve(query, k=len(self.documents))

        dense_full = np.zeros((len(self.documents),), dtype=float)
        bm25_full = np.zeros((len(self.documents),), dtype=float)
        dense_full[dense_indices] = dense_scores
        bm25_full[bm25_indices] = bm25_scores

        dense_norm = dense_full / (np.linalg.norm(dense_full) + 1e-8)
        bm25_norm = bm25_full / (np.linalg.norm(bm25_full) + 1e-8)
        combined = self.alpha * dense_norm + (1.0 - self.alpha) * bm25_norm

        top_indices = np.argsort(combined)[::-1][:k]
        return top_indices, combined[top_indices]

    @staticmethod
    def evaluate(
        predictions: list[list[int] | np.ndarray],
        correct_indices: list[int],
        k_values: list[int] | None = None,
    ) -> tuple[dict[int, float], float]:
        return compute_retrieval_metrics(predictions, correct_indices, k_values=k_values)
