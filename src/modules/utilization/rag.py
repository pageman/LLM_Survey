"""Minimal RAG pipeline built from local retrieval and generation components."""

from __future__ import annotations

import numpy as np

from src.core import ToyTokenizer, softmax
from src.modules.pretraining.causal_lm import CausalLanguageModel
from src.modules.utilization.retrieval import DenseRetriever


class SimpleRAGSequenceGenerator:
    """Very small retrieval-conditioned scoring/generation adapter."""

    def __init__(self, tokenizer: ToyTokenizer, seed: int = 0):
        self.tokenizer = tokenizer
        self.lm = CausalLanguageModel(tokenizer=tokenizer, seed=seed)

    def score_document(self, query: str, document: str, target: str) -> float:
        combined = f"{query} {document} {target}"
        return -self.lm.score_text(combined)["mean_loss"]

    def generate(self, query: str, document: str, max_new_tokens: int = 4) -> str:
        prompt = f"{query} {document}"
        return self.lm.generate(prompt, max_new_tokens=max_new_tokens)


class SimpleRAGPipeline:
    """RAG-sequence style pipeline with dense retrieval and document marginalization."""

    def __init__(self, documents: list[str], tokenizer: ToyTokenizer, embedding_dim: int = 64, seed: int = 0):
        self.documents = documents
        self.retriever = DenseRetriever(embedding_dim=embedding_dim)
        self.generator = SimpleRAGSequenceGenerator(tokenizer=tokenizer, seed=seed)

    def retrieve(self, query: str, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
        indices, scores = self.retriever.retrieve(query, self.documents, k=k)
        return indices, softmax(scores, axis=-1)

    def rag_sequence_score(self, query: str, target: str, k: int = 3) -> dict[str, object]:
        indices, retrieval_probs = self.retrieve(query, k=k)
        doc_scores = []
        for idx in indices:
            score = self.generator.score_document(query, self.documents[int(idx)], target)
            doc_scores.append(score)

        doc_scores_array = np.array(doc_scores, dtype=float)
        generation_probs = softmax(doc_scores_array, axis=-1)
        marginal = float(np.sum(retrieval_probs * generation_probs))
        return {
            "doc_indices": indices.tolist(),
            "retrieval_probs": retrieval_probs.tolist(),
            "generation_probs": generation_probs.tolist(),
            "rag_sequence_score": marginal,
        }

    def answer(self, query: str, k: int = 1, max_new_tokens: int = 4) -> dict[str, object]:
        indices, retrieval_probs = self.retrieve(query, k=k)
        best_index = int(indices[0])
        generated = self.generator.generate(query, self.documents[best_index], max_new_tokens=max_new_tokens)
        return {
            "query": query,
            "selected_doc_index": best_index,
            "selected_doc": self.documents[best_index],
            "retrieval_probs": retrieval_probs.tolist(),
            "generated_text": generated,
        }
