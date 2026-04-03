"""Local RAG demo.

Usage:
    python3 experiments/run_rag_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import ToyTokenizer, build_report, write_report
from src.modules.utilization import SimpleRAGPipeline


def main() -> None:
    output_dir = Path("artifacts/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = [
        "paris is the capital of france",
        "tokyo is the capital of japan",
        "manila is the capital of the philippines",
    ]
    texts_for_tokenizer = documents + [
        "what is the capital of france",
        "capital of japan",
        "capital of the philippines",
    ]
    tokenizer = ToyTokenizer.from_texts(texts_for_tokenizer)
    rag = SimpleRAGPipeline(documents=documents, tokenizer=tokenizer, embedding_dim=64, seed=0)

    query = "what is the capital of france"
    target = "paris is the capital"
    score = rag.rag_sequence_score(query, target, k=2)
    answer = rag.answer(query, k=1, max_new_tokens=3)

    result = build_report(
        experiment_id="rag_demo",
        module="utilization.rag",
        metrics={"rag_sequence_score": score["rag_sequence_score"]},
        artifacts={"score": score, "answer": answer},
        notes=["Minimal retrieval-augmented generation sequence-marginalization demo."],
    )
    write_report(result, output_dir / "rag_demo.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
