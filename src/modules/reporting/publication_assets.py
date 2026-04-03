"""Generate publication-facing Markdown, CSV, and SVG artifacts from local reports."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from src.modules.evaluation.benchmark_harness import BenchmarkHarness
from src.modules.evaluation.docs_summary import IMPLEMENTATION_TARGETS
from src.modules.evaluation.report_index import ReportIndex
from src.modules.reporting.fidelity_band_dashboard import FidelityBandDashboard
from src.modules.reporting.module_provenance_dashboard import ModuleProvenanceDashboard
from src.modules.reporting.paper_section_dashboard import PaperSectionDashboard


SURVEY_PDF = "/Users/hifi/Downloads/2303.18223v19.pdf"
DONOR_ROOT = "/Users/hifi/sutskever-30-implementations"
DONOR_MANIFEST = "/Users/hifi/Downloads/LLM_Survey/third_party/sutskever_30_manifest.md"
SURVEY_MAP_EVIDENCE = {
    "resources.public_model_registry": {
        "evidence_file": SURVEY_PDF,
        "donor_origin": "repo-authored resource synthesis",
        "supporting_file": DONOR_MANIFEST,
        "extraction_notes": "Open-weight public families summarized against survey scope and local donor manifest.",
    },
    "resources.closed_model_registry": {
        "evidence_file": SURVEY_PDF,
        "donor_origin": "repo-authored resource synthesis",
        "supporting_file": DONOR_MANIFEST,
        "extraction_notes": "Closed/API-only families captured as survey-facing resource metadata rather than runnable implementations.",
    },
    "resources.corpus_profile_demo": {
        "evidence_file": SURVEY_PDF,
        "donor_origin": f"donor context: {DONOR_ROOT}/README.md",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/docs/full_survey_gap_audit.md",
        "extraction_notes": "Domain proportions are repo-authored but anchored to survey data-mixture coverage themes.",
    },
    "resources.library_stack_matrix": {
        "evidence_file": f"{DONOR_ROOT}/README.md",
        "donor_origin": f"donor manifest: {DONOR_MANIFEST}",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/docs/reuse_audit.md",
        "extraction_notes": "Library capability rows summarize the repo and donor implementation stack rather than external package benchmarking.",
    },
    "resources.framework_stack_matrix": {
        "evidence_file": f"{DONOR_ROOT}/TRAINING_UTILS_README.md",
        "donor_origin": f"donor manifest: {DONOR_MANIFEST}",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/docs/reuse_audit.md",
        "extraction_notes": "Framework roles reflect local implementation and training-utility coverage, not exhaustive ecosystem surveys.",
    },
    "resources.dataset_license_audit": {
        "evidence_file": SURVEY_PDF,
        "donor_origin": "repo-authored licensing summary",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/docs/limitations.md",
        "extraction_notes": "License audit is an educational approximation meant to expose redistribution risk categories.",
    },
    "resources.model_release_timeline": {
        "evidence_file": SURVEY_PDF,
        "donor_origin": "repo-authored release timeline",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/docs/release_notes.md",
        "extraction_notes": "Timeline is a compressed capability chronology for survey orientation rather than a historical database.",
    },
    "reporting.paper_section_dashboard": {
        "evidence_file": "/Users/hifi/Downloads/LLM_Survey/src/modules/evaluation/docs_summary.py",
        "donor_origin": "generated from tracked implementation targets",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/docs/scoreboard.md",
        "extraction_notes": "Section completion is computed directly from canonical local reports and target prefixes.",
    },
    "reporting.module_provenance_dashboard": {
        "evidence_file": "/Users/hifi/Downloads/LLM_Survey/src/modules/evaluation/paper_scope_completion.py",
        "donor_origin": "generated from dedicated-vs-generated module ledger",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/module_provenance_dashboard_demo.json",
        "extraction_notes": "Dedicated provenance comes from the maintained baseline implementation ledger.",
    },
    "reporting.fidelity_band_dashboard": {
        "evidence_file": "/Users/hifi/Downloads/LLM_Survey/src/modules/reporting/fidelity_band_dashboard.py",
        "donor_origin": "repo-authored fidelity classification",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/docs/fidelity_matrix.md",
        "extraction_notes": "Fidelity bands classify modules as mechanism-level or survey-map for publication transparency.",
    },
    "benchmark.cross_section_summary": {
        "evidence_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/benchmark_harness_demo.json",
        "donor_origin": "generated from canonical benchmark slices",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/docs/figures_and_tables.md",
        "extraction_notes": "Cross-section summary aggregates representative canonical demos across major survey areas.",
    },
    "benchmark.risk_bundle_summary": {
        "evidence_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/benchmark_harness_demo.json",
        "donor_origin": "generated risk bundle",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/risk_bundle_summary_demo.json",
        "extraction_notes": "Risk bundle combines safety, bias, privacy, truthfulness, and hallucination-facing artifacts.",
    },
    "benchmark.adaptation_bundle_summary": {
        "evidence_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/adaptation_summary_demo.json",
        "donor_origin": "generated adaptation bundle",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/adaptation_bundle_summary_demo.json",
        "extraction_notes": "Adaptation bundle compares canonical local adaptation demos under a common gain/loss interface.",
    },
    "benchmark.utilization_bundle_summary": {
        "evidence_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/benchmark_harness_demo.json",
        "donor_origin": "generated utilization bundle",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/utilization_bundle_summary_demo.json",
        "extraction_notes": "Utilization bundle summarizes retrieval, reasoning, and tool-use slices from canonical reports.",
    },
}
MECHANISM_EVIDENCE = {
    "foundations.rnn_lm": {
        "evidence_file": f"{DONOR_ROOT}/02_char_rnn_karpathy.ipynb",
        "donor_origin": f"donor notebook: {DONOR_ROOT}/02_char_rnn_karpathy.ipynb",
        "supporting_file": f"{DONOR_ROOT}/README.md",
        "extraction_notes": "Character-level recurrence and sampling flow were normalized into importable local modules and smoke-tested locally.",
    },
    "foundations.lstm_lm": {
        "evidence_file": f"{DONOR_ROOT}/03_lstm_understanding.ipynb",
        "donor_origin": f"donor notebook: {DONOR_ROOT}/03_lstm_understanding.ipynb",
        "supporting_file": DONOR_MANIFEST,
        "extraction_notes": "Gate structure and sequence-state transitions were extracted into a lightweight local LSTM language-model demo.",
    },
    "foundations.seq2seq_basics": {
        "evidence_file": f"{DONOR_ROOT}/14_bahdanau_attention.ipynb",
        "donor_origin": f"donor notebook: {DONOR_ROOT}/14_bahdanau_attention.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/docs/reuse_audit.md",
        "extraction_notes": "Encoder-decoder toy behavior was reduced to a deterministic educational sequence-reversal baseline aligned with the survey architecture discussion.",
    },
    "foundations.transformer_basics": {
        "evidence_file": f"{DONOR_ROOT}/13_attention_is_all_you_need.ipynb",
        "donor_origin": f"donor notebook: {DONOR_ROOT}/13_attention_is_all_you_need.ipynb",
        "supporting_file": DONOR_MANIFEST,
        "extraction_notes": "Scaled dot-product attention, masking, feed-forward layers, and positional encodings were factored into the local decoder-only transformer baseline.",
    },
    "architecture.encoder_decoder_demo": {
        "evidence_file": f"{DONOR_ROOT}/14_bahdanau_attention.ipynb",
        "donor_origin": f"donor notebook: {DONOR_ROOT}/14_bahdanau_attention.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/docs/paper_map.md",
        "extraction_notes": "Cross-attention and copy-focused behavior were re-expressed as a mechanism-level encoder-decoder demo rather than a survey placeholder.",
    },
    "pretraining.masked_lm_demo": {
        "evidence_file": SURVEY_PDF,
        "donor_origin": "repo-authored bidirectional masking demo",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/docs/full_survey_gap_audit.md",
        "extraction_notes": "Mask prediction uses local bidirectional context counts to demonstrate denoising-style pretraining behavior on a tiny corpus.",
    },
    "pretraining.causal_lm": {
        "evidence_file": f"{DONOR_ROOT}/13_attention_is_all_you_need.ipynb",
        "donor_origin": f"transformer core adapted from {DONOR_ROOT}/13_attention_is_all_you_need.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/docs/reuse_audit.md",
        "extraction_notes": "Decoder-only next-token scoring reuses the local transformer core and reports perplexity/generation behavior on canonical toy corpora.",
    },
    "pretraining.scaling_laws": {
        "evidence_file": f"{DONOR_ROOT}/22_scaling_laws.ipynb",
        "donor_origin": f"donor notebook: {DONOR_ROOT}/22_scaling_laws.ipynb",
        "supporting_file": DONOR_MANIFEST,
        "extraction_notes": "Power-law fitting and sectioned compute/data/parameter curves are direct educational descendants of the donor scaling notebook.",
    },
    "pretraining.multi_token_prediction": {
        "evidence_file": f"{DONOR_ROOT}/27_multi_token_prediction.ipynb",
        "donor_origin": f"donor notebook: {DONOR_ROOT}/27_multi_token_prediction.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/docs/module_matrix.md",
        "extraction_notes": "Multi-step horizon prediction is implemented as a lightweight surrogate for multi-token objective behavior.",
    },
    "pretraining.prefix_decoder_demo": {
        "evidence_file": f"{DONOR_ROOT}/13_attention_is_all_you_need.ipynb",
        "donor_origin": f"donor notebook: {DONOR_ROOT}/13_attention_is_all_you_need.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/prefix_decoder_demo.json",
        "extraction_notes": "Prefix-conditioned decoding reuses the same local transformer scaffolding that was originally extracted from the donor transformer notebook.",
    },
    "systems.pipeline_parallelism": {
        "evidence_file": f"{DONOR_ROOT}/09_gpipe.ipynb",
        "donor_origin": f"donor notebook: {DONOR_ROOT}/09_gpipe.ipynb",
        "supporting_file": DONOR_MANIFEST,
        "extraction_notes": "Pipeline stage scheduling and bubble accounting are summarized as a local mechanism demo for systems coverage.",
    },
    "systems.kv_cache_toy": {
        "evidence_file": f"{DONOR_ROOT}/13_attention_is_all_you_need.ipynb",
        "donor_origin": f"transformer cache logic derived from {DONOR_ROOT}/13_attention_is_all_you_need.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/kv_cache_toy_demo.json",
        "extraction_notes": "KV-cache behavior is exposed as a decoder-state reuse demo on top of the donor-derived transformer core.",
    },
    "utilization.retrieval": {
        "evidence_file": f"{DONOR_ROOT}/28_dense_passage_retrieval.ipynb",
        "donor_origin": f"donor notebook: {DONOR_ROOT}/28_dense_passage_retrieval.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/docs/reuse_audit.md",
        "extraction_notes": "Dense, sparse, and hybrid retrieval metrics were reworked into comparable local retrieval slices with recall and MRR outputs.",
    },
    "utilization.rag": {
        "evidence_file": f"{DONOR_ROOT}/29_rag.ipynb",
        "donor_origin": f"donor notebook: {DONOR_ROOT}/29_rag.ipynb",
        "supporting_file": DONOR_MANIFEST,
        "extraction_notes": "Retriever-reader composition is represented as a lightweight local RAG pipeline that preserves grounding-oriented report outputs.",
    },
    "utilization.react_demo": {
        "evidence_file": SURVEY_PDF,
        "donor_origin": "repo-authored ReAct-lite implementation",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/docs/full_survey_gap_audit.md",
        "extraction_notes": "Reason-act-observe traces expose tool choice and environment updates rather than only logging a selected tool name.",
    },
    "evaluation.long_context": {
        "evidence_file": f"{DONOR_ROOT}/30_lost_in_middle.ipynb",
        "donor_origin": f"donor notebook: {DONOR_ROOT}/30_lost_in_middle.ipynb",
        "supporting_file": DONOR_MANIFEST,
        "extraction_notes": "Position-sensitive retrieval behavior follows the same long-context intuition as the donor lost-in-the-middle analysis.",
    },
    "evaluation.position_bias_eval": {
        "evidence_file": f"{DONOR_ROOT}/30_lost_in_middle.ipynb",
        "donor_origin": f"donor context-position analysis: {DONOR_ROOT}/30_lost_in_middle.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/position_bias_eval_demo.json",
        "extraction_notes": "Position-bias scoring is treated as a compact follow-on measurement to the donor long-context position analysis.",
    },
    "architecture.prefix_lm_demo": {
        "evidence_file": f"{DONOR_ROOT}/13_attention_is_all_you_need.ipynb",
        "donor_origin": f"donor transformer scaffold: {DONOR_ROOT}/13_attention_is_all_you_need.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/prefix_lm_demo.json",
        "extraction_notes": "Prefix visibility and constrained-context behavior are demonstrated by adapting the donor-derived transformer stack into a prefix-LM setting.",
    },
    "architecture.bidirectional_encoder_demo": {
        "evidence_file": f"{DONOR_ROOT}/13_attention_is_all_you_need.ipynb",
        "donor_origin": f"donor transformer scaffold: {DONOR_ROOT}/13_attention_is_all_you_need.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/bidirectional_encoder_demo.json",
        "extraction_notes": "The bidirectional encoder proxy keeps the donor transformer attention structure but removes causal masking to expose full-context encoding effects.",
    },
    "architecture.moe_demo": {
        "evidence_file": f"{DONOR_ROOT}/13_attention_is_all_you_need.ipynb",
        "donor_origin": f"donor transformer scaffold: {DONOR_ROOT}/13_attention_is_all_you_need.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/moe_demo.json",
        "extraction_notes": "Expert routing is implemented on top of the same donor-derived transformer-style block structure used elsewhere in the architecture slice.",
    },
    "architecture.configuration_scaling_demo": {
        "evidence_file": f"{DONOR_ROOT}/22_scaling_laws.ipynb",
        "donor_origin": f"donor scaling intuition: {DONOR_ROOT}/22_scaling_laws.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/configuration_scaling_demo.json",
        "extraction_notes": "Depth/width configuration tradeoffs are evaluated through the same local scaling-oriented lens used in the donor scaling notebook.",
    },
    "training.objective_mixture_demo": {
        "evidence_file": f"{DONOR_ROOT}/22_scaling_laws.ipynb",
        "donor_origin": f"donor scaling/data-mixture intuition: {DONOR_ROOT}/22_scaling_laws.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/objective_mixture_demo.json",
        "extraction_notes": "Objective mixture behavior is treated as a compact extension of donor scaling/data-allocation ideas rather than an unrelated synthetic dashboard.",
    },
    "training.optimizer_schedule_demo": {
        "evidence_file": f"{DONOR_ROOT}/22_scaling_laws.ipynb",
        "donor_origin": f"donor optimization/scaling intuition: {DONOR_ROOT}/22_scaling_laws.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/optimizer_schedule_demo.json",
        "extraction_notes": "Optimizer schedule behavior is framed as a compact local extension of the donor scaling notebook's treatment of optimization-sensitive training curves.",
    },
    "training.warmup_decay_demo": {
        "evidence_file": f"{DONOR_ROOT}/22_scaling_laws.ipynb",
        "donor_origin": f"donor optimization/scaling intuition: {DONOR_ROOT}/22_scaling_laws.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/warmup_decay_demo.json",
        "extraction_notes": "Warmup/decay scheduling is evaluated with the same donor-linked optimization intuition used for the local scaling-law slice.",
    },
    "training.gradient_checkpointing_demo": {
        "evidence_file": f"{DONOR_ROOT}/09_gpipe.ipynb",
        "donor_origin": f"donor systems/training utility scaffold: {DONOR_ROOT}/09_gpipe.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/gradient_checkpointing_demo.json",
        "extraction_notes": "Checkpointing is treated as a local memory-saving extension of the donor pipeline/systems training scaffold.",
    },
    "training.memory_partitioning_demo": {
        "evidence_file": f"{DONOR_ROOT}/09_gpipe.ipynb",
        "donor_origin": f"donor systems/training utility scaffold: {DONOR_ROOT}/09_gpipe.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/memory_partitioning_demo.json",
        "extraction_notes": "Memory partitioning is modeled as a local extension of the donor pipeline/systems training scaffold for parameter/state sharding tradeoffs.",
    },
    "training.batch_scaling_demo": {
        "evidence_file": f"{DONOR_ROOT}/22_scaling_laws.ipynb",
        "donor_origin": f"donor scaling/data-allocation intuition: {DONOR_ROOT}/22_scaling_laws.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/batch_scaling_demo.json",
        "extraction_notes": "Batch-size tradeoffs are analyzed through the same donor-linked scaling lens used for local optimization and compute-allocation slices.",
    },
    "systems.speculative_decoding_demo": {
        "evidence_file": f"{DONOR_ROOT}/13_attention_is_all_you_need.ipynb",
        "donor_origin": f"donor transformer scaffold: {DONOR_ROOT}/13_attention_is_all_you_need.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/speculative_decoding_demo.json",
        "extraction_notes": "Speculative decoding reuses the donor-derived decoder-style scoring path to compare draft-versus-target token acceptance locally.",
    },
    "systems.optimization_stability_demo": {
        "evidence_file": f"{DONOR_ROOT}/TRAINING_UTILS_README.md",
        "donor_origin": f"donor training utility context: {DONOR_ROOT}/TRAINING_UTILS_README.md",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/optimization_stability_demo.json",
        "extraction_notes": "Optimization-stability heuristics are linked back to the donor training-utility notes and exposed as a compact local stability probe.",
    },
    "systems.inference_batching_demo": {
        "evidence_file": f"{DONOR_ROOT}/13_attention_is_all_you_need.ipynb",
        "donor_origin": f"donor decoder throughput scaffold: {DONOR_ROOT}/13_attention_is_all_you_need.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/inference_batching_demo.json",
        "extraction_notes": "Inference batching is modeled on the same donor-derived decoder scoring path used by local serving and cache-efficiency slices.",
    },
    "pretraining.data_mixture_toy": {
        "evidence_file": f"{DONOR_ROOT}/22_scaling_laws.ipynb",
        "donor_origin": f"donor data-allocation intuition: {DONOR_ROOT}/22_scaling_laws.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/data_mixture_toy_demo.json",
        "extraction_notes": "Data-mixture tradeoffs are expressed as a compact local allocation experiment aligned with the donor scaling notebook's treatment of data distribution effects.",
    },
    "pretraining.data_curriculum_demo": {
        "evidence_file": f"{DONOR_ROOT}/22_scaling_laws.ipynb",
        "donor_origin": f"donor data-ordering/scaling intuition: {DONOR_ROOT}/22_scaling_laws.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/data_curriculum_demo.json",
        "extraction_notes": "Curriculum ordering is represented as a local data-ordering extension of the donor-linked scaling and data-allocation viewpoint.",
    },
    "utilization.retrieval_selection_demo": {
        "evidence_file": f"{DONOR_ROOT}/28_dense_passage_retrieval.ipynb",
        "donor_origin": f"donor retrieval scaffold: {DONOR_ROOT}/28_dense_passage_retrieval.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/retrieval_selection_demo.json",
        "extraction_notes": "Retriever selection is framed as a follow-on policy over the donor-derived dense/sparse retrieval signals used in the base retrieval demo.",
    },
    "utilization.toolformer_style_demo": {
        "evidence_file": f"{DONOR_ROOT}/29_rag.ipynb",
        "donor_origin": f"donor retrieval-augmented generation scaffold: {DONOR_ROOT}/29_rag.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/toolformer_style_demo.json",
        "extraction_notes": "The toolformer-style demo extends donor-derived retrieval-augmented control flow into explicit tool-call selection and execution traces.",
    },
    "utilization.context_packing_demo": {
        "evidence_file": f"{DONOR_ROOT}/30_lost_in_middle.ipynb",
        "donor_origin": f"donor context-placement analysis: {DONOR_ROOT}/30_lost_in_middle.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/context_packing_demo.json",
        "extraction_notes": "Context packing is treated as a placement-sensitive extension of the donor long-context analysis, focusing on where evidence lands inside a finite window.",
    },
    "utilization.example_selection_demo": {
        "evidence_file": f"{DONOR_ROOT}/28_dense_passage_retrieval.ipynb",
        "donor_origin": f"donor retrieval/ranking scaffold: {DONOR_ROOT}/28_dense_passage_retrieval.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/example_selection_demo.json",
        "extraction_notes": "Example selection is framed as a small retrieval-and-ranking policy over candidate demonstrations, grounded in the donor retrieval notebook's ranking mechanics.",
    },
    "reasoning_faithfulness_eval": {
        "evidence_file": f"{DONOR_ROOT}/30_lost_in_middle.ipynb",
        "donor_origin": f"donor context-position analysis: {DONOR_ROOT}/30_lost_in_middle.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/reasoning_faithfulness_eval_demo.json",
        "extraction_notes": "Reasoning-faithfulness is evaluated with the same local emphasis on evidence placement and recoverability that underlies the donor long-context analysis.",
    },
    "evaluation.truthfulness_eval": {
        "evidence_file": SURVEY_PDF,
        "donor_origin": "repo-authored truthfulness scoring demo",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/docs/limitations.md",
        "extraction_notes": "Truthfulness is measured through tiny local factual-choice probes and imitation-gap scoring rather than a benchmark replication.",
    },
    "evaluation.verifier_eval": {
        "evidence_file": SURVEY_PDF,
        "donor_origin": "repo-authored verifier-style evaluation demo",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/docs/full_survey_gap_audit.md",
        "extraction_notes": "Verifier behavior is approximated with local answer-checking and confidence-weighted acceptance logic.",
    },
    "adaptation.finetuning": {
        "evidence_file": f"{DONOR_ROOT}/02_char_rnn_karpathy.ipynb",
        "donor_origin": f"local RNN fine-tuning built on {DONOR_ROOT}/02_char_rnn_karpathy.ipynb",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/finetuning_demo.json",
        "extraction_notes": "Baseline and adapted checkpoints are selected by eval loss to keep the local fine-tuning demo mechanically honest.",
    },
    "adaptation.peft_lora": {
        "evidence_file": SURVEY_PDF,
        "donor_origin": "repo-authored LoRA-lite parameter-efficient adapter",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/peft_lora_demo.json",
        "extraction_notes": "The demo isolates trainable adapter mass and reports trainable fraction to keep parameter efficiency explicit.",
    },
    "adaptation.dpo_toy": {
        "evidence_file": SURVEY_PDF,
        "donor_origin": "repo-authored DPO-lite preference optimization demo",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/dpo_toy_demo.json",
        "extraction_notes": "Chosen-versus-rejected preference updates are implemented directly instead of routing through generic adaptation scaffolding.",
    },
    "code_generation_risk_eval": {
        "evidence_file": SURVEY_PDF,
        "donor_origin": "repo-authored code risk evaluation demo",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/code_generation_risk_eval_demo.json",
        "extraction_notes": "Code-generation risk is scored through compact local probes for unsafe suggestions and brittle completion behavior rather than a benchmark-scale audit.",
    },
    "capability_vs_alignment_tradeoff_demo": {
        "evidence_file": SURVEY_PDF,
        "donor_origin": "repo-authored capability/alignment tradeoff demo",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/capability_vs_alignment_tradeoff_demo.json",
        "extraction_notes": "The tradeoff demo pairs capability gain and alignment cost on the same local scale so the publication layer can expose the tension explicitly.",
    },
    "memorization_vs_generalization_demo": {
        "evidence_file": SURVEY_PDF,
        "donor_origin": "repo-authored memorization/generalization demo",
        "supporting_file": "/Users/hifi/Downloads/LLM_Survey/artifacts/generated/memorization_vs_generalization_demo.json",
        "extraction_notes": "This demo contrasts near-copy memorization and transfer-style generalization with a toy local split instead of a full memorization audit.",
    },
}


def _band_for_module(module_name: str, fidelity: dict[str, object]) -> str:
    bands = fidelity["bands"]
    if module_name in bands["mechanism_level"]:
        return "mechanism-level"
    if module_name in bands["survey_map"]:
        return "survey-map"
    return "synthetic-lite"


def _demo_filename_for_module(module_name: str, report_index: dict[str, object]) -> str:
    experiment_ids = report_index["modules"].get(module_name, [])
    return f"{experiment_ids[0]}.json" if experiment_ids else "n/a"


def _recall_at_max_k(recall_payload: dict[str, float]) -> float:
    if not recall_payload:
        return 0.0
    max_key = max(recall_payload, key=lambda key: int(key))
    return float(recall_payload[max_key])


def _default_mechanism_evidence(
    module_name: str,
    demo_filename: str,
    generated_dir: Path,
) -> dict[str, str]:
    repo_root = generated_dir.parents[1]
    supporting_file = str(generated_dir / demo_filename) if demo_filename != "n/a" else "/Users/hifi/Downloads/LLM_Survey/docs/module_matrix.md"
    prefix = module_name.split(".", 1)[0]
    short_name = module_name.split(".", 1)[1] if "." in module_name else module_name
    exact_source = repo_root / "src" / "modules" / prefix / f"{short_name}.py"
    if prefix in {
        "code_generation_risk_eval",
        "retrieval_grounding_eval",
        "reasoning_faithfulness_eval",
        "safety_reasoning_tradeoff_demo",
        "capability_vs_alignment_tradeoff_demo",
        "memorization_vs_generalization_demo",
    }:
        exact_source = repo_root / "src" / "modules" / "crosscutting" / f"{module_name}.py"
    evidence_file = str(exact_source) if exact_source.exists() else SURVEY_PDF
    if prefix in {"foundations", "pretraining", "architecture", "systems", "utilization", "adaptation"}:
        donor_origin = "repo-authored dedicated mechanism demo"
    elif prefix in {"training", "evaluation", "applications", "multilingual", "code_pretraining"}:
        donor_origin = "repo-authored dedicated training/evaluation demo"
    else:
        donor_origin = "repo-authored dedicated module"
    if prefix == "pretraining":
        extraction_notes = "Local pre-training proxy focuses on corpus, objective, or data-mixture behavior and reports a canonical educational slice instead of a full-scale pretraining pipeline."
    elif prefix == "training":
        extraction_notes = "Training dynamics are compressed into a local optimization or efficiency probe so the report isolates one scheduling, scaling, or memory effect."
    elif prefix == "architecture":
        extraction_notes = "Architecture demo exposes one structural bias or routing behavior in isolation rather than reproducing a full model family."
    elif prefix == "systems":
        extraction_notes = "Systems demo records one serving or memory-management tradeoff with local metrics instead of end-to-end infrastructure replication."
    elif prefix == "utilization":
        extraction_notes = "Utilization demo highlights prompting, retrieval, planning, or tool-use behavior through a narrow local mechanism and canonical report output."
    elif prefix == "evaluation":
        extraction_notes = "Evaluation demo compresses a capability or risk concept into a small local probe with directly inspectable metrics."
    elif prefix == "adaptation":
        extraction_notes = "Adaptation demo isolates one update rule or alignment workflow and reports its local gain/loss behavior under the shared schema."
    elif prefix == "multilingual":
        extraction_notes = "Multilingual demo focuses on transfer or prompting behavior under a compact cross-lingual proxy task."
    elif prefix == "code_pretraining":
        extraction_notes = "Code-focused demo uses synthetic or toy program tasks to expose code-pretraining or program-reasoning behavior locally."
    elif prefix == "applications":
        extraction_notes = "Application demo is a task-shaped local stub that preserves the survey-facing interaction pattern rather than a production system."
    else:
        extraction_notes = "Canonical local demo output is linked directly as the module's dedicated mechanism-level artifact."

    if "reasoning" in short_name or "scratchpad" in short_name:
        extraction_notes = "Reasoning-oriented demo traces intermediate decomposition or verification behavior so the local artifact exposes process, not only final score."
    elif "retrieval" in short_name or "rag" in short_name:
        extraction_notes = "Retrieval-grounded demo reports ranking or grounding behavior on a compact local corpus with canonical JSON outputs for comparison."
    elif "bias" in short_name or "safety" in short_name or "privacy" in short_name or "truth" in short_name:
        extraction_notes = "Risk-oriented demo uses a compact local probe to expose one safety, fairness, privacy, or truthfulness tradeoff with inspectable metrics."
    elif "tokenizer" in short_name or "token" in short_name:
        extraction_notes = "Tokenization or token-prediction demo isolates one representation choice and reports its local effect through a narrow proxy task."
    elif "curriculum" in short_name or "mixture" in short_name or "quality" in short_name or "contamination" in short_name or "dedup" in short_name:
        extraction_notes = "Data-centric demo isolates one curation or mixture effect and reports its impact through a small controlled corpus slice."
    elif "optimizer" in short_name or "warmup" in short_name or "batch" in short_name or "checkpoint" in short_name or "partition" in short_name:
        extraction_notes = "Optimization demo isolates one schedule, batch, or memory-control choice so the local report focuses on one training-side tradeoff."
    elif "tool" in short_name or "planning" in short_name or "react" in short_name:
        extraction_notes = "Tool-use or planning demo emphasizes action selection and local control flow, exposing stepwise behavior rather than only end results."
    return {
        "evidence_file": evidence_file,
        "donor_origin": donor_origin,
        "supporting_file": supporting_file,
        "extraction_notes": extraction_notes,
        "artifact_mode": "mechanism-level demo",
        "canonical_output": supporting_file,
    }


def _confidence_tag(donor_origin: str) -> str:
    lowered = donor_origin.lower()
    if (
        "donor notebook" in lowered
        or "adapted from" in lowered
        or "derived from" in lowered
        or "donor manifest" in lowered
        or "donor context" in lowered
        or lowered.startswith("donor ")
    ):
        return "donor-derived"
    if "generated" in lowered or "computed" in lowered:
        return "computed-summary"
    return "repo-authored"


def _default_survey_map_row(module_name: str, meta: dict[str, str], report_index: dict[str, object], generated_dir: Path) -> dict[str, str]:
    demo_filename = _demo_filename_for_module(module_name, report_index)
    canonical_output = str(generated_dir / demo_filename) if demo_filename != "n/a" else meta["supporting_file"]
    artifact_mode = {
        "resources": "resource table",
        "reporting": "reporting dashboard",
        "benchmark": "bundle summary",
    }.get(module_name.split(".", 1)[0], "survey-map artifact")
    return {
        "module": module_name,
        **meta,
        "artifact_mode": artifact_mode,
        "canonical_output": canonical_output,
    }


def _write_svg_bar_chart(
    output_path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    max_value: float,
    bar_color: str,
) -> None:
    width = 920
    row_h = 32
    left = 240
    top = 56
    chart_w = 620
    height = top + row_h * len(labels) + 40
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfbf7"/>',
        f'<text x="24" y="34" font-family="Georgia, serif" font-size="24" fill="#202124">{title}</text>',
    ]
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = top + idx * row_h
        bar_w = 0 if max_value <= 0 else (value / max_value) * chart_w
        lines.append(f'<text x="24" y="{y + 20}" font-family="Courier New, monospace" font-size="13" fill="#333">{label}</text>')
        lines.append(f'<rect x="{left}" y="{y + 6}" width="{chart_w}" height="18" fill="#e6e1d8"/>')
        lines.append(f'<rect x="{left}" y="{y + 6}" width="{bar_w:.2f}" height="18" fill="{bar_color}"/>')
        lines.append(
            f'<text x="{left + chart_w + 12}" y="{y + 20}" font-family="Courier New, monospace" font-size="12" fill="#333">{value:.3f}</text>'
        )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n")


def _write_svg_line_chart(
    output_path: Path,
    title: str,
    caption: str,
    series: list[dict[str, object]],
    y_min: float,
    y_max: float,
) -> None:
    width = 960
    height = 460
    left = 80
    right = 40
    top = 50
    bottom = 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#1d4e89", "#b35c2e", "#2d6a4f", "#8e3b46"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfbf7"/>',
        f'<text x="24" y="32" font-family="Georgia, serif" font-size="24" fill="#202124">{title}</text>',
        f'<text x="24" y="52" font-family="Georgia, serif" font-size="13" fill="#5b5b5b">{caption}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#555" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#555" stroke-width="1.5"/>',
    ]
    for tick in range(5):
        y_value = y_min + (y_max - y_min) * tick / 4.0
        y = top + plot_h - (plot_h * tick / 4.0)
        lines.append(f'<line x1="{left - 4}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="#ddd" stroke-width="1"/>')
        lines.append(f'<text x="12" y="{y + 4}" font-family="Courier New, monospace" font-size="12" fill="#555">{y_value:.2f}</text>')
    for idx, item in enumerate(series):
        values = item["values"]
        if len(values) == 1:
            xs = [left + plot_w / 2.0]
        else:
            xs = [left + (plot_w * i / (len(values) - 1)) for i in range(len(values))]
        ys = [top + plot_h - ((value - y_min) / max(y_max - y_min, 1e-9)) * plot_h for value in values]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        color = colors[idx % len(colors)]
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
        for x, y in zip(xs, ys):
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
        lines.append(f'<rect x="{left + plot_w - 190}" y="{top + 10 + idx * 18}" width="12" height="3" fill="{color}"/>')
        lines.append(
            f'<text x="{left + plot_w - 170}" y="{top + 20 + idx * 18}" font-family="Courier New, monospace" font-size="12" fill="{color}">{item["label"]}</text>'
        )
    for point_idx in range(max(len(item["values"]) for item in series)):
        if point_idx >= len(series[0]["x_labels"]):
            break
        x = left if len(series[0]["x_labels"]) == 1 else left + (plot_w * point_idx / (len(series[0]["x_labels"]) - 1))
        lines.append(
            f'<text x="{x - 16:.1f}" y="{top + plot_h + 24}" font-family="Courier New, monospace" font-size="11" fill="#555">{series[0]["x_labels"][point_idx]}</text>'
        )
    lines.append(
        f'<text x="24" y="{height - 16}" font-family="Georgia, serif" font-size="12" fill="#5b5b5b">All series share the same y-scale for direct visual comparison.</text>'
    )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n")


def _write_svg_multi_panel_line_figure(
    output_path: Path,
    title: str,
    panels: list[dict[str, object]],
    y_min: float,
    y_max: float,
) -> None:
    width = 1240
    height = 1120
    left = 90
    panel_width = 1040
    panel_height = 250
    colors = ["#1d4e89", "#b35c2e", "#2d6a4f", "#8e3b46"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfbf7"/>',
        '<rect x="12" y="12" width="1216" height="1090" rx="18" fill="none" stroke="#d8d2c3" stroke-width="1.5"/>',
        f'<text x="24" y="34" font-family="Georgia, serif" font-size="28" fill="#202124">{title}</text>',
        '<text x="24" y="58" font-family="Georgia, serif" font-size="14" fill="#5b5b5b">Figure Set A: analytic trend panels. Panels A-C share a common normalized 0-1 y-scale for direct visual comparison.</text>',
        '<text x="24" y="78" font-family="Courier New, monospace" font-size="12" fill="#4a4a4a">Label key: A=Adaptation, B=Retrieval, C=Risk.</text>',
    ]
    for panel_idx, panel in enumerate(panels):
        panel_top = 116 + panel_idx * 320
        lines.append(f'<rect x="20" y="{panel_top - 10}" width="1190" height="286" rx="14" fill="#fffdfa" stroke="#e4ddd0" stroke-width="1"/>')
        plot_left = left
        plot_top = panel_top + 42
        plot_width = panel_width - 150
        plot_height = panel_height - 72
        lines.append(
            f'<text x="24" y="{panel_top + 20}" font-family="Georgia, serif" font-size="18" fill="#202124">{chr(65 + panel_idx)}. {panel["title"]}</text>'
        )
        lines.append(
            f'<text x="48" y="{panel_top + 40}" font-family="Georgia, serif" font-size="12" fill="#5b5b5b">{panel["caption"]}</text>'
        )
        lines.append(f'<line x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" stroke="#555" stroke-width="1.5"/>')
        lines.append(f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}" stroke="#555" stroke-width="1.5"/>')
        for tick in range(5):
            y_value = y_min + (y_max - y_min) * tick / 4.0
            y = plot_top + plot_height - (plot_height * tick / 4.0)
            lines.append(f'<line x1="{plot_left - 4}" y1="{y}" x2="{plot_left + plot_width}" y2="{y}" stroke="#ddd" stroke-width="1"/>')
            lines.append(f'<text x="20" y="{y + 4}" font-family="Courier New, monospace" font-size="11" fill="#555">{y_value:.2f}</text>')
        for series_idx, series in enumerate(panel["series"]):
            values = series["values"]
            x_labels = series["x_labels"]
            xs = [plot_left + (plot_width * i / max(len(values) - 1, 1)) for i in range(len(values))]
            ys = [plot_top + plot_height - ((value - y_min) / max(y_max - y_min, 1e-9)) * plot_height for value in values]
            color = colors[series_idx % len(colors)]
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
            for x, y in zip(xs, ys):
                lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
            legend_y = panel_top + 18 + series_idx * 16
            lines.append(f'<rect x="{left + 910}" y="{legend_y - 8}" width="12" height="3" fill="{color}"/>')
            lines.append(f'<text x="{left + 930}" y="{legend_y}" font-family="Courier New, monospace" font-size="11" fill="{color}">{series["label"]}</text>')
            for point_idx, x_label in enumerate(x_labels):
                if point_idx >= len(xs):
                    break
                lines.append(
                    f'<text x="{xs[point_idx] - 18:.1f}" y="{plot_top + plot_height + 22}" font-family="Courier New, monospace" font-size="10" fill="#555">{x_label}</text>'
                )
    lines.append('<text x="24" y="1098" font-family="Georgia, serif" font-size="12" fill="#5b5b5b">Shared publication caption: use the normalized y-axis for cross-panel comparison; use the companion callout board for exact slice values.</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n")


def _write_svg_callout_board(
    output_path: Path,
    title: str,
    sections: list[dict[str, object]],
) -> None:
    width = 1240
    height = 980
    section_height = 280
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfbf7"/>',
        '<rect x="12" y="12" width="1216" height="948" rx="18" fill="none" stroke="#d8d2c3" stroke-width="1.5"/>',
        f'<text x="24" y="34" font-family="Georgia, serif" font-size="28" fill="#202124">{title}</text>',
        '<text x="24" y="56" font-family="Georgia, serif" font-size="14" fill="#5b5b5b">Figure Set B: companion callout boards for the publication trend figures.</text>',
        '<text x="24" y="76" font-family="Courier New, monospace" font-size="12" fill="#4a4a4a">Label key: A=Adaptation, B=Retrieval, C=Risk.</text>',
    ]
    for idx, section in enumerate(sections):
        top = 112 + idx * section_height
        lines.append(f'<rect x="20" y="{top - 12}" width="1190" height="240" rx="14" fill="#fffdfa" stroke="#e4ddd0" stroke-width="1"/>')
        lines.append(f'<text x="24" y="{top}" font-family="Georgia, serif" font-size="18" fill="#202124">{chr(65 + idx)}. {section["title"]}</text>')
        lines.append(f'<text x="48" y="{top + 20}" font-family="Georgia, serif" font-size="12" fill="#5b5b5b">{section["caption"]}</text>')
        header_y = top + 48
        col_x = section["col_x"]
        for col_idx, header in enumerate(section["headers"]):
            lines.append(f'<text x="{col_x[col_idx]}" y="{header_y}" font-family="Courier New, monospace" font-size="12" fill="#333">{header}</text>')
        lines.append(f'<line x1="24" y1="{header_y + 8}" x2="1190" y2="{header_y + 8}" stroke="#bbb" stroke-width="1"/>')
        for row_idx, row in enumerate(section["rows"]):
            y = header_y + 30 + row_idx * 22
            for col_idx, cell in enumerate(row):
                lines.append(f'<text x="{col_x[col_idx]}" y="{y}" font-family="Courier New, monospace" font-size="12" fill="#333">{cell}</text>')
    lines.append('<text x="24" y="956" font-family="Georgia, serif" font-size="12" fill="#5b5b5b">Shared publication caption: these tables mirror the plotted slices so figure panels and callout boards can be read as one cohesive set.</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n")


@dataclass
class PublicationAssets:
    repo_root: str | Path

    def build(self) -> dict[str, object]:
        repo_root = Path(self.repo_root)
        docs_dir = repo_root / "docs"
        generated_dir = repo_root / "artifacts" / "generated"
        tables_dir = generated_dir / "tables"
        figures_dir = generated_dir / "figures"
        tools_dir = repo_root / "scripts"
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        tools_dir.mkdir(parents=True, exist_ok=True)

        report_index = ReportIndex(generated_dir).build()
        fidelity = FidelityBandDashboard(generated_dir).build()
        provenance = ModuleProvenanceDashboard(generated_dir).build()
        sections = PaperSectionDashboard(generated_dir).build()
        benchmark = BenchmarkHarness(generated_dir).compare()

        module_rows = []
        for module_name in IMPLEMENTATION_TARGETS:
            demo_name = _demo_filename_for_module(module_name, report_index)
            module_rows.append(
                {
                    "module": module_name,
                    "status": "Implemented" if demo_name != "n/a" else "Missing",
                    "fidelity_band": _band_for_module(module_name, fidelity),
                    "provenance": "dedicated" if module_name in provenance["dedicated_modules"] else "generated",
                    "demo": demo_name,
                }
            )

        survey_map_rows = [
            _default_survey_map_row(module, meta, report_index, generated_dir) for module, meta in SURVEY_MAP_EVIDENCE.items()
        ]
        for row in survey_map_rows:
            row["confidence_tag"] = _confidence_tag(str(row["donor_origin"]))
        mechanism_rows = []
        for row in module_rows:
            if row["fidelity_band"] != "mechanism-level":
                continue
            mechanism_row = {
                "module": row["module"],
                **_default_mechanism_evidence(row["module"], row["demo"], generated_dir),
                **MECHANISM_EVIDENCE.get(row["module"], {}),
            }
            mechanism_row["confidence_tag"] = _confidence_tag(str(mechanism_row["donor_origin"]))
            mechanism_rows.append(mechanism_row)
        confidence_lookup = {row["module"]: row["confidence_tag"] for row in survey_map_rows}
        confidence_lookup.update({row["module"]: row["confidence_tag"] for row in mechanism_rows})
        confidence_counts = {
            "donor-derived": sum(1 for row in module_rows if confidence_lookup.get(row["module"]) == "donor-derived"),
            "repo-authored": sum(1 for row in module_rows if confidence_lookup.get(row["module"]) == "repo-authored"),
            "computed-summary": sum(1 for row in module_rows if confidence_lookup.get(row["module"]) == "computed-summary"),
        }

        adaptation_rows = json.loads((generated_dir / "adaptation_leaderboard_demo.json").read_text())["artifacts"]["top_by_gain"][:5]
        retrieval_rows = json.loads((generated_dir / "retrieval_demo.json").read_text())["metrics"]
        retrieval_artifacts = json.loads((generated_dir / "retrieval_demo.json").read_text())["artifacts"]
        risk_sources = {
            "truthfulness_eval_demo.json": "truthfulness_score",
            "hallucination_checks_demo.json": "supported_rate",
            "safety_eval_demo.json": "refusal_rate",
            "bias_eval_demo.json": "fairness_score",
            "privacy_leakage_eval_demo.json": "privacy_risk",
        }
        risk_rows = []
        for filename, metric_name in risk_sources.items():
            payload = json.loads((generated_dir / filename).read_text())
            risk_rows.append({"label": filename.replace("_demo.json", ""), "value": float(payload["metrics"][metric_name])})
        truthfulness_payload = json.loads((generated_dir / "truthfulness_eval_demo.json").read_text())
        hallucination_payload = json.loads((generated_dir / "hallucination_checks_demo.json").read_text())
        safety_payload = json.loads((generated_dir / "safety_eval_demo.json").read_text())
        bias_payload = json.loads((generated_dir / "bias_eval_demo.json").read_text())
        privacy_payload = json.loads((generated_dir / "privacy_leakage_eval_demo.json").read_text())

        fidelity_matrix_path = docs_dir / "fidelity_matrix.md"
        module_matrix_path = docs_dir / "module_matrix.md"
        figures_tables_path = docs_dir / "figures_and_tables.md"
        mechanism_provenance_path = docs_dir / "mechanism_provenance.md"
        resource_provenance_path = docs_dir / "resource_provenance.md"
        survey_map_provenance_path = docs_dir / "survey_map_provenance.md"
        module_matrix_csv = tables_dir / "module_matrix.csv"
        benchmark_csv = tables_dir / "benchmark_experiment_scores.csv"
        benchmark_family_csv = tables_dir / "benchmark_family_scores.csv"
        benchmark_family_group_csv = tables_dir / "benchmark_family_group_scores.csv"
        section_csv = tables_dir / "paper_section_completion.csv"
        survey_map_csv = tables_dir / "survey_map_provenance.csv"
        mechanism_csv = tables_dir / "mechanism_provenance.csv"
        section_svg = figures_dir / "paper_section_completion.svg"
        benchmark_svg = figures_dir / "benchmark_family_scores.svg"
        fidelity_svg = figures_dir / "fidelity_band_split.svg"
        adaptation_svg = figures_dir / "adaptation_gain_trends.svg"
        retrieval_svg = figures_dir / "retrieval_slice_trends.svg"
        risk_svg = figures_dir / "risk_slice_trends.svg"
        trend_panels_svg = figures_dir / "trend_panels.svg"
        trend_callouts_svg = figures_dir / "trend_callouts.svg"

        with module_matrix_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["module", "status", "fidelity_band", "confidence_tag", "provenance", "demo"])
            writer.writeheader()
            writer.writerows(
                [{**row, "confidence_tag": confidence_lookup.get(row["module"], "repo-authored")} for row in module_rows]
            )

        with benchmark_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["experiment_id", "module", "normalized_score", "num_metrics"])
            writer.writeheader()
            writer.writerows(benchmark["experiment_scores"])

        with benchmark_family_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["family", "family_group", "normalized_score", "num_metrics"])
            writer.writeheader()
            writer.writerows(benchmark["family_scores"])

        with benchmark_family_group_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["family_group", "normalized_score", "num_metrics"])
            writer.writeheader()
            writer.writerows(benchmark["family_group_scores"])

        with section_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["section", "prefix", "implemented", "target", "percentage"])
            writer.writeheader()
            writer.writerows(sections["sections"])

        with survey_map_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "module",
                    "artifact_mode",
                    "confidence_tag",
                    "evidence_file",
                    "donor_origin",
                    "supporting_file",
                    "canonical_output",
                    "extraction_notes",
                ],
            )
            writer.writeheader()
            writer.writerows(survey_map_rows)

        with mechanism_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "module",
                    "artifact_mode",
                    "confidence_tag",
                    "evidence_file",
                    "donor_origin",
                    "supporting_file",
                    "canonical_output",
                    "extraction_notes",
                ],
            )
            writer.writeheader()
            writer.writerows(mechanism_rows)

        _write_svg_bar_chart(
            section_svg,
            "Paper Section Completion",
            [row["section"] for row in sections["sections"]],
            [float(row["percentage"]) for row in sections["sections"]],
            100.0,
            "#b35c2e",
        )
        _write_svg_bar_chart(
            benchmark_svg,
            "Benchmark Family Scores",
            [row["family"] for row in benchmark["family_scores"]],
            [float(row["normalized_score"]) for row in benchmark["family_scores"]],
            1.0,
            "#2d6a4f",
        )
        _write_svg_bar_chart(
            fidelity_svg,
            "Fidelity Band Split",
            ["mechanism-level", "survey-map", "synthetic-lite"],
            [
                float(fidelity["mechanism_level_fraction"]),
                float(fidelity["survey_map_fraction"]),
                max(0.0, 1.0 - float(fidelity["mechanism_level_fraction"]) - float(fidelity["survey_map_fraction"])),
            ],
            1.0,
            "#1d4e89",
        )
        _write_svg_line_chart(
            adaptation_svg,
            "Adaptation Trends",
            "Normalized gain, efficiency, and loss-improvement curves across the top adaptation demos.",
            [
                {
                    "label": "gain_norm",
                    "values": [
                        float(row["gain"]) / max(float(top["gain"]) for top in adaptation_rows)
                        for row in adaptation_rows
                    ],
                    "x_labels": [row["experiment_id"].replace("_demo", "") for row in adaptation_rows],
                },
                {
                    "label": "efficiency_norm",
                    "values": [
                        float(row.get("efficiency_score", row["gain"])) / max(
                            float(top.get("efficiency_score", top["gain"])) for top in adaptation_rows
                        )
                        for row in adaptation_rows
                    ],
                    "x_labels": [row["experiment_id"].replace("_demo", "") for row in adaptation_rows],
                },
                {
                    "label": "loss_improvement_norm",
                    "values": [
                        (max(float(top["adapted_loss"]) for top in adaptation_rows) - float(row["adapted_loss"]))
                        / max(
                            max(float(top["adapted_loss"]) for top in adaptation_rows)
                            - min(float(top["adapted_loss"]) for top in adaptation_rows),
                            1e-9,
                        )
                        for row in adaptation_rows
                    ],
                    "x_labels": [row["experiment_id"].replace("_demo", "") for row in adaptation_rows],
                },
            ],
            y_min=0.0,
            y_max=1.0,
        )
        _write_svg_line_chart(
            retrieval_svg,
            "Retrieval Slice Trends",
            "Shared-scale retrieval quality panel showing ranking and recall behavior across retriever families.",
            [
                {
                    "label": "MRR",
                    "values": [
                        float(retrieval_rows["dense_mrr"]),
                        float(retrieval_rows["bm25_mrr"]),
                        float(retrieval_rows["hybrid_mrr"]),
                    ],
                    "x_labels": ["dense", "bm25", "hybrid"],
                },
                {
                    "label": "Recall",
                    "values": [
                        _recall_at_max_k(retrieval_artifacts["dense_recall"]),
                        _recall_at_max_k(retrieval_artifacts["bm25_recall"]),
                        _recall_at_max_k(retrieval_artifacts["hybrid_recall"]),
                    ],
                    "x_labels": ["dense", "bm25", "hybrid"],
                }
            ],
            y_min=0.0,
            y_max=1.0,
        )
        _write_svg_line_chart(
            risk_svg,
            "Risk Slice Trends",
            "Risk panel with aligned beneficial-versus-harm signals so slices remain directly comparable.",
            [
                {
                    "label": "beneficial_signal",
                    "values": [
                        float(truthfulness_payload["metrics"]["truthfulness_score"]),
                        float(hallucination_payload["metrics"]["supported_rate"]),
                        float(safety_payload["metrics"]["refusal_rate"]),
                        float(bias_payload["metrics"]["fairness_score"]),
                        1.0 - float(privacy_payload["metrics"]["privacy_risk"]),
                    ],
                    "x_labels": [row["label"].replace("_", "-") for row in risk_rows],
                },
                {
                    "label": "harm_signal",
                    "values": [
                        1.0 - float(truthfulness_payload["metrics"]["truthfulness_score"]),
                        float(hallucination_payload["metrics"]["hallucination_rate"]),
                        float(safety_payload["metrics"]["jailbreak_success_rate"]),
                        1.0 - float(bias_payload["metrics"]["fairness_score"]),
                        float(privacy_payload["metrics"]["privacy_risk"]),
                    ],
                    "x_labels": [row["label"].replace("_", "-") for row in risk_rows],
                }
            ],
            y_min=0.0,
            y_max=1.0,
        )
        _write_svg_multi_panel_line_figure(
            trend_panels_svg,
            "Publication Trend Panels",
            [
                {
                    "title": "Adaptation",
                    "caption": "Gain, efficiency, and loss-improvement share one normalized axis.",
                    "series": [
                        {
                            "label": "gain_norm",
                            "values": [
                                float(row["gain"]) / max(float(top["gain"]) for top in adaptation_rows)
                                for row in adaptation_rows
                            ],
                            "x_labels": [row["experiment_id"].replace("_demo", "") for row in adaptation_rows],
                        },
                        {
                            "label": "efficiency_norm",
                            "values": [
                                float(row.get("efficiency_score", row["gain"])) / max(
                                    float(top.get("efficiency_score", top["gain"])) for top in adaptation_rows
                                )
                                for row in adaptation_rows
                            ],
                            "x_labels": [row["experiment_id"].replace("_demo", "") for row in adaptation_rows],
                        },
                        {
                            "label": "loss_improvement_norm",
                            "values": [
                                (max(float(top["adapted_loss"]) for top in adaptation_rows) - float(row["adapted_loss"]))
                                / max(
                                    max(float(top["adapted_loss"]) for top in adaptation_rows)
                                    - min(float(top["adapted_loss"]) for top in adaptation_rows),
                                    1e-9,
                                )
                                for row in adaptation_rows
                            ],
                            "x_labels": [row["experiment_id"].replace("_demo", "") for row in adaptation_rows],
                        },
                    ],
                },
                {
                    "title": "Retrieval",
                    "caption": "MRR and recall are aligned for dense, sparse, and hybrid retrieval.",
                    "series": [
                        {
                            "label": "MRR",
                            "values": [
                                float(retrieval_rows["dense_mrr"]),
                                float(retrieval_rows["bm25_mrr"]),
                                float(retrieval_rows["hybrid_mrr"]),
                            ],
                            "x_labels": ["dense", "bm25", "hybrid"],
                        },
                        {
                            "label": "Recall",
                            "values": [
                                _recall_at_max_k(retrieval_artifacts["dense_recall"]),
                                _recall_at_max_k(retrieval_artifacts["bm25_recall"]),
                                _recall_at_max_k(retrieval_artifacts["hybrid_recall"]),
                            ],
                            "x_labels": ["dense", "bm25", "hybrid"],
                        },
                    ],
                },
                {
                    "title": "Risk",
                    "caption": "Beneficial and harm signals are paired for the same evaluation slices.",
                    "series": [
                        {
                            "label": "beneficial_signal",
                            "values": [
                                float(truthfulness_payload["metrics"]["truthfulness_score"]),
                                float(hallucination_payload["metrics"]["supported_rate"]),
                                float(safety_payload["metrics"]["refusal_rate"]),
                                float(bias_payload["metrics"]["fairness_score"]),
                                1.0 - float(privacy_payload["metrics"]["privacy_risk"]),
                            ],
                            "x_labels": [row["label"].replace("_", "-") for row in risk_rows],
                        },
                        {
                            "label": "harm_signal",
                            "values": [
                                1.0 - float(truthfulness_payload["metrics"]["truthfulness_score"]),
                                float(hallucination_payload["metrics"]["hallucination_rate"]),
                                float(safety_payload["metrics"]["jailbreak_success_rate"]),
                                1.0 - float(bias_payload["metrics"]["fairness_score"]),
                                float(privacy_payload["metrics"]["privacy_risk"]),
                            ],
                            "x_labels": [row["label"].replace("_", "-") for row in risk_rows],
                        },
                    ],
                },
            ],
            y_min=0.0,
            y_max=1.0,
        )
        _write_svg_callout_board(
            trend_callouts_svg,
            "Trend Callout Boards",
            [
                {
                    "title": "Adaptation Callouts",
                    "caption": "Normalized gain, efficiency, and loss-improvement for the top adaptation slice.",
                    "headers": ["Demo", "Gain", "Eff.", "Loss+"],
                    "col_x": [40, 520, 680, 840],
                    "rows": [
                        [
                            row["experiment_id"],
                            f"{float(row['gain']) / max(float(top['gain']) for top in adaptation_rows):.3f}",
                            f"{float(row.get('efficiency_score', row['gain'])) / max(float(top.get('efficiency_score', top['gain'])) for top in adaptation_rows):.3f}",
                            f"{((max(float(top['adapted_loss']) for top in adaptation_rows) - float(row['adapted_loss'])) / max(max(float(top['adapted_loss']) for top in adaptation_rows) - min(float(top['adapted_loss']) for top in adaptation_rows), 1e-9)):.3f}",
                        ]
                        for row in adaptation_rows
                    ],
                },
                {
                    "title": "Retrieval Callouts",
                    "caption": "MRR and recall-at-max-k for dense, sparse, and hybrid retrieval.",
                    "headers": ["Retriever", "MRR", "Recall@max_k"],
                    "col_x": [40, 520, 700],
                    "rows": [
                        ["dense", f"{float(retrieval_rows['dense_mrr']):.3f}", f"{_recall_at_max_k(retrieval_artifacts['dense_recall']):.3f}"],
                        ["bm25", f"{float(retrieval_rows['bm25_mrr']):.3f}", f"{_recall_at_max_k(retrieval_artifacts['bm25_recall']):.3f}"],
                        ["hybrid", f"{float(retrieval_rows['hybrid_mrr']):.3f}", f"{_recall_at_max_k(retrieval_artifacts['hybrid_recall']):.3f}"],
                    ],
                },
                {
                    "title": "Risk Callouts",
                    "caption": "Beneficial and harm signals aligned per risk slice.",
                    "headers": ["Slice", "Benefit", "Harm"],
                    "col_x": [40, 520, 700],
                    "rows": [
                        ["truthfulness-eval", f"{float(truthfulness_payload['metrics']['truthfulness_score']):.3f}", f"{1.0 - float(truthfulness_payload['metrics']['truthfulness_score']):.3f}"],
                        ["hallucination-checks", f"{float(hallucination_payload['metrics']['supported_rate']):.3f}", f"{float(hallucination_payload['metrics']['hallucination_rate']):.3f}"],
                        ["safety-eval", f"{float(safety_payload['metrics']['refusal_rate']):.3f}", f"{float(safety_payload['metrics']['jailbreak_success_rate']):.3f}"],
                        ["bias-eval", f"{float(bias_payload['metrics']['fairness_score']):.3f}", f"{1.0 - float(bias_payload['metrics']['fairness_score']):.3f}"],
                        ["privacy-leakage-eval", f"{1.0 - float(privacy_payload['metrics']['privacy_risk']):.3f}", f"{float(privacy_payload['metrics']['privacy_risk']):.3f}"],
                    ],
                },
            ],
        )

        fidelity_lines = [
            "# Fidelity Matrix",
            "",
            f"Confidence summary: donor-derived `{confidence_counts['donor-derived']}`, repo-authored `{confidence_counts['repo-authored']}`, computed-summary `{confidence_counts['computed-summary']}`.",
            "",
            "| Module | Fidelity Band | Confidence | Provenance | Demo |",
            "|---|---|---|---|---|",
        ]
        for row in module_rows:
            fidelity_lines.append(
                f"| `{row['module']}` | {row['fidelity_band']} | {confidence_lookup.get(row['module'], 'repo-authored')} | {row['provenance']} | `{row['demo']}` |"
            )
        fidelity_matrix_path.write_text("\n".join(fidelity_lines) + "\n")

        module_lines = [
            "# Module Matrix",
            "",
            "This document is generated from the current canonical local reports.",
            "",
            f"Confidence summary: donor-derived `{confidence_counts['donor-derived']}`, repo-authored `{confidence_counts['repo-authored']}`, computed-summary `{confidence_counts['computed-summary']}`.",
            "",
            "| Module | Status | Fidelity Band | Confidence | Demo |",
            "|---|---|---|---|---|",
        ]
        for row in module_rows:
            module_lines.append(
                f"| `{row['module']}` | {row['status']} | {row['fidelity_band']} | {confidence_lookup.get(row['module'], 'repo-authored')} | `{row['demo']}` |"
            )
        module_matrix_path.write_text("\n".join(module_lines) + "\n")

        survey_map_lines = [
            "# Survey-Map Provenance",
            "",
            "These survey-map modules are grounded in exact local evidence files, donor origins, canonical outputs, and row-level extraction notes.",
            "",
            "| Module | Artifact Mode | Confidence | Evidence File | Donor Origin | Supporting File | Canonical Output | Extraction Notes |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in survey_map_rows:
            survey_map_lines.append(
                f"| `{row['module']}` | {row['artifact_mode']} | {row['confidence_tag']} | `{row['evidence_file']}` | {row['donor_origin']} | `{row['supporting_file']}` | `{row['canonical_output']}` | {row['extraction_notes']} |"
            )
        survey_map_provenance_path.write_text("\n".join(survey_map_lines) + "\n")

        resource_lines = [
            "# Resource Provenance",
            "",
            "This subset isolates the resource-facing analytical tables from the broader survey-map provenance layer and keeps their canonical outputs explicit.",
            "",
            "| Module | Artifact Mode | Confidence | Evidence File | Donor Origin | Supporting File | Canonical Output | Extraction Notes |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in survey_map_rows:
            if row["module"].startswith("resources."):
                resource_lines.append(
                    f"| `{row['module']}` | {row['artifact_mode']} | {row['confidence_tag']} | `{row['evidence_file']}` | {row['donor_origin']} | `{row['supporting_file']}` | `{row['canonical_output']}` | {row['extraction_notes']} |"
                )
        resource_provenance_path.write_text("\n".join(resource_lines) + "\n")

        mechanism_lines = [
            "# Mechanism Provenance",
            "",
            "Mechanism-level modules now include donor or source extraction notes, canonical outputs, and artifact-mode labeling rather than only a representative subset.",
            "",
            "| Module | Artifact Mode | Confidence | Evidence File | Donor Origin | Supporting File | Canonical Output | Extraction Notes |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in mechanism_rows:
            mechanism_lines.append(
                f"| `{row['module']}` | {row['artifact_mode']} | {row['confidence_tag']} | `{row['evidence_file']}` | {row['donor_origin']} | `{row['supporting_file']}` | `{row['canonical_output']}` | {row['extraction_notes']} |"
            )
        mechanism_provenance_path.write_text("\n".join(mechanism_lines) + "\n")

        top_benchmark = benchmark["experiment_scores"][:10]
        figure_lines = [
            "# Figures And Tables",
            "",
            "## Section Completion",
            "",
            f"![Paper section completion]({section_svg})",
            "",
            "| Section | Implemented | Target | Completion |",
            "|---|---:|---:|---:|",
        ]
        for row in sections["sections"]:
            figure_lines.append(
                f"| {row['section']} | {row['implemented']} | {row['target']} | {row['percentage']:.1f}% |"
            )
        figure_lines.extend(
            [
                "",
                "## Benchmark Families",
                "",
                f"![Benchmark family scores]({benchmark_svg})",
                "",
                "| Family | Score | Metrics |",
                "|---|---:|---:|",
            ]
        )
        for row in benchmark["family_scores"]:
            figure_lines.append(f"| `{row['family']}` | {row['normalized_score']:.4f} | {row['num_metrics']} |")
        figure_lines.extend(
            [
                "",
                "## Benchmark Family Groups",
                "",
                "| Group | Score | Metrics |",
                "|---|---:|---:|",
            ]
        )
        for row in benchmark["family_group_scores"]:
            figure_lines.append(f"| `{row['family_group']}` | {row['normalized_score']:.4f} | {row['num_metrics']} |")
        figure_lines.extend(
            [
                "",
                "## Adaptation Trends",
                "",
                f"![Adaptation trends]({adaptation_svg})",
                "",
                "Caption: Gain, efficiency, and loss-improvement are normalized onto one shared axis so the leading adaptation demos can be compared without mixing raw scales.",
                "",
                "## Retrieval Trends",
                "",
                f"![Retrieval trends]({retrieval_svg})",
                "",
                "Caption: Dense, sparse, and hybrid retrieval are compared with shared-scale MRR and recall series.",
                "",
                "## Risk Trends",
                "",
                f"![Risk trends]({risk_svg})",
                "",
                "Caption: Beneficial and harmful signals are paired per evaluation slice to make tradeoffs visible at a glance.",
                "",
                "## Publication Trend Panels",
                "",
                f"![Publication trend panels]({trend_panels_svg})",
                "",
                "Caption: Figure Set A uses a shared normalized scale and aligned framing so the panels behave like one publication figure.",
                "",
                "## Trend Callout Boards",
                "",
                f"![Trend callout boards]({trend_callouts_svg})",
                "",
                "Caption: Figure Set B mirrors the plotted slices with exact values so the callout boards and panel figure can be read together.",
                "",
                "## Top Normalized Benchmark Scores",
                "",
                "| Experiment | Module | Score | Metrics |",
                "|---|---|---:|---:|",
            ]
        )
        for row in top_benchmark:
            figure_lines.append(
                f"| `{row['experiment_id']}` | `{row['module']}` | {row['normalized_score']:.4f} | {row['num_metrics']} |"
            )
        figure_lines.extend(
            [
                "",
                "## Fidelity Split",
                "",
                f"![Fidelity band split]({fidelity_svg})",
                "",
                "## Artifact Tables",
                "",
                f"- Module matrix CSV: `{module_matrix_csv}`",
                f"- Benchmark experiment scores CSV: `{benchmark_csv}`",
                f"- Benchmark family scores CSV: `{benchmark_family_csv}`",
                f"- Benchmark family-group scores CSV: `{benchmark_family_group_csv}`",
                f"- Section completion CSV: `{section_csv}`",
                f"- Survey-map provenance CSV: `{survey_map_csv}`",
                f"- Mechanism provenance CSV: `{mechanism_csv}`",
            ]
        )
        figures_tables_path.write_text("\n".join(figure_lines) + "\n")

        return {
            "module_count": len(module_rows),
            "benchmark_ranked_count": len(benchmark["experiment_scores"]),
            "benchmark_family_count": len(benchmark["family_scores"]),
            "benchmark_family_group_count": len(benchmark["family_group_scores"]),
            "mechanism_provenance_count": len(mechanism_rows),
            "survey_map_provenance_count": len(survey_map_rows),
            "stale_report_count": report_index["stale_report_count"],
            "output_paths": {
                "fidelity_matrix": str(fidelity_matrix_path),
                "module_matrix": str(module_matrix_path),
                "figures_and_tables": str(figures_tables_path),
                "mechanism_provenance": str(mechanism_provenance_path),
                "resource_provenance": str(resource_provenance_path),
                "survey_map_provenance": str(survey_map_provenance_path),
                "module_matrix_csv": str(module_matrix_csv),
                "benchmark_csv": str(benchmark_csv),
                "benchmark_family_csv": str(benchmark_family_csv),
                "benchmark_family_group_csv": str(benchmark_family_group_csv),
                "section_csv": str(section_csv),
                "survey_map_csv": str(survey_map_csv),
                "mechanism_csv": str(mechanism_csv),
                "section_svg": str(section_svg),
                "benchmark_svg": str(benchmark_svg),
                "fidelity_svg": str(fidelity_svg),
                "adaptation_svg": str(adaptation_svg),
                "retrieval_svg": str(retrieval_svg),
                "risk_svg": str(risk_svg),
                "trend_panels_svg": str(trend_panels_svg),
                "trend_callouts_svg": str(trend_callouts_svg),
            },
        }
