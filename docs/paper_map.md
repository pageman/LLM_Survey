# Paper Map

This document maps the survey's technical areas to implementation targets in this repository, identifies reuse opportunities from `sutskever-30-implementations`, and defines what should happen next.

## Working Interpretation

The survey is broad, so "re-implement everything" must be translated into an executable standard. The practical interpretation for this repository is:

- implement representative runnable artifacts for each major technical area in the survey
- preserve conceptual coverage over exhaustive literature replication
- reuse the local source repo wherever the underlying mechanism already exists
- isolate gaps and fill them with minimal new code

## Coverage Levels

- `Direct reuse`: existing source implementation is close enough to migrate with small refactors.
- `Partial reuse`: source implementation covers some core mechanism but needs meaningful extension.
- `Net-new`: little or no useful source coverage exists; write a fresh minimal implementation.

## Top-Level Coverage Matrix

| Survey Area | Target Modules | Source Reuse | Coverage Level | Next Action |
|---|---|---|---|---|
| Background and evolution of language modeling | `rnn_lm`, `lstm_lm`, `seq2seq_basics`, `transformer_basics` | `02`, `03`, `14`, `13` | Direct reuse | Extract model primitives into `src/core/` |
| Transformer architecture and self-attention | `attention`, `transformer_basics`, `causal_lm` | `13`, `14`, `06`, `08` | Partial reuse | Refactor attention code and add decoder-only path |
| Pre-training objectives | `causal_lm`, `masked_lm_demo`, `multi_token_prediction` | `27`, `02`, `13` | Partial reuse | Implement explicit LM objective modules |
| Scaling laws and compute tradeoffs | `scaling_laws`, `data_scaling_demo` | `22` | Direct reuse | Convert notebook logic into scripts and tests |
| Data engineering and corpus quality | `data_mixture_toy`, `dedup_demo`, `contamination_demo` | none obvious | Net-new | Design synthetic corpora and metrics |
| Adaptation tuning and fine-tuning | `finetuning`, `instruction_tuning`, `prompt_tuning_demo` | weak indirect reuse | Net-new | Define toy datasets and training loops |
| Parameter-efficient adaptation | `peft_lora`, `adapter_demo` | none obvious | Net-new | Implement matrix low-rank updates |
| Alignment and preference optimization | `alignment_sft`, `dpo_toy`, `rm_toy` | none obvious | Net-new | Build tiny preference datasets and trainers |
| In-context learning and prompting | `icl_demo`, `cot_prompting`, `prompt_eval` | `02`, `13` as base LMs only | Partial reuse | Add prompt templates and evaluation harness |
| Retrieval-augmented generation | `retrieval`, `rag`, `hybrid_retrieval_demo` | `28`, `29` | Direct reuse | Extract retriever/generator interfaces |
| Long-context behavior and position bias | `long_context`, `position_bias_eval` | `30` | Direct reuse | Port evaluation into repeatable scripts |
| Systems for efficient training/inference | `pipeline_parallelism`, `multi_token_prediction`, `kv_cache_toy` | `09`, `27` | Partial reuse | Split systems demos into focused modules |
| Evaluation and benchmarking | `benchmark_harness`, `calibration`, `hallucination_checks` | scattered only | Net-new | Define common interfaces and metrics |
| External tools / agents / augmentation | `tool_use_stub`, `agent_loop_demo` | `29` adjacent only | Net-new | Keep deliberately small and explicit |

## Donor Notebook Map

| Source File | Best Use In `LLM_Survey` | Priority |
|---|---|---|
| `/Users/hifi/sutskever-30-implementations/02_char_rnn_karpathy.ipynb` | character-level causal LM core, sequence sampling, recurrent training basics | High |
| `/Users/hifi/sutskever-30-implementations/03_lstm_understanding.ipynb` | LSTM cell implementation and recurrent training mechanics | High |
| `/Users/hifi/sutskever-30-implementations/06_pointer_networks.ipynb` | attention-as-selection patterns and decoder interfaces | Medium |
| `/Users/hifi/sutskever-30-implementations/08_seq2seq_for_sets.ipynb` | encoder-decoder scaffolding and attention-based sequence processing | Medium |
| `/Users/hifi/sutskever-30-implementations/09_gpipe.ipynb` | pipeline scheduling and systems pedagogy | Medium |
| `/Users/hifi/sutskever-30-implementations/13_attention_is_all_you_need.ipynb` | self-attention, multi-head attention, positional encoding, transformer blocks | Highest |
| `/Users/hifi/sutskever-30-implementations/14_bahdanau_attention.ipynb` | additive attention and seq2seq alignment visualization | High |
| `/Users/hifi/sutskever-30-implementations/22_scaling_laws.ipynb` | power-law fitting and compute-quality tradeoff demos | Highest |
| `/Users/hifi/sutskever-30-implementations/27_multi_token_prediction.ipynb` | multi-token objective demo and inference-speed narrative | High |
| `/Users/hifi/sutskever-30-implementations/28_dense_passage_retrieval.ipynb` | retriever embeddings, scoring, negative sampling demos | Highest |
| `/Users/hifi/sutskever-30-implementations/29_rag.ipynb` | retrieval-generation orchestration | Highest |
| `/Users/hifi/sutskever-30-implementations/30_lost_in_middle.ipynb` | long-context evaluation harness and position-bias experiments | Highest |

## What Should Happen Next

The immediate next phase should be treated as an engineering program, not a loose set of ideas.

### Phase 0: Repository Contract

Goal:

- freeze the paper interpretation
- define coverage boundaries
- define what "implemented" means per survey topic

Actions:

1. Keep this file current as the source of truth.
2. Add `docs/reuse_audit.md` with one row per donor notebook.
3. Decide the minimum runnable artifact per topic:
   - one module
   - one experiment entrypoint
   - one output artifact
   - one smoke test

Deliverable:

- a stable coverage contract that prevents scope drift

### Phase 1: Reuse Audit And Code Extraction

Goal:

- turn notebook-heavy donor code into stable importable components

Actions:

1. Inspect the highest-priority donor notebooks cell by cell.
2. Separate reusable logic from visualization-only and exposition-only cells.
3. Move common math and helper code into:
   - `src/core/attention.py`
   - `src/core/rnn.py`
   - `src/core/lstm.py`
   - `src/core/transformer.py`
   - `src/core/data.py`
   - `src/core/metrics.py`
4. Normalize naming, interfaces, and output shapes.
5. Add smoke tests before extending behavior.

Why this must happen first:

- the source repo is strong conceptually but notebook-oriented
- copying notebook cells repeatedly will create drift and duplication
- most later modules depend on a clean shared core

Deliverable:

- importable reusable core with provenance notes

### Phase 2: Build The Foundational Spine

Goal:

- cover the survey's historical and architectural backbone

Priority modules:

1. `rnn_lm`
2. `lstm_lm`
3. `attention`
4. `seq2seq_basics`
5. `transformer_basics`
6. `causal_lm`
7. `scaling_laws`

Actions:

1. Port the smallest working recurrent LM from notebook form.
2. Add LSTM-based sequence modeling and generation.
3. Refactor additive and dot-product attention into a common API.
4. Implement a decoder-only transformer path even if the source repo is encoder-decoder oriented.
5. Build a toy next-token training loop and synthetic token corpus.
6. Repackage scaling-law demos as `experiments/scaling_laws.py`.

Deliverable:

- the repo can already demonstrate how modern LLMs emerged from prior sequence modeling work

### Phase 3: Add The Survey-Specific LLM Core

Goal:

- cover the technical center of modern LLM practice

Priority modules:

1. `masked_lm_demo`
2. `multi_token_prediction`
3. `data_mixture_toy`
4. `dedup_demo`
5. `contamination_demo`

Actions:

1. Add explicit objective modules for causal LM and MLM.
2. Port and tighten the multi-token prediction donor implementation.
3. Build synthetic corpora showing:
   - duplicate data inflation
   - contamination leakage
   - curriculum and mixture effects
4. Save comparable plots for all data/pre-training experiments.

Reason:

- the survey spends substantial effort on pre-training choices, not just architectures
- these topics are under-covered in the donor repo and must be built deliberately

Deliverable:

- runnable demonstrations of pre-training objective and data-quality tradeoffs

### Phase 4: Adaptation, Instruction Tuning, And Alignment

Goal:

- cover what happens after pre-training

Priority modules:

1. `finetuning`
2. `instruction_tuning`
3. `prompt_tuning_demo`
4. `peft_lora`
5. `alignment_sft`
6. `rm_toy`
7. `dpo_toy`

Actions:

1. Define tiny supervised instruction datasets.
2. Implement a generic fine-tuning loop on top of `causal_lm`.
3. Implement low-rank adaptation with explicit frozen-base semantics.
4. Add a tiny reward-model style pairwise scorer.
5. Add a toy preference optimization path.

Important constraint:

- these should stay small and educational; no fake industrial RLHF stack

Deliverable:

- a minimal but coherent adaptation-and-alignment section

### Phase 5: Retrieval, Augmentation, And Long Context

Goal:

- cover the most reusable practical LLM augmentations

Priority modules:

1. `retrieval`
2. `rag`
3. `hybrid_retrieval_demo`
4. `long_context`
5. `position_bias_eval`

Actions:

1. Extract retriever scoring and indexing logic from `28_dense_passage_retrieval.ipynb`.
2. Extract the retrieval-generation orchestration from `29_rag.ipynb`.
3. Standardize document store and query interfaces.
4. Port position-bias experiments from `30_lost_in_middle.ipynb`.
5. Make results reproducible through scripts, not notebook-only cells.

Deliverable:

- a strong practical section with immediate educational value

### Phase 6: Systems And Efficiency

Goal:

- cover scaling-adjacent systems ideas without over-building infrastructure

Priority modules:

1. `pipeline_parallelism`
2. `multi_token_prediction`
3. `kv_cache_toy`
4. `batching_latency_demo`

Actions:

1. Refactor GPipe-style scheduling demos into explicit simulation utilities.
2. Tie multi-token prediction to throughput narratives.
3. Add a small key-value cache simulation to explain autoregressive inference speed.
4. Measure latency and throughput on tiny workloads.

Deliverable:

- pedagogical systems modules that explain inference and training tradeoffs

### Phase 7: Unified Evaluation Harness

Goal:

- stop every module from inventing its own metrics and artifact style

Priority modules:

1. `benchmark_harness`
2. `calibration`
3. `hallucination_checks`
4. `reporting`

Actions:

1. Define a common result schema.
2. Standardize metric computation and artifact saving.
3. Add one command that regenerates all artifacts for implemented modules.
4. Ensure each experiment emits machine-readable output.

Deliverable:

- one evaluation substrate used across the repository

## Recommended Build Order

If working linearly, the best sequence is:

1. `src/core/attention.py`
2. `src/core/rnn.py`
3. `src/core/lstm.py`
4. `src/core/transformer.py`
5. `src/modules/foundations/rnn_lm.py`
6. `src/modules/foundations/lstm_lm.py`
7. `src/modules/foundations/transformer_basics.py`
8. `src/modules/pretraining/causal_lm.py`
9. `src/modules/pretraining/scaling_laws.py`
10. `src/modules/utilization/retrieval.py`
11. `src/modules/utilization/rag.py`
12. `src/modules/utilization/long_context.py`
13. `src/modules/adaptation/finetuning.py`
14. `src/modules/adaptation/peft_lora.py`
15. `src/modules/adaptation/dpo_toy.py`
16. `src/modules/evaluation/benchmark_harness.py`

## Risks

### Risk 1: Scope Explosion

The survey is much broader than one implementation track. Mitigation:

- require one minimal runnable artifact per topic
- prioritize representative mechanisms over paper-by-paper cloning

### Risk 2: Notebook Code Drift

If notebook code is copied ad hoc, the repo will become inconsistent quickly. Mitigation:

- centralize extraction into `src/core/`
- record provenance in `docs/reuse_audit.md`

### Risk 3: Fake Completeness

It is easy to claim coverage without meaningful implementations. Mitigation:

- require scripts, tests, and artifacts
- mark incomplete topics explicitly

### Risk 4: Overbuilding Alignment/Agents

These topics can absorb unlimited time. Mitigation:

- keep toy-scale and pedagogical
- avoid infrastructure that does not support the paper map

## Completion Standard Per Module

A module should only be marked complete when all of the following exist:

- one importable implementation in `src/modules/`
- one runnable experiment entrypoint
- one smoke test
- one generated artifact
- one documented provenance or net-new rationale

## Near-Term Deliverables

The next concrete deliverables after this document should be:

1. `docs/reuse_audit.md`
2. `src/core/attention.py`
3. `src/core/rnn.py`
4. `src/core/lstm.py`
5. `src/core/transformer.py`
6. first experiments for `scaling_laws`, `retrieval`, and `long_context`
