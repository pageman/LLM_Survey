# Full Survey Replan

This document revisits the scope of `LLM_Survey` against the structure of *A Survey of Large Language Models* and identifies the missing paper-scope modules needed for a more faithful NumPy-only implementation program.

## Scope Reset

The initial implementation target list was intentionally narrow: it focused on a runnable backbone for foundations, adaptation, retrieval, and long-context evaluation. That narrower list is now exhausted, but it does **not** mean the full survey paper has been implemented.

For re-planning, the correct interpretation is:

- keep one runnable educational artifact per major paper topic cluster
- expand the tracked module set to cover the broader survey structure
- continue using the same local report schema and benchmark views

## Paper Structure Used For Re-Planning

The survey is organized around these major areas:

- resources of LLMs
- pre-training
- adaptation tuning
- utilization
- capacity and evaluation
- practical guide for prompt design
- applications of LLMs
- conclusion and future directions

For implementation planning, the most important technical build areas are:

- pre-training
- adaptation tuning
- utilization
- capacity and evaluation
- selected prompt-design and application modules

## What Is Already Implemented

Current tracked implemented modules:

- `foundations.rnn_lm`
- `foundations.lstm_lm`
- `foundations.transformer_basics`
- `pretraining.scaling_laws`
- `pretraining.causal_lm`
- `utilization.retrieval`
- `utilization.rag`
- `evaluation.long_context`
- `evaluation.position_bias_eval`
- `evaluation.benchmark_harness`
- `adaptation.alignment_sft`
- `adaptation.finetuning`
- `adaptation.instruction_tuning`
- `adaptation.peft_lora`
- `adaptation.preference_tuning`

These are the core spine, but they leave substantial paper coverage gaps.

## Missing Topics By Survey Area

### 1. Foundations And Architecture Gaps

Still missing:

- `foundations.seq2seq_basics`
- `pretraining.masked_lm_demo`
- `pretraining.prefix_decoder_demo`

Why these matter:

- the survey covers multiple mainstream architectures, not just decoder-only transformers
- encoder-decoder and prefix-decoder patterns are part of the architectural picture
- masked objectives remain part of the historical and technical comparison

### 2. Pre-Training Data And Objective Gaps

Still missing:

- `pretraining.multi_token_prediction`
- `pretraining.tokenizer_demo`
- `pretraining.data_mixture_toy`
- `pretraining.data_curriculum_demo`
- `pretraining.dedup_demo`
- `pretraining.contamination_demo`
- `pretraining.data_quality_filter_demo`

Why these matter:

- the survey puts major emphasis on data quality, deduplication, contamination, data scheduling, and tokenizer choices
- these are currently underrepresented in the repo relative to the paper

### 3. Training Systems And Optimization Gaps

Still missing:

- `systems.pipeline_parallelism`
- `systems.optimization_stability_demo`
- `systems.kv_cache_toy`

Why these matter:

- the survey covers model training settings, acceleration, and efficiency techniques
- the repo currently has scaling-law coverage but not enough of the systems/training engineering side

### 4. Adaptation Gaps

Still missing:

- `adaptation.reward_model_toy`

Status:

- `finetuning`, `instruction_tuning`, `alignment_sft`, `preference_tuning`, and `peft_lora` are present
- the remaining major adaptation gap is an explicit reward-model style module

### 5. Utilization And Prompting Gaps

Still missing:

- `utilization.icl_demo`
- `utilization.cot_prompting`
- `utilization.self_consistency_demo`
- `utilization.tool_use_stub`
- `utilization.planning_agent_demo`

Why these matter:

- the survey’s utilization section goes well beyond retrieval and long-context behavior
- prompting, reasoning, tool use, and planning are major missing pieces

### 6. Evaluation Gaps

Still missing:

- `evaluation.calibration_eval`
- `evaluation.hallucination_checks`
- `evaluation.safety_eval`
- `evaluation.bias_eval`

Why these matter:

- the survey’s evaluation material includes broad capacity assessment, factuality, alignment, and risk-sensitive behaviors
- the repo has long-context and a general benchmark harness, but not enough specific evaluation probes

### 7. Application Gaps

Still missing:

- `applications.code_generation_demo`
- `applications.embodied_agent_stub`
- `applications.scientific_assistant_demo`

Why these matter:

- the paper includes an applications section and discusses representative domain use
- the repo currently has no application-layer demo modules at all

## Recommended Re-Plan

The next implementation waves should be:

### Wave A: Pre-Training Data Coverage

Build next:

- `pretraining.data_mixture_toy`
- `pretraining.dedup_demo`
- `pretraining.contamination_demo`
- `pretraining.data_quality_filter_demo`
- `pretraining.tokenizer_demo`

Reason:

- these are repeatedly emphasized in the survey and currently the largest obvious gap

### Wave B: Utilization Core

Build next:

- `utilization.icl_demo`
- `utilization.cot_prompting`
- `utilization.self_consistency_demo`
- `utilization.tool_use_stub`

Reason:

- these are central to how LLMs are actually used in the survey

### Wave C: Training/Systems

Build next:

- `systems.pipeline_parallelism`
- `systems.optimization_stability_demo`
- `systems.kv_cache_toy`

Reason:

- needed to reflect the survey’s training and deployment engineering content

### Wave D: Evaluation Expansion

Build next:

- `evaluation.hallucination_checks`
- `evaluation.calibration_eval`
- `evaluation.safety_eval`
- `evaluation.bias_eval`

Reason:

- this will make the benchmark harness materially more useful

### Wave E: Architecture And Applications

Build next:

- `foundations.seq2seq_basics`
- `pretraining.masked_lm_demo`
- `pretraining.prefix_decoder_demo`
- `applications.code_generation_demo`
- `applications.embodied_agent_stub`
- `applications.scientific_assistant_demo`

## Updated Completion Interpretation

Under the expanded paper-scope target list now tracked in the scoreboard generator, completion should be interpreted against the broader set of modules above, not the previous minimal spine.
