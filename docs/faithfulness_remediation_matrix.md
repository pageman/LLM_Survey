# Faithfulness Remediation Matrix

This is the first-pass remediation queue for `LLM_Survey`. It prioritizes actual modules that are most likely to benefit from code improvements, trace expansion, evaluation tightening, or provenance refactoring.

The matrix is intentionally selective. It is not a row for every module yet. It is the initial high-value queue.

## Legend

- `Mechanism Strength`
  - `A`: strong mechanism exposure
  - `B`: good lite mechanism
  - `C`: shallow or compressed
  - `D`: mostly summary or naming surface
- `Derivation Quality`
  - `A`: direct donor or exact local primitive trace
  - `B`: good local derivation but still broad
  - `C`: plausible but under-explained
  - `D`: opaque
- `Publication Risk`
  - `Low`, `Medium`, `High`
- `Priority`
  - `P0`: immediate
  - `P1`: next wave
  - `P2`: later

## First-Pass Queue

| Module | Category | Mechanism Strength | Derivation Quality | Publication Risk | Priority | Main Risk | Recommended Refactor |
|---|---|---:|---:|---|---|---|---|
| `utilization.tool_use_stub` | utilization | C | B | High | P0 | Tool use may read as a label-selection demo rather than an actual control-flow mechanism. | Add explicit planner state, tool-call arguments, observation handling, and failure branches; emit step traces in the JSON artifact. |
| `utilization.planning_agent_demo` | utilization | C | C | High | P0 | Planning behavior may be overcompressed relative to the topic name. | Refactor around explicit plan states, branch selection, replanning triggers, and action/evidence traces. |
| `utilization.self_consistency_demo` | utilization | C | C | Medium | P0 | Self-consistency may reduce to repeated sampling without enough aggregation logic. | Make sample set, voting rule, disagreement rate, and confidence aggregation explicit. |
| `utilization.react_demo` | utilization | B | B | Medium | P0 | ReAct-lite can still underexpose reasoning/tool interplay. | Separate reasoning token trace, action trace, and observation integration; add per-step state snapshots. |
| `utilization.world_model_planning_demo` | utilization | C | C | High | P0 | "World model" may overclaim if state evolution is too shallow. | Add explicit latent/state transition objects and plan scoring across hypothetical next states. |
| `utilization.program_aided_reasoning_demo` | utilization | B | C | Medium | P0 | Program-aided reasoning may look like formatted prompting instead of execution-grounded reasoning. | Add program synthesis, execution result, error handling, and answer reconciliation traces. |
| `adaptation.constitutional_ai_demo` | adaptation | C | C | High | P0 | Constitutional AI is easy to overname relative to a simple rewrite loop. | Make constitution rules explicit, show critique pass, revision pass, and policy/rule application outputs. |
| `adaptation.ppo_rlhf_toy` | adaptation | C | C | High | P0 | PPO-style alignment can appear method-faithful while hiding actual update logic. | Expose reward signal, clipped objective surrogate, policy delta, and acceptance statistics in artifacts. |
| `adaptation.rejection_sampling_demo` | adaptation | C | B | Medium | P0 | Rejection sampling may be too scalar and not enough sample-policy behavior. | Emit full sample pool statistics, rejection thresholds, accepted candidate rationale, and efficiency metrics. |
| `adaptation.preference_tuning` | adaptation | C | C | Medium | P0 | Preference optimization may blur distinctions with DPO/RM updates. | Separate scoring model, preference pair handling, and update rule traces more explicitly. |
| `evaluation.truthfulness_vs_helpfulness_eval` | evaluation | C | C | High | P0 | One of the easiest modules to overread beyond the strength of its probes. | Add item-level conflict cases, separate helpfulness and truthfulness views, and failure-mode labels. |
| `evaluation.verifier_eval` | evaluation | C | C | High | P0 | Verifier-style evaluation risks being too shallow if it is just answer checking. | Add proposal generation, verifier signal, acceptance thresholding, and false-accept/false-reject accounting. |
| `evaluation.reward_model_overoptimization_demo` | evaluation | C | C | Medium | P0 | Overoptimization is subtle and easy to flatten into one score. | Add per-step optimization trajectory, reward divergence, and proxy-vs-target mismatch traces. |
| `evaluation.jailbreak_transfer_eval` | evaluation | C | B | Medium | P0 | Transfer may read as a binary success probe without attack family structure. | Group prompts by attack family, target family, and transfer direction; emit a transfer matrix artifact. |
| `crosscutting.capability_vs_alignment_tradeoff_demo` | crosscutting | C | C | High | P0 | Crosscutting tradeoff claims can overstate a tiny local proxy. | Make both axes explicit, expose item-level tradeoff frontiers, and avoid one collapsed score. |
| `crosscutting.memorization_vs_generalization_demo` | crosscutting | C | C | High | P0 | Memorization/generalization is too subtle for one coarse toy split. | Split exact-match, near-copy, paraphrase, and transfer buckets; report all separately. |
| `crosscutting.safety_reasoning_tradeoff_demo` | crosscutting | C | C | Medium | P1 | May conflate refusal behavior with reasoning degradation. | Add separate reasoning quality and refusal-safety trajectories instead of one joint view. |
| `evaluation.calibration_eval` | evaluation | B | C | Medium | P1 | Calibration can remain overly compressed if bucket structure is too light. | Add bucketed reliability tables and per-bin residuals to the artifact. |
| `evaluation.hallucination_checks` | evaluation | B | C | Medium | P1 | Supported-rate alone can hide failure patterns. | Add evidence-missing, evidence-conflict, and unsupported generation categories. |
| `evaluation.privacy_leakage_eval` | evaluation | C | C | High | P1 | Privacy leakage claims need clearer threat-model boundaries. | Add attack type labels, memorized span classes, and risk interpretations per attack family. |
| `evaluation.code_eval_demo` | evaluation | B | C | Medium | P1 | Code evaluation can be too pass/fail oriented. | Separate syntax validity, semantic correctness, and repairability metrics. |
| `adaptation.reward_model_toy` | adaptation | B | C | Medium | P1 | Stronger than many toy modules, but still under-documented in derivation. | Add clearer reward-feature lineage and pairwise scoring traces. |
| `adaptation.instruction_tuning` | adaptation | B | C | Medium | P1 | Instruction tuning may need stronger data/target split exposition. | Add instruction-source split summaries and item-level before/after behavior. |
| `pretraining.dedup_demo` | pretraining | B | C | Medium | P1 | Good topic, but curation logic may be too narrow. | Add exact duplicate, near duplicate, and semantic duplicate modes with separate counts. |
| `pretraining.contamination_demo` | pretraining | B | C | Medium | P1 | Train-test contamination should be more structurally explicit. | Add contamination type labels and pre/post contamination difficulty deltas. |
| `pretraining.domain_coverage_demo` | pretraining | C | C | Medium | P1 | Current notes suggest topic fit is stronger than method fit. | Refactor to domain-bucket sampling and transfer sensitivity rather than generic corpus proportions. |
| `pretraining.toxicity_filter_demo` | pretraining | C | C | Medium | P1 | Filter quality may be too heuristic. | Add threshold sweep, precision/recall-style proxy accounting, and retained-corpus tradeoff curves. |
| `training.optimizer_ablation_dashboard` | training | C | C | Medium | P1 | This is labeled as mechanism-level but behaves dashboard-like. | Convert it into a true ablation runner with comparable trajectories, not only summary rows. |
| `systems.kv_cache_fragmentation_demo` | systems | B | C | Medium | P1 | Fragmentation claims need clearer serving-state interpretation. | Add per-step allocation maps, fragmentation ratios, and batch-shape sensitivity. |
| `architecture.multilingual_architecture_demo` | architecture | C | C | Medium | P1 | Architecture distinction may be weaker than the topic label implies. | Make embedding sharing, vocabulary split, and transfer consequences explicit. |
| `architecture.code_model_architecture_demo` | architecture | C | C | Medium | P1 | Could still read like a themed transformer variant without enough architectural consequence. | Add code-structure biases and token-pattern effects rather than naming-only deltas. |
| `multilingual.prompting_demo` | multilingual | B | C | Medium | P1 | Stronger than many demos, but derivation and trace structure can improve. | Add cross-lingual prompt variants, response drift summaries, and prompt-format sensitivity. |
| `multilingual.transfer_eval` | multilingual | B | C | Medium | P1 | Transfer needs clearer directionality and asymmetry. | Add source-target matrix views and transfer asymmetry metrics. |
| `resources.public_model_registry` | resource/reporting | D | B | Low | P2 | Honest but mostly inventory-level. | Convert more rows into evidence-linked analytical columns instead of summary fields. |
| `resources.closed_model_registry` | resource/reporting | D | B | Low | P2 | Same as above. | Tighten row evidence and versioning semantics. |
| `reporting.fidelity_band_dashboard` | survey-map | D | B | Low | P2 | Useful, but classification logic should stay visibly interpretive. | Add explicit methodology note in the artifact and docs. |
| `benchmark.cross_section_summary` | survey-map | D | A | Medium | P2 | Cross-section summaries are useful but inherently easy to overread. | Keep as summary-only; add stricter "not a universal ranking" language in the artifact and docs. |

## Shared Refactor Themes

These improvements should be implemented as reusable internal patterns, not only one-off module patches.

### Trace Schema Expansion

Apply to:

- prompting
- tool-use
- planning
- preference optimization
- evaluator modules

Add:

- `trace`
- `steps`
- `decision_points`
- `failure_modes`
- `item_level_results`

### Derivation Metadata Expansion

Apply to:

- all repo-authored mechanism modules with `C` derivation quality

Add:

- `derivation_mode`
- `reused_local_primitives`
- `donor_conceptual_ancestor`

### Family-Local Evaluation Tightening

Apply to:

- benchmark and evaluation modules

Add:

- explicit metric-family declarations
- incompatible-comparison guards
- family-specific captions and notes

## Suggested Execution Order

1. `utilization.tool_use_stub`
2. `utilization.planning_agent_demo`
3. `utilization.self_consistency_demo`
4. `utilization.react_demo`
5. `adaptation.constitutional_ai_demo`
6. `adaptation.ppo_rlhf_toy`
7. `evaluation.truthfulness_vs_helpfulness_eval`
8. `evaluation.verifier_eval`
9. `crosscutting.capability_vs_alignment_tradeoff_demo`
10. `crosscutting.memorization_vs_generalization_demo`
11. `training.optimizer_ablation_dashboard`
12. `pretraining.domain_coverage_demo`

## Exit Criteria For The First Refactor Wave

This first wave is complete when:

- all `P0` modules above have a concrete code refactor
- each of those modules emits richer traces or item-level artifacts
- derivation notes are sharper for the updated modules
- publication docs do not overstate those modules before the refactor lands
- benchmark and provenance layers reflect the stronger semantics
