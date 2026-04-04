# NumPy Compliance Program

This note defines the internal standards used for the repository's NumPy-only
compliance work. It is intentionally short and operational.

## Stage Definitions

### Stage 1: Clean Layer

- use typed payload rows for high-use structured artifacts
- add explicit shape/return docstrings where module behavior is not obvious
- use shared local aliases and protocols where they materially clarify the API

### Stage 2: Standard Professional NumPy

- prefer array-native operations over manual row-wise logic
- keep attention masks semantically consistent:
  - `0.0` means visible
  - non-zero means blocked
  - blocked entries are converted inside the attention path
- prefer broadcasting and vectorization over repeated Python loops

### Stage 3: Power User NumPy

Use `np.einsum` only when it improves tensor readability.

Preferred cases:
- projection or routing logic with named tensor roles
- attention score and value aggregation paths
- places where `@` plus multiple transposes obscures intent

Avoid:
- scalar/reporting logic
- simple one-step matrix products where `@` is already clearer

Status note:
- the Stage 3 pass is intentionally selective; once only `@`-clear sites remain,
  the stage should be frozen as complete rather than extended for style alone

Stage 3 completion decision:
- complete
- the final audit reviewed the remaining tensor-heavy `@` sites in core and
  module code
- the remaining uses are intentional in places where `@` is clearer than
  `einsum`, including simple feed-forward products, RNN/LSTM affine steps,
  retrieval similarity, and lightweight aggregation helpers
- no further ceremonial `einsum` rewrites should be added

### Stage 4: Algorithmic Interventions

- add advanced educational mechanisms as explicit modules or experiments
- tie them back to the core helpers they validate
- do not hide advanced behavior behind legacy names

### Stage 5: Experimental Layer

- keep experimental sparse/quantized/efficient attention work as a coherent suite
- compare families with shared metrics where possible
- mark clearly that these are educational simulations, not production kernels

## Current Interpretation

The goal is not to maximize novelty. The goal is to make the repository read
like a professional NumPy-first educational codebase whose abstractions,
numerics, and experiments are internally consistent.
