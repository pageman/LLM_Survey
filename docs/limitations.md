# Limitations

`LLM_Survey` is a local, NumPy-only educational reconstruction of the survey space. It is designed for inspectability and breadth, not for reproducing frontier-scale training systems.

## What The `124/124` Score Means

- every tracked paper-scope target in the current repository plan has a canonical artifact in the shared JSON/report-index/scoreboard pipeline
- all tracked targets are now represented by dedicated modules rather than a generic fallback generator
- the publication layer is generated locally from canonical reports, provenance rows, and figure/table assets

## What It Does Not Mean

- it does not mean every tracked target has the same fidelity as the strongest mechanism-level demos
- it does not mean the repository reproduces full-scale LLM training, evaluation, or deployment systems
- it does not mean every cited method in the survey has a paper-faithful implementation here

## Artifact Classes

- `mechanism-level`: runnable educational modules that expose a concrete algorithmic idea or evaluation behavior
- `survey-map`: analytical tables or dashboards that summarize coverage, provenance, or cross-section comparisons rather than implement a standalone mechanism
- `resource/reporting`: publication-oriented inventories, provenance tables, and reporting summaries that organize the implementation surface

These classes are intentionally different. A mechanism-level module should be read as code-first technical reconstruction; a survey-map or resource/reporting artifact should be read as evidence-backed analytical scaffolding around that code.

## Fidelity Notes

- mechanism-level artifacts vary from stronger donor-derived mini-implementations to repo-authored lite mechanisms
- survey-map and resource/reporting artifacts are publication surfaces, not claims of method replication
- provenance tables identify which rows are donor-derived, repo-authored, or computed-summary artifacts through explicit confidence tags

## Recommended Next Fidelity Work

- expand explicit donor/source extraction notes across more mechanism-level modules
- keep converting analytical tables from descriptive summaries into row-wise evidence-backed views
- improve figure normalization and publication captions without changing the underlying report schema
