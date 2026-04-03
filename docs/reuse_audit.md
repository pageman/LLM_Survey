# Reuse Audit

This file will track how code from `sutskever-30-implementations` is reused in `LLM_Survey`.

## Rules

- Prefer extracting core logic rather than copying notebook cells wholesale.
- Record the exact donor file for every migrated implementation.
- Record whether the target is a direct port, adaptation, or fresh implementation.

## Initial Donor Set

| Donor File | Target Area | Planned Action | Status |
|---|---|---|---|
| `/Users/hifi/sutskever-30-implementations/02_char_rnn_karpathy.ipynb` | recurrent language modeling | extract reusable RNN LM core | Planned |
| `/Users/hifi/sutskever-30-implementations/03_lstm_understanding.ipynb` | LSTM language modeling | extract LSTM cell and training helpers | Planned |
| `/Users/hifi/sutskever-30-implementations/13_attention_is_all_you_need.ipynb` | transformer foundations | extract attention and transformer primitives | Planned |
| `/Users/hifi/sutskever-30-implementations/14_bahdanau_attention.ipynb` | seq2seq attention | extract additive attention baseline | Planned |
| `/Users/hifi/sutskever-30-implementations/22_scaling_laws.ipynb` | scaling laws | port into repeatable script/module | Planned |
| `/Users/hifi/sutskever-30-implementations/27_multi_token_prediction.ipynb` | advanced pre-training objective | port and simplify for module use | Planned |
| `/Users/hifi/sutskever-30-implementations/28_dense_passage_retrieval.ipynb` | retrieval | extract retriever scoring/eval | Planned |
| `/Users/hifi/sutskever-30-implementations/29_rag.ipynb` | RAG | extract retrieval-generation pipeline | Planned |
| `/Users/hifi/sutskever-30-implementations/30_lost_in_middle.ipynb` | long-context eval | port position-bias experiments | Planned |
