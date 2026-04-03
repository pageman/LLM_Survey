"""Reusable core primitives extracted from donor notebook implementations."""

from .attention import (
    BahdanauAttention,
    MultiHeadAttention,
    create_causal_mask,
    scaled_dot_product_attention,
    softmax,
)
from .data import ToyTokenizer, hashed_bow_embedding, make_next_token_pairs, one_hot_sequence
from .lstm import LSTMCell, LSTMSequenceModel, sigmoid
from .metrics import compute_retrieval_metrics, cross_entropy_from_probs, perplexity_from_losses, top_k_accuracy
from .reporting import SCHEMA_VERSION, build_report, write_report
from .rnn import VanillaRNNLanguageModel, one_hot
from .transformer import TransformerBlock, feed_forward, layer_norm, positional_encoding

__all__ = [
    "BahdanauAttention",
    "LSTMCell",
    "LSTMSequenceModel",
    "MultiHeadAttention",
    "SCHEMA_VERSION",
    "TransformerBlock",
    "ToyTokenizer",
    "VanillaRNNLanguageModel",
    "build_report",
    "compute_retrieval_metrics",
    "create_causal_mask",
    "cross_entropy_from_probs",
    "feed_forward",
    "hashed_bow_embedding",
    "layer_norm",
    "make_next_token_pairs",
    "one_hot",
    "one_hot_sequence",
    "perplexity_from_losses",
    "positional_encoding",
    "scaled_dot_product_attention",
    "sigmoid",
    "softmax",
    "top_k_accuracy",
    "write_report",
]
