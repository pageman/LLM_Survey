"""Systems and efficiency modules."""

from .attention_length_stability_demo import AttentionLengthStabilityDemo
from .advanced_attention_suite_demo import AdvancedAttentionSuiteDemo
from .flash_attention_comparison_demo import FlashAttentionComparisonDemo
from .block_sparse_attention_demo import BlockSparseAttentionDemo
from .efficient_attention_suite_board_demo import EfficientAttentionSuiteBoardDemo
from .scale_stability_demo import ScaleStabilityDemo
from .inference_batching_demo import InferenceBatchingDemo
from .kv_cache_fragmentation_demo import KVCacheFragmentationDemo
from .kv_cache_toy import KVCacheToy
from .flash_block_sweep_demo import FlashBlockSweepDemo
from .kv_long_context_board_demo import KVLongContextBoardDemo
from .long_context_flash_demo import LongContextFlashDemo
from .numeric_stability_demo import NumericStabilityDemo
from .optimization_stability_demo import OptimizationStabilityDemo
from .pipeline_parallelism import PipelineParallelismDemo
from .quantization_sim_demo import QuantizationSimDemo
from .ring_attention_demo import RingAttentionDemo
from .sliding_window_kv_demo import SlidingWindowKVDemo
from .sparse_family_board_demo import SparseFamilyBoardDemo
from .sparse_dense_benchmark_demo import SparseDenseBenchmarkDemo
from .sparse_attention_demo import SparseAttentionDemo
from .speculative_decoding_demo import SpeculativeDecodingDemo

__all__ = [
    "AdvancedAttentionSuiteDemo",
    "AttentionLengthStabilityDemo",
    "BlockSparseAttentionDemo",
    "EfficientAttentionSuiteBoardDemo",
    "FlashAttentionComparisonDemo",
    "FlashBlockSweepDemo",
    "InferenceBatchingDemo",
    "KVCacheFragmentationDemo",
    "KVCacheToy",
    "KVLongContextBoardDemo",
    "LongContextFlashDemo",
    "NumericStabilityDemo",
    "OptimizationStabilityDemo",
    "PipelineParallelismDemo",
    "QuantizationSimDemo",
    "ScaleStabilityDemo",
    "RingAttentionDemo",
    "SlidingWindowKVDemo",
    "SparseFamilyBoardDemo",
    "SparseDenseBenchmarkDemo",
    "SparseAttentionDemo",
    "SpeculativeDecodingDemo",
]
