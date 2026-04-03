"""Architecture-focused lite demos."""

from .bidirectional_encoder_demo import BidirectionalEncoderDemo
from .code_model_architecture_demo import CodeModelArchitectureDemo
from .configuration_scaling_demo import ConfigurationScalingDemo
from .encoder_decoder_demo import EncoderDecoderDemo
from .moe_demo import MoEDemo
from .multilingual_architecture_demo import MultilingualArchitectureDemo
from .prefix_lm_demo import PrefixLMDemo

__all__ = [
    "BidirectionalEncoderDemo",
    "CodeModelArchitectureDemo",
    "ConfigurationScalingDemo",
    "EncoderDecoderDemo",
    "MoEDemo",
    "MultilingualArchitectureDemo",
    "PrefixLMDemo",
]
