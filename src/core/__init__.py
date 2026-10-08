"""
Fed-GRPO Core Module: Unified implementation for federated GRPO training
Combines server, client, and all training methods into a single module
"""

__version__ = "0.2.0"

from .server import FedGRPOServer
from .client import GRPOClient
from .compression import compress_delta, decompress_to_state_dict

__all__ = [
    "FedGRPOServer",
    "GRPOClient",
    "compress_delta",
    "decompress_to_state_dict",
]
