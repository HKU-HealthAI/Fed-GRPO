"""
Fed-GRPO Server: Central server responsible for global model aggregation
"""

import torch
import torch.nn as nn
import logging
from typing import List, Dict, Any
from utils.experiment import is_main_process

logger = logging.getLogger(__name__)


class FedGRPOServer:
    """
    Federated GRPO Central Server

    Aggregation: signal_weighted (Fed-GRPO) | sample_weighted | uniform (FedAvg).

    Memory Optimization: Reference model weights are stored on CPU and only loaded
    to GPU when needed, reducing GPU memory usage significantly for large models.

    Reference Model Update (DeepSeek-R1-Zero style):
    - Reference model is updated periodically by copying current global model weights
    - Update interval is configurable via ref_model_update_interval parameter
    """

    def __init__(
        self,
        global_model: nn.Module = None,
        model: nn.Module = None,
        aggregation_method: str = 'signal_weighted',
        ref_model_update_interval: int = 10,
        ref_model_mixup_alpha: float = 1.0,
        device: str = "cuda",
        use_ref_model: bool = True,
        use_effective_samples: bool = True,
        use_update_normalization: bool = True,
        calibration_alpha: float = 0.5
    ):
        """
        Initialize Fed-GRPO Server

        Args:
            global_model: Global model (preferred parameter name)
            model: Global model (legacy parameter name, for backward compatibility)
            aggregation_method: One of 'signal_weighted', 'sample_weighted', 'uniform'.
                   signal_weighted: w_k = s_k / S (Fed-GRPO, signal strength).
                   sample_weighted: w_k = n_k / N (FedAvg by sample count).
                   uniform: w_k = 1 / n (FedAvg equal weight, n = num_clients).
            ref_model_update_interval: Update reference model every N rounds (DeepSeek-R1-Zero style)
            ref_model_mixup_alpha: Mixup coefficient for reference model update.
                                   Formula: π_ref = α * π_θ + (1 - α) * π_ref_prev
                                   1.0 = direct copy (DeepSeek-R1-Zero style)
                                   <1.0 = EMA-like smooth update (e.g., 0.5 = 50% new + 50% old)
            device: Device
            use_ref_model: Whether to create reference model (set False when beta_kl=0)
            use_effective_samples: Use effective sample count (discount zero-variance prompts)
            use_update_normalization: Normalize client update magnitudes before aggregation
            calibration_alpha: Mixing coefficient for effective_samples vs signal_strength.
                               w_k = α * (e_k / E) + (1 - α) * (σ_k / Σσ)
                               0.0 = pure signal strength, 1.0 = pure effective samples.
        """
        # Use Accelerator's device if available, otherwise use specified device
        try:
            from accelerate import PartialState
            self.device = PartialState().device
        except (ImportError, RuntimeError) as e:
            if is_main_process():
                logger.debug(f"PartialState not available, using specified device: {e}")
            self.device = device

        # Handle backward compatibility for model parameter
        if global_model is None and model is not None:
            global_model = model
        if global_model is None:
            raise ValueError("Must provide global_model or model")

        self.global_model = global_model.to(self.device)
        self.aggregation_method = aggregation_method

        # Memory optimization: Store reference model weights on CPU instead of GPU
        # This saves significant GPU memory for large models
        self.use_ref_model = use_ref_model
        self._ref_state_dict = None  # Store on CPU

        if use_ref_model:
            # Store initial reference weights on CPU
            self._ref_state_dict = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
            if is_main_process():
                logger.info("Reference model weights stored on CPU (memory optimization enabled)")

        # Reference model update interval (DeepSeek-R1-Zero style)
        self.ref_model_update_interval = ref_model_update_interval
        
        # Mixup alpha for reference model update
        # Formula: π_ref = α * π_θ + (1 - α) * π_ref_prev
        # 1.0 = direct copy (DeepSeek-R1-Zero), <1.0 = EMA-like smooth update
        self.ref_model_mixup_alpha = ref_model_mixup_alpha

        self.round = 0

        # Global reward calibration stats (updated each round in aggregate())
        self.global_reward_mean = None  # μ_global
        self.global_reward_std = None   # σ_global

        # Cross-client reward calibration settings
        self.use_effective_samples = use_effective_samples
        self.use_update_normalization = use_update_normalization
        self.calibration_alpha = calibration_alpha

        if is_main_process():
            logger.info(f"Fed-GRPO Server initialized: aggregation={aggregation_method}, "
                       f"ref_update_interval={ref_model_update_interval}, ref_mixup_alpha={ref_model_mixup_alpha}, "
                       f"effective_samples={use_effective_samples}, update_norm={use_update_normalization}, "
                       f"calibration_alpha={calibration_alpha}")
    
    def get_ref_weights(self) -> Dict[str, torch.Tensor]:
        """Get reference model weights (returns CPU tensors to reduce GPU memory)"""
        if self._ref_state_dict is None:
            return None
        return {k: v.clone() for k, v in self._ref_state_dict.items()}
    
    def aggregate(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Aggregate client updates

        Args:
            client_weights: List of client model weights
            client_metrics: List of client metrics, each containing:
                - num_samples: Number of samples
                - reward_std: Reward standard deviation (signal strength for Fed-GRPO)
                - frac_reward_zero_std: Fraction of prompts with zero reward variance

        Returns:
            Aggregation statistics
        """
        self.round += 1
        num_clients = len(client_weights)

        # Compute weighting coefficients
        total_samples = sum(m['num_samples'] for m in client_metrics)
        total_signal = sum(m.get('reward_std', m.get('avg_std', 0.0)) for m in client_metrics)

        if self.aggregation_method == 'uniform':
            # FedAvg equal weight: w_k = 1 / n
            weights = [1.0 / num_clients] * num_clients
        elif self.aggregation_method == 'sample_weighted':
            # FedAvg by sample count: w_k = n_k / N
            weights = [m['num_samples'] / total_samples for m in client_metrics]
        else:
            # signal_weighted (Fed-GRPO): calibrated weighting
            if total_signal < 1e-6:
                if is_main_process():
                    logger.warning("Total signal strength too small, falling back to sample_weighted")
                weights = [m['num_samples'] / total_samples for m in client_metrics]
            else:
                signal_strengths = [
                    m.get('reward_std', m.get('avg_std', 0.0)) for m in client_metrics
                ]

                if self.use_effective_samples:
                    # Mixed weight: α * (e_k / E) + (1 - α) * (σ_k / Σσ)
                    effective_samples = [
                        m['num_samples'] * max(1.0 - m.get('frac_reward_zero_std', 0.0), 1e-6)
                        for m in client_metrics
                    ]
                    total_effective = sum(effective_samples)
                    alpha = self.calibration_alpha
                    weights = [
                        alpha * (e / total_effective) + (1.0 - alpha) * (s / total_signal)
                        for e, s in zip(effective_samples, signal_strengths)
                    ]
                else:
                    # Pure signal strength: w_k = σ_k / Σσ
                    weights = [s / total_signal for s in signal_strengths]
                # Normalize to sum to 1
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                else:
                    weights = [1.0 / num_clients] * num_clients

                if is_main_process() and self.use_effective_samples:
                    logger.info(
                        f"Round {self.round} effective_samples: "
                        f"{[f'{e:.1f}' for e in effective_samples]}"
                    )

        if is_main_process():
            logger.info(f"Round {self.round} aggregation weights: {[f'{w:.4f}' for w in weights]}")

        # Aggregate weights
        global_state = self.global_model.state_dict()

        if self.use_update_normalization and self.aggregation_method == 'signal_weighted':
            # Delta-based aggregation with update normalization
            # 1. Compute deltas
            client_deltas = []
            for cw in client_weights:
                delta = {k: cw[k].to(self.device) - global_state[k] for k in global_state}
                client_deltas.append(delta)

            # 2. Compute L2 norms
            delta_norms = []
            for delta in client_deltas:
                norm_sq = sum(torch.sum(v ** 2).item() for v in delta.values())
                delta_norms.append(norm_sq ** 0.5)

            # 3. Normalize to median norm (robust to outliers)
            valid_norms = sorted(n for n in delta_norms if n > 1e-8)
            if valid_norms:
                target_norm = valid_norms[len(valid_norms) // 2]
                scale_factors = [
                    target_norm / n if n > 1e-8 else 0.0 for n in delta_norms
                ]
            else:
                target_norm = 0.0
                scale_factors = [1.0] * num_clients

            if is_main_process():
                logger.info(
                    f"Round {self.round} delta norms: {[f'{n:.4f}' for n in delta_norms]}, "
                    f"target_norm: {target_norm:.4f}, "
                    f"scale_factors: {[f'{s:.4f}' for s in scale_factors]}"
                )

            # 4. Aggregation: θ_new = θ_global + Σ w_k * s_k * Δ_k
            aggregated_state = {}
            for key in global_state:
                aggregated_state[key] = global_state[key].clone()
                for i in range(num_clients):
                    aggregated_state[key] += weights[i] * scale_factors[i] * client_deltas[i][key]

            del client_deltas
        else:
            # Standard weighted aggregation (for sample_weighted / uniform / no normalization)
            aggregated_state = {}
            for key in global_state:
                aggregated_state[key] = torch.zeros_like(global_state[key], device=self.device)
                for i, client_state in enumerate(client_weights):
                    aggregated_state[key] += client_state[key].to(self.device) * weights[i]

        # Load aggregated state into global model
        self.global_model.load_state_dict(aggregated_state)

        # Free up CPU memory from client weights immediately after aggregation
        del client_weights
        del aggregated_state

        # Update reference model periodically (DeepSeek-R1-Zero style)
        if self._ref_state_dict is not None and self.round % self.ref_model_update_interval == 0:
            self._update_ref_model()

        # Compute aggregation statistics
        avg_loss = sum(m.get('loss', m.get('avg_loss', 0)) for m in client_metrics) / num_clients
        avg_reward = sum(m.get('avg_reward', 0) for m in client_metrics) / num_clients

        # Compute global reward stats for per-prompt calibration
        client_avg_rewards = [m.get('avg_reward', 0.0) for m in client_metrics]
        self.global_reward_mean = sum(client_avg_rewards) / num_clients
        if num_clients > 1:
            variance = sum((r - self.global_reward_mean) ** 2 for r in client_avg_rewards) / (num_clients - 1)
            self.global_reward_std = variance ** 0.5
        else:
            self.global_reward_std = 0.0

        if is_main_process():
            logger.info(
                f"Round {self.round} global reward stats: "
                f"μ_global={self.global_reward_mean:.4f}, σ_global={self.global_reward_std:.4f}"
            )

        stats = {
            'round': self.round,
            'num_clients': num_clients,
            'avg_samples': total_samples / num_clients,
            'avg_signal_strength': total_signal / num_clients,
            'avg_loss': avg_loss,
            'avg_reward': avg_reward,
            'global_reward_mean': self.global_reward_mean,
            'global_reward_std': self.global_reward_std,
            'weights': weights,
            'aggregation_method': self.aggregation_method
        }

        if is_main_process():
            logger.info(
                f"Round {self.round} aggregation complete: "
                f"clients={num_clients}, "
                f"loss={avg_loss:.4f}, "
                f"reward={avg_reward:.4f}"
            )

        return stats

    def get_global_reward_stats(self):
        """Get global reward statistics for per-prompt calibration.

        Returns:
            Tuple of (global_reward_mean, global_reward_std) or (None, None) if not yet computed.
        """
        return self.global_reward_mean, self.global_reward_std

    def _update_ref_model(self):
        """Update reference model by mixing current global model weights.

        Formula: π_ref = α * π_θ + (1 - α) * π_ref_prev
        (Consistent with TRL's ref_model_mixup_alpha in centralized training)

        When ref_model_mixup_alpha = 1.0: Direct copy (DeepSeek-R1-Zero style)
            ref = global
        When ref_model_mixup_alpha < 1.0: EMA-like mixup
            ref = alpha * global + (1 - alpha) * ref_prev

        Memory optimization: Weights are stored on CPU to avoid loading the full model to GPU.
        """
        if self._ref_state_dict is None:
            return

        with torch.no_grad():
            alpha = self.ref_model_mixup_alpha
            if alpha < 1.0:
                # Mixup: ref = alpha * global + (1 - alpha) * ref_prev
                for key, value in self.global_model.state_dict().items():
                    global_val = value.cpu()
                    self._ref_state_dict[key] = alpha * global_val + (1 - alpha) * self._ref_state_dict[key]
                if is_main_process():
                    logger.info(f"Reference model updated at round {self.round} with mixup (alpha={alpha})")
            else:
                # Direct copy (alpha=1.0, DeepSeek-R1-Zero style)
                for key, value in self.global_model.state_dict().items():
                    self._ref_state_dict[key] = value.cpu().clone()
                if is_main_process():
                    logger.info(f"Reference model updated at round {self.round} (direct copy)")
