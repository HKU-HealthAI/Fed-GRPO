"""
Communication Compression for Federated GRPO
"""

import torch
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def compress_delta(
    global_state: Dict[str, torch.Tensor],
    local_state: Dict[str, torch.Tensor],
    reward_std: float,
    compression_config: Dict[str, Any],
    prev_reward_stds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compress client delta (local_state - global_state) for communication.

    The compression works in two stages:
      Stage 1 (BF16 sparse): Identify parameters that actually changed at BF16 precision.
              Only these need to be transmitted. With budget=1.0, this is lossless.
      Stage 2 (Module selection): When budget < 1.0, rank modules by efficiency
              (||Δ||² / bf16_changed_count) and greedily select top modules within budget.

    Args:
        global_state: Global model state dict (BF16 tensors on CPU)
        local_state: Local model state dict after training (BF16 tensors on CPU)
        reward_std: This client's reward_std from GRPO training
        compression_config: Dict with keys: use_compression, budget, adaptive, adaptive_alpha
        prev_reward_stds: Previous round's reward_stds for all clients (for adaptive mode)

    Returns:
        Dict with:
            'compressed': True
            'selected_modules': Dict[param_name -> {indices, values, shape}]
            'skipped_modules': List of param names not transmitted
            'stats': compression statistics
    """
    budget = compression_config.get('budget', 1.0)
    adaptive = compression_config.get('adaptive', False)
    adaptive_alpha = compression_config.get('adaptive_alpha', 0.5)

    # --- Step 1: Compute BF16 delta and bf16_changed_mask per module ---
    module_info = {}
    total_bf16_changed = 0
    total_params = 0

    for name in global_state:
        if name not in local_state:
            continue

        g = global_state[name]   # BF16 on CPU
        l = local_state[name]    # BF16 on CPU

        # Exact BF16 round-trip: which params actually changed?
        bf16_changed_mask = (g != l)
        bf16_changed_count = bf16_changed_mask.sum().item()
        num_params = g.numel()
        total_params += num_params

        if bf16_changed_count == 0:
            continue  # No BF16-visible change, skip entirely

        total_bf16_changed += bf16_changed_count

        # Compute efficiency = ||Δ||² / bf16_changed_count
        delta = l.float() - g.float()
        delta_norm_sq = (delta ** 2).sum().item()
        efficiency = delta_norm_sq / (bf16_changed_count + 1)

        module_info[name] = {
            'bf16_changed_mask': bf16_changed_mask,
            'bf16_changed_count': bf16_changed_count,
            'delta_norm_sq': delta_norm_sq,
            'efficiency': efficiency,
            'num_params': num_params,
        }

    # --- Step 2: Compute actual budget ---
    if 0 < budget < 1.0 and adaptive and prev_reward_stds and max(prev_reward_stds) > 0:
        max_rs = max(prev_reward_stds)
        ratio = reward_std / max_rs
        avg_ratio = sum(rs / max_rs for rs in prev_reward_stds) / len(prev_reward_stds)
        actual_budget = budget * (1.0 + adaptive_alpha * (ratio - avg_ratio))
        actual_budget = max(0.01, min(1.0, actual_budget))
    else:
        actual_budget = budget

    budget_count = int(total_bf16_changed * actual_budget)

    # --- Step 3: Sort by efficiency, greedy select modules ---
    sorted_modules = sorted(module_info.items(), key=lambda x: x[1]['efficiency'], reverse=True)

    selected = {}
    cum_bf16_changed = 0
    cum_delta_norm_sq = 0
    total_delta_norm_sq = sum(info['delta_norm_sq'] for info in module_info.values())

    for name, info in sorted_modules:
        if actual_budget < 1.0 and cum_bf16_changed + info['bf16_changed_count'] > budget_count:
            break
        selected[name] = info
        cum_bf16_changed += info['bf16_changed_count']
        cum_delta_norm_sq += info['delta_norm_sq']

    # --- Step 4: Extract sparse (indices, values) for selected modules ---
    selected_modules = {}
    for name, info in selected.items():
        mask = info['bf16_changed_mask']
        flat_mask = mask.view(-1)
        indices = flat_mask.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
        values = local_state[name].view(-1)[indices.long()].clone()

        selected_modules[name] = {
            'indices': indices,
            'values': values,
            'shape': list(local_state[name].shape),
        }

    skipped_modules = [name for name in module_info if name not in selected]

    # --- Stats ---
    delta_preserved_frac = cum_delta_norm_sq / (total_delta_norm_sq + 1e-10)
    param_compression = total_params / (cum_bf16_changed + 1)

    stats = {
        'total_params': total_params,
        'total_bf16_changed': total_bf16_changed,
        'transmitted_bf16_changed': cum_bf16_changed,
        'num_selected_modules': len(selected_modules),
        'num_skipped_modules': len(skipped_modules),
        'actual_budget': actual_budget,
        'delta_preserved_frac': delta_preserved_frac,
        'param_compression_ratio': param_compression,
        'bf16_absorbed_frac': 1 - total_bf16_changed / (total_params + 1e-10),
    }

    return {
        'compressed': True,
        'selected_modules': selected_modules,
        'skipped_modules': skipped_modules,
        'stats': stats,
    }


def decompress_to_state_dict(
    compressed: Dict[str, Any],
    global_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Decompress sparse data back to a full state dict.

    For selected modules: reconstruct by applying sparse updates to global weights.
    For skipped modules: use global weights (delta = 0).

    Args:
        compressed: Output of compress_delta()
        global_state: Global model state dict (BF16 tensors on CPU)

    Returns:
        Full state dict (Dict[str, Tensor]) compatible with server.aggregate()
    """
    selected_modules = compressed['selected_modules']

    state_dict = {}
    for name, g_param in global_state.items():
        if name in selected_modules:
            sparse = selected_modules[name]
            # Start from global, apply sparse updates
            reconstructed = g_param.clone()
            flat = reconstructed.view(-1)
            flat[sparse['indices'].long()] = sparse['values']
            state_dict[name] = reconstructed
        else:
            # Skipped module: use global weights (no update)
            state_dict[name] = g_param.clone()

    return state_dict
