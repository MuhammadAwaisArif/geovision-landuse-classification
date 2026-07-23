import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import prune


def prune_unstructured(model: nn.Module, amount: float = 0.30):
    pruned = copy.deepcopy(model).cpu()
    for module in pruned.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return pruned


def dynamically_quantize(model: nn.Module):
    # 1. Make a deep copy to keep the original model clean
    quantized_model = copy.deepcopy(model).cpu()
    
    # 2. Temporarily extract and remove the transformer block so PyTorch can't see it
    transformer_backup = None
    if hasattr(quantized_model, 'transformer'):
        transformer_backup = quantized_model.transformer
        quantized_model.transformer = nn.Identity() # Placeholder
        
    # 3. Apply quantization to only the remaining Linear layers (like the classification head)
    quantized_model = torch.ao.quantization.quantize_dynamic(
        quantized_model, {nn.Linear}, dtype=torch.qint8
    )
    
    # 4. Put the original, untouched transformer block back in place
    if transformer_backup is not None:
        quantized_model.transformer = transformer_backup
        
    return quantized_model


def model_statistics(model: nn.Module, checkpoint_path: str | Path | None = None):
    parameters = list(model.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    zeros = sum(torch.count_nonzero(parameter.detach() == 0).item() for parameter in parameters)
    stats = {
        "parameter_count": total,
        "zero_parameter_count": zeros,
        "parameter_sparsity": zeros / total if total else 0.0,
    }
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        stats["checkpoint_size_bytes"] = path.stat().st_size
    return stats

