"""GPT-2 projection catalog and module-resolution helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn as nn
from torch import Tensor
from transformers.pytorch_utils import Conv1D


PROJECTION_NAMES = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")


@dataclass(frozen=True)
class ProjectionHandle:
    projection_id: str
    hf_module_path: str
    block_index: int
    projection_name: str
    parent: nn.Module
    attribute: str
    module: nn.Module


def _parent_and_attribute(root: Any, dotted_path: str) -> tuple[Any, str]:
    parts = dotted_path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def iter_gpt2_projections(model: Any) -> Iterator[ProjectionHandle]:
    for block_index, block in enumerate(model.transformer.h):
        for projection_name in PROJECTION_NAMES:
            parent, attribute = _parent_and_attribute(block, projection_name)
            module = getattr(parent, attribute)
            yield ProjectionHandle(
                projection_id=f"block_{block_index}/{projection_name}",
                hf_module_path=f"transformer.h.{block_index}.{projection_name}",
                block_index=block_index,
                projection_name=projection_name,
                parent=parent,
                attribute=attribute,
                module=module,
            )


def canonical_weight_bias(module: nn.Module) -> tuple[Tensor, Tensor | None]:
    if isinstance(module, Conv1D):
        weight = module.weight.detach().T.contiguous()
        bias = module.bias.detach().contiguous()
    elif isinstance(module, nn.Linear):
        weight = module.weight.detach().contiguous()
        bias = None if module.bias is None else module.bias.detach().contiguous()
    else:
        raise TypeError(f"Unsupported projection type: {type(module).__name__}")
    return weight.float(), None if bias is None else bias.float()


def linear_from_canonical(
    weight: Tensor,
    bias: Tensor | None,
    device: torch.device,
) -> nn.Linear:
    weight = weight.detach().to(device=device, dtype=torch.float32).contiguous()
    bias_device = None if bias is None else bias.detach().to(device=device, dtype=torch.float32)
    layer = nn.Linear(
        in_features=int(weight.shape[1]),
        out_features=int(weight.shape[0]),
        bias=bias is not None,
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        layer.weight.copy_(weight)
        if bias_device is not None:
            layer.bias.copy_(bias_device)
    return layer
