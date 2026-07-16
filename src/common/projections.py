"""Canonical GPT-2 projection discovery and weight-orientation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn as nn
from torch import Tensor
from transformers.pytorch_utils import Conv1D


@dataclass(frozen=True)
class ProjectionHandle:
    projection_id: str
    module_path: str
    module: nn.Module
    parent: nn.Module
    attribute: str
    block_index: int | None
    role: str
    in_features: int
    out_features: int
    parameter_count: int
    macs_per_token: int
    tied_to_embedding: bool = False


def canonical_weight_bias(module: nn.Module) -> tuple[Tensor, Tensor | None]:
    if isinstance(module, Conv1D):
        # Hugging Face Conv1D stores [in, out]; canonical form is [out, in].
        return module.weight.detach().T.contiguous(), (
            None if module.bias is None else module.bias.detach().contiguous()
        )
    if isinstance(module, nn.Linear):
        return module.weight.detach().contiguous(), (
            None if module.bias is None else module.bias.detach().contiguous()
        )
    raise TypeError(f"Unsupported projection module: {type(module).__name__}")


def linear_from_canonical(
    weight: Tensor, bias: Tensor | None, device: torch.device
) -> nn.Linear:
    out_features, in_features = weight.shape
    linear = nn.Linear(in_features, out_features, bias=bias is not None, device=device)
    with torch.no_grad():
        linear.weight.copy_(weight.to(device=device, dtype=torch.float32))
        if bias is not None and linear.bias is not None:
            linear.bias.copy_(bias.to(device=device, dtype=torch.float32))
    return linear


def restore_canonical_weight(module: nn.Module, weight: Tensor, bias: Tensor | None) -> None:
    with torch.no_grad():
        if isinstance(module, Conv1D):
            module.weight.copy_(weight.T.to(module.weight.device, module.weight.dtype))
        elif isinstance(module, nn.Linear):
            module.weight.copy_(weight.to(module.weight.device, module.weight.dtype))
        else:
            raise TypeError(type(module).__name__)
        if bias is not None and getattr(module, "bias", None) is not None:
            module.bias.copy_(bias.to(module.bias.device, module.bias.dtype))


def get_module_by_path(root: nn.Module, path: str) -> nn.Module:
    current: Any = root
    for component in path.split("."):
        current = current[int(component)] if component.isdigit() else getattr(current, component)
    if not isinstance(current, nn.Module):
        raise TypeError(f"{path} is not an nn.Module.")
    return current


def get_parent_and_attribute(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = get_module_by_path(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    return parent, parts[-1]


def iter_gpt2_projections(model: nn.Module, *, include_lm_head: bool = True) -> Iterator[ProjectionHandle]:
    roles = (
        ("attn.c_attn", "attn.c_attn"),
        ("attn.c_proj", "attn.c_proj"),
        ("mlp.c_fc", "mlp.c_fc"),
        ("mlp.c_proj", "mlp.c_proj"),
    )
    blocks = model.transformer.h
    for block_index, block in enumerate(blocks):
        for relative_path, role in roles:
            module = get_module_by_path(block, relative_path)
            canonical, _ = canonical_weight_bias(module)
            out_features, in_features = map(int, canonical.shape)
            full_path = f"transformer.h.{block_index}.{relative_path}"
            parent, attribute = get_parent_and_attribute(model, full_path)
            yield ProjectionHandle(
                projection_id=f"block_{block_index}/{role}",
                module_path=full_path,
                module=module,
                parent=parent,
                attribute=attribute,
                block_index=block_index,
                role=role,
                in_features=in_features,
                out_features=out_features,
                parameter_count=in_features * out_features,
                macs_per_token=in_features * out_features,
            )
    if include_lm_head:
        module = model.lm_head
        canonical, _ = canonical_weight_bias(module)
        out_features, in_features = map(int, canonical.shape)
        parent, attribute = get_parent_and_attribute(model, "lm_head")
        tied = bool(module.weight.data_ptr() == model.transformer.wte.weight.data_ptr())
        yield ProjectionHandle(
            projection_id="lm_head",
            module_path="lm_head",
            module=module,
            parent=parent,
            attribute=attribute,
            block_index=None,
            role="lm_head",
            in_features=in_features,
            out_features=out_features,
            parameter_count=in_features * out_features,
            macs_per_token=in_features * out_features,
            tied_to_embedding=tied,
        )


def projection_catalog(model: nn.Module, *, include_lm_head: bool = True) -> list[dict[str, Any]]:
    return [
        {
            "projection_id": h.projection_id,
            "module_path": h.module_path,
            "role": h.role,
            "block_index": h.block_index,
            "in_features": h.in_features,
            "out_features": h.out_features,
            "parameter_count": h.parameter_count,
            "macs_per_token": h.macs_per_token,
            "tied_to_embedding": h.tied_to_embedding,
        }
        for h in iter_gpt2_projections(model, include_lm_head=include_lm_head)
    ]
