from __future__ import annotations

import contextlib
import hashlib
import math
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file
from torch import nn

from common import read_json, write_json
from settings import (
    ADAPTER_ALPHA,
    ADAPTER_RANK,
    ADAPTER_TARGETS,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODEL_WEIGHT_SHA256,
)


def no_autocast(device_type: str):
    if device_type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device_type, enabled=False)
    return contextlib.nullcontext()


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base_layer = base
        self.scale = alpha / rank
        initial_A = torch.empty(
            rank, base.in_features, dtype=torch.float32, device="cpu"
        )
        nn.init.kaiming_uniform_(initial_A, a=math.sqrt(5))
        self.lora_A = nn.Parameter(
            initial_A.to(device=base.weight.device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                rank,
                dtype=torch.float32,
                device=base.weight.device,
            )
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_layer.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        with no_autocast(inputs.device.type):
            update = functional.linear(
                functional.linear(inputs.float(), self.lora_A),
                self.lora_B,
            )
        return base + update.to(base.dtype) * self.scale


def attach_adapter(model: Any) -> Any:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise RuntimeError("model does not expose decoder layers")
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise RuntimeError("decoder layer has no self attention")
        for name in ADAPTER_TARGETS:
            base = getattr(attention, name, None)
            if not isinstance(base, nn.Linear):
                raise RuntimeError(f"{name} is not linear")
            setattr(
                attention,
                name,
                LoRALinear(base, ADAPTER_RANK, ADAPTER_ALPHA),
            )
    return model


def adapter_parameters(model: Any) -> dict[str, nn.Parameter]:
    values = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.endswith(".lora_A") or name.endswith(".lora_B")
    }
    if not values:
        raise RuntimeError("adapter has no parameters")
    return values


def save_adapter(model: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "adapter.safetensors").unlink(missing_ok=True)
    (output_dir / "adapter_config.json").unlink(missing_ok=True)
    save_file(
        {
            name: parameter.detach().cpu().contiguous()
            for name, parameter in adapter_parameters(model).items()
        },
        output_dir / "adapter.safetensors",
    )
    write_json(
        output_dir / "adapter_config.json",
        {
            "adapter_type": "attention_lora",
            "alpha": ADAPTER_ALPHA,
            "rank": ADAPTER_RANK,
            "target_modules": list(ADAPTER_TARGETS),
        },
        exclusive=True,
    )


def load_adapter(model: Any, output_dir: Path) -> None:
    config = read_json(output_dir / "adapter_config.json")
    if (
        config.get("adapter_type") != "attention_lora"
        or int(config["rank"]) != ADAPTER_RANK
        or float(config["alpha"]) != ADAPTER_ALPHA
        or tuple(config["target_modules"]) != ADAPTER_TARGETS
    ):
        raise RuntimeError("adapter configuration differs")
    attach_adapter(model)
    parameters = adapter_parameters(model)
    tensors = load_file(output_dir / "adapter.safetensors")
    if set(parameters) != set(tensors):
        raise RuntimeError("adapter tensor keys differ")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(
                tensors[name].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )


def prepare_base_model(base_model_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    base_model_dir = base_model_dir.resolve()
    weight = base_model_dir / "model.safetensors"
    if not weight.is_file():
        downloaded = Path(
            hf_hub_download(
                repo_id=MODEL_REPOSITORY,
                filename=weight.name,
                revision=MODEL_REVISION,
                local_dir=base_model_dir,
            )
        )
        if downloaded.resolve() != weight:
            raise RuntimeError(
                f"model weight downloaded to {downloaded}, not {weight}"
            )
    digest = hashlib.sha256()
    with weight.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != MODEL_WEIGHT_SHA256:
        raise RuntimeError(
            "base model digest differs: "
            f"expected {MODEL_WEIGHT_SHA256}, found {actual}"
        )
    cache = base_model_dir / ".cache"
    if cache.is_dir():
        shutil.rmtree(cache)


def load_model(
    base_model_dir: Path,
    *,
    adapter_dir: Path | None,
    device: torch.device,
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not (base_model_dir / "model.safetensors").is_file():
        prepare_base_model(base_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_dir.resolve(),
        local_files_only=True,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir.resolve(),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    if adapter_dir is None:
        attach_adapter(model)
    else:
        load_adapter(model, adapter_dir.resolve())
    model.config.use_cache = False
    return model, tokenizer
