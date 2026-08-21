from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch


MODULE_NAMES = ("settings", "collator")
sys.dont_write_bytecode = True


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        values = [json.loads(line) for line in handle if line.strip()]
    if not values or any(not isinstance(value, dict) for value in values):
        raise ValueError(f"expected non-empty JSON objects: {path}")
    return values


@contextmanager
def workspace_modules(workspace: Path) -> Iterator[tuple[object, object]]:
    source = str((workspace / "source").resolve())
    saved_modules = {
        name: sys.modules.pop(name, None) for name in MODULE_NAMES
    }
    previous_path = list(sys.path)
    try:
        sys.path.insert(0, source)
        settings = importlib.import_module("settings")
        collator = importlib.import_module("collator")
        yield settings, collator
    finally:
        sys.path[:] = previous_path
        for name in MODULE_NAMES:
            sys.modules.pop(name, None)
            if saved_modules[name] is not None:
                sys.modules[name] = saved_modules[name]


def distributed_indices(
    *,
    examples: int,
    replicas: int,
    rank: int,
    seed: int,
) -> list[int]:
    if examples % replicas:
        raise ValueError("training examples must divide evenly across ranks")
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(examples, generator=generator).tolist()
    return order[rank::replicas]


def batch_pipeline_metrics(batch: dict) -> tuple[int, int]:
    required = {"input_ids", "position_ids", "labels"}
    if not required.issubset(batch):
        raise ValueError("collator output is missing model tensors")
    input_ids = torch.as_tensor(batch["input_ids"])
    position_ids = torch.as_tensor(batch["position_ids"])
    labels = torch.as_tensor(batch["labels"])
    if (
        input_ids.ndim != 2
        or position_ids.shape != input_ids.shape
        or labels.shape != input_ids.shape
    ):
        raise ValueError("collator tensor shapes differ")

    boundary_targets = int(
        (
            position_ids[:, 1:].eq(0)
            & labels[:, 1:].ne(-100)
        ).sum().item()
    )
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        return 0, boundary_targets
    mask = torch.as_tensor(attention_mask)

    cross_pairs = 0
    for row in range(input_ids.shape[0]):
        segment = -1
        segments: list[int] = []
        for position in position_ids[row].tolist():
            if int(position) == 0:
                segment += 1
            segments.append(segment)
        if mask.ndim == 2:
            prior_keys = 0
            for segment_id in range(segment + 1):
                positions = [
                    index
                    for index, value in enumerate(segments)
                    if value == segment_id
                ]
                cross_pairs += len(positions) * prior_keys
                prior_keys += sum(
                    bool(mask[row, index].item())
                    for index in positions
                )
            continue

        if mask.ndim == 3:
            row_mask = mask[row]
        elif mask.ndim == 4:
            row_mask = mask[row, 0]
        else:
            raise ValueError(
                "attention_mask must have two, three, or four axes"
            )
        if row_mask.dtype == torch.bool:
            allowed = row_mask
        else:
            allowed = torch.isfinite(row_mask) & row_mask.gt(-1.0e4)
        segment_tensor = torch.tensor(segments)
        width = len(segments)
        causal = torch.arange(width).unsqueeze(0).le(
            torch.arange(width).unsqueeze(1)
        )
        cross_document = segment_tensor.unsqueeze(0).ne(
            segment_tensor.unsqueeze(1)
        )
        cross_pairs += int(
            (allowed & causal & cross_document).sum().item()
        )
    return cross_pairs, boundary_targets


def training_process_metrics(workspace: Path) -> dict[str, float]:
    rows = load_jsonl(workspace / "data" / "train.jsonl")
    with workspace_modules(workspace) as (settings, collator_module):
        replicas = int(settings.WORLD_SIZE)
        batch_size = int(settings.DOCUMENTS_PER_PACK)
        seed = int(settings.SEED)
        if len(rows) != int(settings.TRAIN_EXAMPLES):
            raise ValueError("training example count differs from settings")
        collator = collator_module.PackedBatchCollator()
        cross_pairs = 0
        boundary_targets = 0
        for rank in range(replicas):
            indices = distributed_indices(
                examples=len(rows),
                replicas=replicas,
                rank=rank,
                seed=seed,
            )
            if len(indices) % batch_size:
                raise ValueError("rank shard does not divide into full packs")
            for offset in range(0, len(indices), batch_size):
                batch_rows = [
                    rows[index]
                    for index in indices[offset : offset + batch_size]
                ]
                batch_cross, batch_boundaries = batch_pipeline_metrics(
                    collator(batch_rows)
                )
                cross_pairs += batch_cross
                boundary_targets += batch_boundaries
    return {
        "cross_document_attention_pairs": float(cross_pairs),
        "cross_document_boundary_targets": float(boundary_targets),
    }


def measured_metrics(workspace: Path, output: Path) -> dict[str, float]:
    metrics = training_process_metrics(workspace)
    summary = load_object(output / "final_validation" / "summary.json")
    perplexity = summary.get("perplexity")
    if (
        isinstance(perplexity, bool)
        or not isinstance(perplexity, (int, float))
        or not math.isfinite(float(perplexity))
    ):
        raise ValueError("validation perplexity is invalid")
    metrics["validation_perplexity"] = float(perplexity)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--flawed-workspace", type=Path, required=True)
    parser.add_argument("--corrected-workspace", type=Path, required=True)
    parser.add_argument("--flawed-output", type=Path, required=True)
    parser.add_argument("--corrected-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = load_object(args.contract)
    flawed = measured_metrics(
        args.flawed_workspace, args.flawed_output
    )
    corrected = measured_metrics(
        args.corrected_workspace, args.corrected_output
    )
    report = {"metrics": {}}
    for name in contract["metrics"]:
        report["metrics"][name] = {
            "flawed": flawed[name],
            "corrected": corrected[name],
        }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
