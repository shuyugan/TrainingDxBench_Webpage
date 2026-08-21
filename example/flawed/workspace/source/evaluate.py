from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from common import write_json, write_jsonl
from data import ChatDocumentDataset
from model import load_model
from settings import EVAL_BATCH_SIZE


def collate_documents(rows: list[dict[str, Any]]) -> dict[str, Any]:
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.zeros(len(rows), width, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        input_ids[index, : len(values)] = values
        attention_mask[index, : len(values)] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": torch.arange(width).expand(len(rows), -1),
        "example_ids": [str(row["example_id"]) for row in rows],
    }


def evaluate_model(
    *,
    model: Any,
    records_path: Path,
    device: torch.device,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    dataset = ChatDocumentDataset(records_path.resolve())
    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_documents,
    )
    total_log_probability = 0.0
    total_tokens = 0
    predictions = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = {
                name: batch[name].to(device)
                for name in (
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                )
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(**inputs, use_cache=False).logits
            token_logits = logits[:, :-1].float()
            targets = inputs["input_ids"][:, 1:]
            active = inputs["attention_mask"][:, 1:].bool()
            selected = functional.log_softmax(
                token_logits, dim=-1
            ).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            predicted = token_logits.argmax(dim=-1)
            total_log_probability += float(
                selected[active].double().sum()
            )
            total_tokens += int(active.sum())
            for index, example_id in enumerate(batch["example_ids"]):
                row_active = active[index]
                predictions.append(
                    {
                        "example_id": str(example_id),
                        "target_token_ids": [
                            int(value)
                            for value in targets[index, row_active].cpu()
                        ],
                        "predicted_token_ids": [
                            int(value)
                            for value in predicted[index, row_active].cpu()
                        ],
                        "target_token_log_probabilities": [
                            float(value)
                            for value in selected[index, row_active].cpu()
                        ],
                    }
                )
    return {
        "examples": len(dataset),
        "perplexity": math.exp(-total_log_probability / total_tokens),
    }, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, _ = load_model(
        args.base_model_dir,
        adapter_dir=(
            args.adapter_dir.resolve()
            if args.adapter_dir is not None
            else None
        ),
        device=device,
    )
    summary, predictions = evaluate_model(
        model=model,
        records_path=args.records.resolve(),
        device=device,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        output_dir / "summary.json", summary, exclusive=True
    )
    write_jsonl(
        output_dir / "predictions.jsonl",
        predictions,
        exclusive=True,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
