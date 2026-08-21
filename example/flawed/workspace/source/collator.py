from __future__ import annotations

from typing import Any

import torch


class PackedBatchCollator:
    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            raise RuntimeError("cannot pack an empty document batch")
        input_ids = torch.tensor(
            [[
                token
                for row in rows
                for token in row["input_ids"]
            ]],
            dtype=torch.long,
        )
        position_ids = torch.tensor(
            [[
                position
                for row in rows
                for position in range(len(row["input_ids"]))
            ]],
            dtype=torch.long,
        )
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": position_ids,
            "labels": labels,
            "example_ids": [str(row["example_id"]) for row in rows],
            "document_lengths": [
                len(row["input_ids"]) for row in rows
            ],
        }
