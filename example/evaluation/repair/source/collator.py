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
            # No dense attention_mask is provided here. A packed batch
            # concatenates multiple documents into one sequence and relies on
            # the per-document reset in `position_ids` for the model to
            # auto-detect document boundaries and build a block-diagonal
            # causal mask (each document only attends to itself). Passing an
            # all-ones attention_mask instead disables that auto-detection
            # and causal-attention leaks across documents in the same pack.
            "position_ids": position_ids,
            "labels": labels,
            "example_ids": [str(row["example_id"]) for row in rows],
            "document_lengths": [
                len(row["input_ids"]) for row in rows
            ],
        }
