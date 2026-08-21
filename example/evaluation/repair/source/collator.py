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
        document_lengths = [len(row["input_ids"]) for row in rows]
        total_length = input_ids.shape[1]
        # Packed documents share one sequence but must stay mutually
        # invisible: each document's tokens may only attend to earlier
        # tokens within that same document, mirroring the isolated,
        # per-document attention that evaluation uses. Without this,
        # later documents in a pack silently attend into unrelated
        # earlier documents (and across nonsensical negative RoPE
        # offsets, since position_ids resets per document).
        segment_ids = torch.repeat_interleave(
            torch.arange(len(rows)),
            torch.tensor(document_lengths),
        )
        same_document = segment_ids.unsqueeze(0) == segment_ids.unsqueeze(1)
        causal = torch.tril(
            torch.ones(total_length, total_length, dtype=torch.bool)
        )
        attention_mask = (same_document & causal).view(
            1, 1, total_length, total_length
        )
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
            "example_ids": [str(row["example_id"]) for row in rows],
            "document_lengths": document_lengths,
        }
