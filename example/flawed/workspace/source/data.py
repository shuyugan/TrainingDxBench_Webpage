from __future__ import annotations

from pathlib import Path
from typing import Any

from common import read_jsonl
from settings import MAX_DOCUMENT_LENGTH

ROW_FIELDS = {
    "example_id",
    "source_index",
    "source",
    "input_ids",
    "text",
}


class ChatDocumentDataset:
    def __init__(self, path: Path):
        rows = read_jsonl(path.resolve())
        if not rows or any(set(row) != ROW_FIELDS for row in rows):
            raise RuntimeError("chat document schema differs")
        if any(
            not isinstance(row["input_ids"], list)
            or not 2 <= len(row["input_ids"]) <= MAX_DOCUMENT_LENGTH
            or any(type(value) is not int for value in row["input_ids"])
            for row in rows
        ):
            raise RuntimeError("chat document tokens differ")
        if len({str(row["example_id"]) for row in rows}) != len(rows):
            raise RuntimeError("example IDs are not unique")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self.rows[index])
