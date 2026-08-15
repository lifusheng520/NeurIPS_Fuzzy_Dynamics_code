"""Prompt loading utilities for extraction experiments."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any


DEFAULT_PROMPT_FIELDS = (
    "r2(r1(e1)).prompt",
    "prompt",
    "question",
    "query",
    "text",
)


def _select_prompt(row: dict[str, Any], prompt_field: str | None) -> str:
    fields = (prompt_field,) if prompt_field else DEFAULT_PROMPT_FIELDS
    for field in fields:
        if field and row.get(field) not in (None, ""):
            return str(row[field]).strip()
    raise KeyError(
        f"Could not find a prompt. Tried fields: {', '.join(x for x in fields if x)}"
    )


def load_prompt_records(
    path: str | Path,
    prompt_field: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load prompts from CSV, JSON, JSONL, or plain-text files."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()

    if suffix == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(itertools.islice(reader, limit)) if limit is not None else list(reader)
    elif suffix == ".json":
        with source.open(encoding="utf-8") as handle:
            value = json.load(handle)
        rows = value if isinstance(value, list) else [value]
    elif suffix == ".jsonl":
        with source.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    elif suffix in {".txt", ".text"}:
        with source.open(encoding="utf-8") as handle:
            rows = [{"text": line.strip()} for line in handle if line.strip()]
    else:
        raise ValueError(f"Unsupported input format: {suffix}")

    selected_rows = rows[:limit] if limit is not None and suffix != ".csv" else rows
    records: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows):
        if not isinstance(row, dict):
            row = {"text": str(row)}
        records.append(
            {
                "index": index,
                "prompt": _select_prompt(row, prompt_field),
                "uid": row.get("uid", index),
                "bridge_entity": row.get("e2.value"),
                "answer": row.get("e3.value") or row.get("answer"),
                "category": row.get("category") or row.get("fact_comp_type"),
            }
        )
    return records
