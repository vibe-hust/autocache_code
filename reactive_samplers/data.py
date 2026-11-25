"""Read the committed prompt-and-seed manifests."""
import json
from pathlib import Path


def read_prompts(path, expected):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} prompt/seed rows; found {len(rows)}")
    seen = set()
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in row["id"]):
            raise ValueError("Prompt ids must be unique filename-safe strings")
        if row["id"] in seen or not isinstance(row.get("prompt"), str) or not row["prompt"].strip() or type(row.get("seed")) is not int:
            raise ValueError("Each row needs a unique id, nonempty prompt and integer seed")
        seen.add(row["id"])
    return rows


def drawbench_splits(rows, directory):
    # Match prompt/seed pairs; result order and sample IDs can differ.
    directory = Path(directory)
    by_pair = {(row["prompt"], row["seed"]): row for row in rows}
    return {
        split: [by_pair[(row["prompt"], row["seed"])]
                for row in read_prompts(directory / filename, count)]
        for split, filename, count in (("development30", "development.jsonl", 30),
                                       ("evaluation170", "test.jsonl", 170))
    }
