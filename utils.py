# Small helper functions for reading and writing files.

import json
import os

from config import DATA_DIR


def load_items(filename):
    """Load a question file from the data folder, e.g. truthfulqa_items.json."""
    with open(os.path.join(DATA_DIR, filename), encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    """Read a .jsonl file (one JSON object per line). Returns [] if the file does not exist."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def append_jsonl(file, row):
    """Write one row and save it to disk at once, so nothing is lost if the run stops."""
    file.write(json.dumps(row) + "\n")
    file.flush()
