"""Merge all JSON files under data/ into a single HuggingFace Dataset and push to Hub."""

import json
import shutil
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "merged_hf"
REPO_ID = "chiffonng/fatima-prework"


def load_all_json(data_dir: Path) -> list[dict]:
    """Load and concatenate all JSON files in data_dir."""
    records: list[dict] = []
    for json_path in sorted(data_dir.glob("*.json")):
        raw = json.loads(json_path.read_text())
        for record in raw:
            record["source_file"] = json_path.stem
        records.extend(raw)
    return records


def to_dataset(records: list[dict]) -> Dataset:
    """Convert list of dicts to a HuggingFace Dataset, serializing nested fields."""
    rows = {
        "input": [r["input"] for r in records],
        "expected": [r["expected"] for r in records],
        "output": [r["output"] for r in records],
        "tags": [r["tags"] for r in records],
        "metadata": [json.dumps(r["metadata"]) for r in records],
        "source_file": [r["source_file"] for r in records],
    }
    return Dataset.from_dict(rows)


def main() -> None:
    records = load_all_json(DATA_DIR)
    print(f"Loaded {len(records)} records from {DATA_DIR}")

    ds = to_dataset(records)
    print(ds)

    ds.save_to_disk(str(OUTPUT_DIR))
    print(f"Saved dataset to {OUTPUT_DIR}")

    ds.push_to_hub(REPO_ID)
    print(f"Pushed dataset to https://huggingface.co/datasets/{REPO_ID}")

    api = HfApi()
    readme_src = PROJECT_ROOT / "README.md"
    api.upload_file(
        path_or_fileobj=str(readme_src),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print("Uploaded README.md as dataset card")


if __name__ == "__main__":
    main()
