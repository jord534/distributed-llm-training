from pathlib import Path
import json
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent

def count_tokens_seen(run_name):
    log_dir = REPO_ROOT / "artifacts" / "runs" / run_name

    if not log_dir.exists():
        raise FileNotFoundError(f"The specified run called {run_name} was not found in artifacts/run.")

    log_path = log_dir / "log.jsonl"
    train_lines = []
    val_lines = []
    with open(log_path, "r") as file:
        for line in file:
            if line.strip():  # Skip any accidental blank lines
                log_entry = json.loads(line)
                if "val_loss" in log_entry:
                    val_lines.append(log_entry)
                else:
                    train_lines.append(log_entry)
    train_tokens = sum([line["tokens"] for line in train_lines])
    print(f"Run {run_name} : \n \n \t Number of tokens seen in training : {train_tokens}")
    return train_tokens

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: uv run python scripts/train_log_utils.py <run_name>")
    run_name = sys.argv[1]
    count_tokens_seen(run_name)