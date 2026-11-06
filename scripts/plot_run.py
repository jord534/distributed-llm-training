
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent


def plot_run(run_name: str):
    # reads artifacts/runs/<run_name>/log.jsonl and saves one png per metric
    log_path = REPO_ROOT / "artifacts" / "runs" / run_name / "log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"No log found: {log_path}")

    # training lines have "loss", validation lines have "val_loss"
    train_lines = []
    val_lines = []
    with open(log_path, "r") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if "val_loss" in record:
                val_lines.append(record)
            else:
                train_lines.append(record)

    if not train_lines and not val_lines:
        raise ValueError(f"No records found in {log_path}")

    output_dir = REPO_ROOT / "reports" / "plots" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["loss", "grad_norm", "lr", "step_time_s"]:
        # keeps only records containing the metric
        metric_lines = [r for r in train_lines if metric in r and "step" in r]

        if not metric_lines:
            continue

        plt.figure()
        plt.plot(
            [r["step"] for r in metric_lines],
            [r[metric] for r in metric_lines],
            label="train"
        )

        if metric == "loss":
            val_metric_lines = [r for r in val_lines if "val_loss" in r and "step" in r]

            if val_metric_lines:
                plt.plot(
                    [r["step"] for r in val_metric_lines],
                    [r["val_loss"] for r in val_metric_lines],
                    "o--",
                    label="val"
                )

        plt.title(f"{run_name}: {metric}")
        plt.xlabel("step")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png")
        plt.close()

    print(f"Plots written to {output_dir}")
    return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: uv run python scripts/plot_run.py <run name>")
    plot_run(sys.argv[1])
