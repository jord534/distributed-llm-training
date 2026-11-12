import json
from pathlib import Path
import datetime
import hashlib


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

def write_run_metadata(config: dict, manifest: dict, world_size: int, run_dir: Path):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds") # cloud or local agree on time
    metadata = {}
    metadata["run_name"] = config["run_name"]
    metadata["created"] = now
    metadata["model"] = config["model"]
    metadata["world_size"] = world_size

    metadata["data"] = {
        "cache": config["data"]["cache"],
        "fingerprint": {
            "train": manifest["fingerprint"]["train"],
            "val": manifest["fingerprint"]["val"],
        },
    }

    metadata["config"] = config
    metadata["manifest"] = manifest
    uv_lock_path = REPO_ROOT / "uv.lock"
    m = hashlib.sha256()
    with open(uv_lock_path, "rb") as f:
        m.update(f.read())
    metadata["uv_lock_sha256"] = m.hexdigest()
    with open(run_dir / "metadata.json", "w") as file:
        json.dump(metadata, file, indent = 4)
    
    return None