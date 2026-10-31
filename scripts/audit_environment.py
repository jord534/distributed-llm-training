# This outputs a .txt audit of our machine and our dependencies
# the aim is to know and check what we run on and what we run with

import importlib.util
from datetime import datetime # so we can timestamp the audit

LIBRARIES = [
    "torch",
    "transformers",
    "datasets",
    "mistral-common",
    "peft",
    "safetensors",
    "pyyaml",
    "pydantics"
]


def audit_environment():
    return None



if __name__ == "__main__":
    audit_environment()