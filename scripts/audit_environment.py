import importlib.util
from datetime import datetime
import platform
import sys

# Mapping: "Install Name": "Actual Import Name"
LIBRARIES = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "mistral-common": "mistral_common",  # Fixes hyphen issue
    "peft": "peft",
    "safetensors": "safetensors",
    "pyyaml": "yaml",                    # Fixes pyyaml -> yaml issue
    "pydantic": "pydantic"
}

def audit_os(report_file):
    os_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }
    print(f"OS Audit results: {os_info}\n")

    # Use append mode ('a') so we don't wipe the header
    with open(report_file, "a", encoding="utf-8") as f:
        f.write("--- OS Audit ---\n")
        for key, val in os_info.items():
            f.write(f"{key}: {val}\n")
        f.write("\n")

def audit_python(report_file):
    pyver = sys.version
    print(f"Python version: {pyver}\n")

    with open(report_file, "a", encoding="utf-8") as f:
        f.write("--- Python Audit ---\n")
        f.write(f"Python version: {pyver}\n\n")

def audit_environment(report_file):
    results = {
        "failed": [],
        "successful": [],
    }

    print("=" * 40)
    print(f"Trying to import libs: {list(LIBRARIES.keys())}")
    print("=" * 40)

    for install_name, import_name in LIBRARIES.items():
        if importlib.util.find_spec(import_name) is not None:
            try:
                __import__(import_name)
                results["successful"].append(install_name)
                print(f"[SUCCESS] {install_name}")
            except Exception as e:
                results["failed"].append((install_name, f"Import error: {e}"))
                print(f"[FAILED] {install_name}: {e}")
        else:
            results["failed"].append((install_name, "Module not found"))
            print(f"[FAILED] {install_name}: Not found, likely not installed.")

    # Write environment results to file using append mode ('a')
    with open(report_file, "a", encoding="utf-8") as f:
        f.write("--- Environment Audit ---\n")
        f.write(f"Successfully Imported ({len(results['successful'])}):\n")
        for lib in results["successful"]:
            f.write(f"  - {lib}\n")
            
        f.write(f"\nFailed to Import ({len(results['failed'])}):\n")
        for lib, reason in results["failed"]:
            f.write(f"  - {lib} ({reason})\n")
        f.write("\n")

    return "torch" in results["successful"]

def audit_torch(report_file):
    import torch
    version, cuda, cuda_av = torch.__version__, torch.version.cuda, torch.cuda.is_available()
    print("\nPyTorch Version:", version)
    print("CUDA Version Built Against:", cuda)
    print("Is CUDA Available:", cuda_av)
    
    bf_support = False
    if cuda_av:
        bf_support = torch.cuda.is_bf16_supported()
        print("Torch supports bf16:", bf_support)

    # Append torch details to file
    with open(report_file, "a", encoding="utf-8") as f:
        f.write("--- Torch / CUDA Audit ---\n")
        f.write(f"PyTorch Version: {version}\n")
        f.write(f"CUDA Version Built Against: {cuda}\n")
        f.write(f"Is CUDA Available: {cuda_av}\n")
        if cuda_av:
            f.write(f"Torch supports bf16: {bf_support}\n")
        else:
            f.write("Torch supports bf16: N/A\n")
        f.write("\n")

def make_audit_file():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    import os
    os.makedirs("../reports", exist_ok=True)
    report_filename = f"reports/audit_report_{timestamp}.txt"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write("=== Python Environment Audit Report ===\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    return report_filename

if __name__ == "__main__":
    print("Starting audit...")
    report_file = make_audit_file()
    
    audit_os(report_file)
    audit_python(report_file)  # Now integrated into the run
    
    has_torch = audit_environment(report_file)
    if has_torch:
        audit_torch(report_file)
    else:
        print("\nCould not audit CUDA: torch is uninstalled or failed to import.")
    
    print(f"\nAudit complete. Saved to {report_file}")