"""
Environment and Package Verification Script
Solar Panel Fault Detection and Inspection System
Phase 1: Initial Project Setup
"""

import sys
import importlib

packages_to_test = [
    ("Python", sys.version.split()[0], None),
    ("torch", None, "torch"),
    ("torchvision", None, "torchvision"),
    ("numpy", None, "numpy"),
    ("pandas", None, "pandas"),
    ("PIL", None, "PIL"),
    ("cv2", None, "cv2"),
    ("sklearn", None, "sklearn"),
    ("matplotlib", None, "matplotlib"),
    ("seaborn", None, "seaborn"),
    ("albumentations", None, "albumentations"),
    ("yaml", None, "yaml"),
]

print("=" * 65)
print(" Solar Panel Fault Detection AI - Phase 1 Verification")
print("=" * 65)

all_passed = True
results = []

for name, ver, mod_name in packages_to_test:
    if mod_name is None:
        print(f"[PASS] {name:<16} : version {ver}")
        results.append((name, ver, "OK"))
        continue
    try:
        mod = importlib.import_module(mod_name)
        v = getattr(mod, "__version__", "Available (no __version__)")
        print(f"[PASS] {name:<16} : version {v}")
        results.append((name, v, "OK"))
    except Exception as e:
        print(f"[FAIL] {name:<16} : FAILED - {e}")
        results.append((name, "N/A", f"FAIL: {e}"))
        all_passed = False

print("-" * 65)

# Detailed framework checks
try:
    import torch
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (No CUDA device detected)"
    print(f"PyTorch CUDA Available: {cuda_available}")
    print(f"PyTorch Active Device : {device_name}")
    x = torch.rand(3, 3)
    print(f"PyTorch Tensor Compute Test: PASSED (Shape: {x.shape})")
except Exception as e:
    print(f"PyTorch Tensor Compute Test: FAILED ({e})")
    all_passed = False

try:
    import cv2
    print(f"OpenCV Build Info : {cv2.__version__}")
except Exception as e:
    all_passed = False

try:
    import albumentations as A
    print(f"Albumentations Version : {A.__version__}")
except Exception as e:
    all_passed = False

print("=" * 65)
if all_passed:
    print("ALL CHECKS PASSED: Phase 1 environment is ready for ML development.")
    sys.exit(0)
else:
    print("SOME CHECKS FAILED. Please review the errors above.")
    sys.exit(1)
