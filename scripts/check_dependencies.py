#!/usr/bin/env python3
"""
TrustLens Training Dependency Checker.
Audits runtime environment, ML libraries, and hardware acceleration
without modifying packages or executing training.
"""

import sys

def main():
    print("============================================================")
    print("TRUSTLENS ENVIRONMENT & DEPENDENCY CHECK")
    print("============================================================")

    # 1. Python
    print(f"Python:       {sys.version.split()[0]} ({sys.executable})")

    # 2. PyTorch
    torch_installed = False
    cuda_available = False
    gpu_name = "None"
    try:
        import torch
        torch_installed = True
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
        print(f"PyTorch:      {torch_version}")
    except ImportError:
        print("PyTorch:      MISSING")

    # 3. Transformers
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("Transformers: MISSING")

    # 4. Scikit-learn
    try:
        import sklearn
        print(f"scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("scikit-learn: MISSING")

    # 5. LightGBM
    lightgbm_installed = False
    try:
        import lightgbm
        lightgbm_installed = True
        print(f"LightGBM:     INSTALLED ({lightgbm.__version__})")
    except ImportError:
        print("LightGBM:     MISSING")

    # 6. Hardware Acceleration
    print(f"CUDA:         {'AVAILABLE' if cuda_available else 'NOT AVAILABLE'}")
    print(f"GPU:          {gpu_name}")
    print("============================================================")

    if not lightgbm_installed:
        print()
        print("LIGHTGBM PACKAGE MISSING")
        print("Install command:")
        print("pip install lightgbm")
        print("============================================================")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
