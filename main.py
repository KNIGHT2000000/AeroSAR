"""
Top-level launcher for AeroSAR-Optimizer.
Enables running `python main.py` directly from the repository root workspace.
"""

import sys
from pathlib import Path

# Add aerosar_optimizer package to sys.path
package_dir = Path(__file__).resolve().parent / "aerosar_optimizer"
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

# Change directory context so config.yaml and outputs resolve cleanly
import os
os.chdir(package_dir)

from main import main

if __name__ == "__main__":
    main()
