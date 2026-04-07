from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.instability_control.method_benchmark import main


if __name__ == "__main__":
    print("[deprecated] route_comparison.py now forwards to method_benchmark.py")
    main()
