from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.instability_control.status_report import main


if __name__ == "__main__":
    print("[deprecated] final_report.py now forwards to status_report.py")
    main()
