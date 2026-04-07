from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.instability_control.evidence_map import main


if __name__ == "__main__":
    print("[deprecated] family_admission.py now forwards to evidence_map.py; Step 1 is evidence mapping, not sequential admission.")
    main()
