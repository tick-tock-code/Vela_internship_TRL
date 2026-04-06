from __future__ import annotations

import sys
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parents[3].parent
THINK_REASON_LEARN_ROOT = PROJECT_FOLDER / "think-reason-learn"

if THINK_REASON_LEARN_ROOT.exists() and str(THINK_REASON_LEARN_ROOT) not in sys.path:
    sys.path.insert(0, str(THINK_REASON_LEARN_ROOT))
