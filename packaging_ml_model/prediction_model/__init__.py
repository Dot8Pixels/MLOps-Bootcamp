import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PACKAGE_ROOT))
from prediction_model.config import config  # noqa: E402

with open(file=config.PACKAGE_ROOT / "VERSION", encoding="utf-8") as file:
    __version__ = file.read().strip()
