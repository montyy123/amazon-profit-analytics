"""Central paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
DATA_FILE = DATA_DIR / "amazon_orders.csv"
SEED = 42

# Brand-inspired palette
ORANGE, NAVY, TEAL, RED, GREY = "#FF9900", "#232F3E", "#146EB4", "#C0392B", "#8A8F98"
FESTIVE_MONTHS = (7, 10, 11)   # Prime Day + festive season

for _p in (DATA_DIR, REPORTS, FIG):
    _p.mkdir(parents=True, exist_ok=True)
