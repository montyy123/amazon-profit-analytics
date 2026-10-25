"""Run the whole pipeline:  python main.py"""
import json
import time

from src.analysis import run_eda
from src.config import FIG, REPORTS
from src.generate_data import load_or_create
from src.insights import save_summary
from src.model import run_models


def main():
    t0 = time.time()
    for f in FIG.glob("*.png"):          # avoid stale charts
        f.unlink()
    print("[0/3] Loading data")
    df = load_or_create()
    print(f"  {len(df):,} orders, {df.order_date.min():%d %b %Y} to {df.order_date.max():%d %b %Y}")
    print("[1/3] Business analysis")
    analysis = run_eda(df)
    k = analysis["kpis"]
    print(f"  Net revenue INR {k['net_revenue_m']}M | Profit INR {k['profit_m']}M | Margin {k['margin_pct']}% | Loss-making {k['loss_making_pct']}%")
    print("[2/3] Machine learning")
    models = run_models(df)
    print("[3/3] Executive summary")
    save_summary(analysis, models)
    (REPORTS / "metrics.json").write_text(json.dumps({"analysis": analysis, "model": models}, indent=2, default=str))
    clf = models["classification"][models["best_classifier"]]
    print(f"\nDone in {time.time() - t0:.0f}s | {models['best_classifier']} test accuracy {clf['test_accuracy']:.1%} "
          f"(baseline {models['baseline_accuracy']:.1%}) | AUC {clf['test_auc']:.3f}")


if __name__ == "__main__":
    main()
