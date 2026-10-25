"""Executive summary: written by Claude when ANTHROPIC_API_KEY is set, otherwise a
deterministic template built from the same numbers (so the pipeline always runs)."""
import json
import os

from .config import REPORTS

PROMPT = """You are a senior business analyst presenting to the VP of Retail Finance at an
Amazon-style e-commerce company. Using ONLY the JSON below (INR, millions unless stated), write a
one-page executive summary in markdown with: (1) headline, (2) three profit drivers with numbers,
(3) where we lose money and why, (4) three prioritised, quantified recommendations (use the
scenario figures), (5) model performance in plain English, (6) one honest limitation (the data is
synthetic and discount scenarios assume 15% volume loss). Never invent numbers.

DATA:
{data}
"""


def _fallback(a: dict, m: dict) -> str:
    k = a["kpis"]; cats = a["by_category"]
    best = max(cats, key=lambda c: cats[c]["profit"]); worst = min(cats, key=lambda c: cats[c]["margin_pct"])
    top_margin = max(cats, key=lambda c: cats[c]["margin_pct"])
    sc = next(s for s in a["scenarios"] if s["cap"] == 25)
    clf = m["classification"][m["best_classifier"]]
    cm = m["test_confusion"]
    return f"""# Executive Summary - Marketplace Profitability

**Headline:** {k['orders']:,} orders generated INR {k['net_revenue_m']}M net revenue and INR {k['profit_m']}M profit
(**{k['margin_pct']}% margin**). {k['loss_making_pct']}% of orders lose money, and {a['loss_explained_pct']}% of those
involve either a discount above 20% or a customer return.

## What drives profit
1. **Category mix.** {best} contributes the most profit (INR {cats[best]['profit']/1e6:.1f}M); {top_margin} earns the best margin
   ({cats[top_margin]['margin_pct']:.1f}%) while {worst} earns the thinnest ({cats[worst]['margin_pct']:.1f}%).
2. **Discount depth.** Loss-making rate climbs from {a['loss_rate_by_discount']['0-10%']}% at 0-10% discount to
   {a['loss_rate_by_discount']['40%+']}% above 40%. Orders discounted over 20% *destroy* INR
   {abs(a['profit_by_discount_m']['20-30%'] + a['profit_by_discount_m']['30-40%'] + a['profit_by_discount_m']['40%+']):.1f}M of profit.
3. **Seasonality.** Festive months (Jul, Oct, Nov) average {a['festive_avg_discount']}% discount vs {a['regular_avg_discount']}% normally, cutting
   margin from {a['regular_margin_pct']}% to **{a['festive_margin_pct']}%** despite record revenue.

## Where we lose money
- **Returns:** {a['returns']['rate_pct']}% of orders are returned, costing INR {a['returns']['cash_loss_m']}M; {a['returns']['worst_category']} is worst at {a['returns']['worst_category_rate']}%.
- **Pricing leakage:** INR {a['pricing_leakage_m']}M lost on orders that were *kept* by customers but sold below cost-to-serve.
- **Logistics:** Same-day orders earn INR {a['by_shipping_mode']['Same-Day']:.0f} per order vs INR {a['by_shipping_mode']['Standard']:.0f} for standard;
  North-East earns INR {a['by_region']['North-East']:.0f} vs INR {a['by_region']['North']:.0f} in the North.

## Recommendations
1. **Cap discounts at 25%** (affects {sc['pct_orders_affected']}% of orders). Even assuming {int(sc['volume_loss']*100)}% of affected orders are lost,
   profit rises by **INR {sc['uplift_m']}M (+{sc['uplift_pct']}%)**.
2. **Use the loss-risk model at pricing time** to block or reprice orders flagged as likely loss-makers.
3. **Attack returns** in {a['returns']['worst_category']} (size guidance, quality checks) and add a return-risk surcharge for repeat returners.

## Model performance
The best model ({m['best_classifier']}) predicts whether an order will lose money with **{clf['test_accuracy']*100:.1f}% accuracy**
on a chronological hold-out ({m['split']['test_period']}), versus a {m['baseline_accuracy']*100:.1f}% majority-class baseline;
precision {clf['test_precision']*100:.1f}%, recall {clf['test_recall']*100:.1f}%, AUC {clf['test_auc']:.3f}.
Of {cm['fn'] + cm['tp']} true loss-making orders it caught {cm['tp']}; {m['missed_losses']['pct_caused_by_returns']}% of the misses were
caused by customer returns, which cannot be known at order time.

## Limitations
Data is simulated, so absolute figures illustrate the method, not Amazon's real economics. Discount scenarios
assume 15% volume loss on repriced orders; real elasticity should be measured with an A/B test.
"""


def generate_summary(analysis: dict, models: dict) -> str:
    payload = json.dumps({"analysis": analysis, "model": models}, indent=1, default=str)
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        try:
            import anthropic
            msg = anthropic.Anthropic(api_key=key).messages.create(
                model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"), max_tokens=1800,
                messages=[{"role": "user", "content": PROMPT.format(data=payload)}])
            print("  AI summary written by Claude.")
            return msg.content[0].text
        except Exception as e:
            print(f"  LLM unavailable ({type(e).__name__}); using template summary.")
    return _fallback(analysis, models)


def save_summary(analysis, models):
    path = REPORTS / "executive_summary.md"
    path.write_text(generate_summary(analysis, models), encoding="utf-8")
    print(f"  Saved {path.relative_to(REPORTS.parent)}")
