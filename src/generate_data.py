"""Simulate two years of order-level data for an Amazon-style retail business.

Every order carries a full P&L:
    profit = net revenue - COGS - fulfilment - shipping - marketing - return costs

The simulation encodes realistic economics (thin electronics margins, heavy fashion
returns, festive-season discounting, costly same-day shipping, repeat returners).
It is SYNTHETIC: swap in a real order export with the same columns to analyse real data.
"""
import numpy as np
import pandas as pd

from .config import DATA_FILE, SEED

# share, median list price (INR), COGS/list ratio, base return rate, ad-spend rate, base ship cost
CATEGORIES = {
    "Electronics":            (0.20,  9000, 0.73, 0.080, 0.020,  90),
    "Fashion":                (0.18,  1400, 0.50, 0.130, 0.060,  45),
    "Home & Kitchen":         (0.16,  2400, 0.58, 0.050, 0.040,  90),
    "Beauty & Personal Care": (0.10,   700, 0.42, 0.040, 0.070,  30),
    "Books":                  (0.08,   450, 0.60, 0.020, 0.030,  25),
    "Grocery":                (0.10,   550, 0.72, 0.020, 0.025,  25),
    "Toys & Games":           (0.06,  1100, 0.55, 0.050, 0.050,  50),
    "Sports & Fitness":       (0.06,  1900, 0.58, 0.060, 0.045,  70),
    "Appliances":             (0.06, 22000, 0.75, 0.060, 0.015, 400),
}
CAT_DISCOUNT_SHIFT = {"Electronics": 3, "Fashion": 8, "Grocery": -4, "Books": -3,
                      "Appliances": 2, "Beauty & Personal Care": 2}
REGIONS = {"North": (0.24, 1.00), "South": (0.24, 0.95), "West": (0.22, 0.95),
           "East": (0.14, 1.25), "Central": (0.10, 1.15), "North-East": (0.06, 1.55)}
MODE_MULT = {"Standard": 1.0, "Expedited": 1.5, "Same-Day": 2.2}


def generate(n_orders: int = 40_000, n_customers: int = 16_000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    # ---- customers: segment and personal return propensity (a known, historical feature)
    cust = pd.DataFrame({
        "customer_id": [f"CUST{i:06d}" for i in range(n_customers)],
        "customer_segment": rng.choice(["Prime", "Standard"], n_customers, p=[0.55, 0.45]),
        "customer_return_rate": rng.beta(1.2, 16, n_customers).round(3),
    })
    idx = rng.integers(0, n_customers, n_orders)
    o = cust.iloc[idx].reset_index(drop=True)

    # ---- order dates with growth, weekend lift and festive peaks
    days = pd.date_range("2024-01-01", "2025-12-31")
    t = np.arange(len(days))
    month_mult = {7: 1.35, 10: 1.8, 11: 1.6, 12: 1.2}
    w = (1 + 0.0007 * t) * np.array([month_mult.get(d.month, 1.0) for d in days]) \
        * np.where(days.dayofweek >= 5, 1.12, 1.0)
    o["order_date"] = rng.choice(days, n_orders, p=w / w.sum())
    o = o.sort_values("order_date").reset_index(drop=True)
    o["order_id"] = [f"AMZ-{i:07d}" for i in range(n_orders)]
    month = o["order_date"].dt.month.to_numpy()

    # ---- product / order attributes
    names = list(CATEGORIES)
    o["category"] = rng.choice(names, n_orders, p=[CATEGORIES[c][0] for c in names])
    o["region"] = rng.choice(list(REGIONS), n_orders, p=[v[0] for v in REGIONS.values()])
    o["fulfilment"] = rng.choice(["FBA", "Merchant"], n_orders, p=[0.62, 0.38])
    prime = (o.customer_segment == "Prime").to_numpy()
    u = rng.random(n_orders)
    same = np.where(prime, 0.20, 0.07); exp = np.where(prime, 0.35, 0.20)
    o["shipping_mode"] = np.select([u < same, u < same + exp], ["Same-Day", "Expedited"], "Standard")
    o["quantity"] = rng.choice([1, 2, 3, 4], n_orders, p=[0.70, 0.20, 0.07, 0.03])

    med = o.category.map({c: v[1] for c, v in CATEGORIES.items()}).to_numpy()
    o["list_price"] = (med * rng.lognormal(0, 0.45, n_orders)).round(0).clip(99)
    festive = np.isin(month, [7, 10, 11])
    disc = rng.gamma(2.2, 5.0, n_orders) + np.where(festive, 6, 0) \
        + o.category.map(CAT_DISCOUNT_SHIFT).fillna(0).to_numpy()
    o["discount_pct"] = disc.clip(0, 60).round(1)
    o["selling_price"] = (o.list_price * (1 - o.discount_pct / 100)).round(2)
    cost_ratio = o.category.map({c: v[2] for c, v in CATEGORIES.items()}).to_numpy()
    o["unit_cost"] = (o.list_price * cost_ratio * rng.lognormal(0, 0.05, n_orders)).round(2)

    # ---- costs known at order time
    gross = o.selling_price * o.quantity
    o["gross_sales"] = gross.round(2)
    base_ship = o.category.map({c: v[5] for c, v in CATEGORIES.items()}).to_numpy()
    reg_mult = o.region.map({r: v[1] for r, v in REGIONS.items()}).to_numpy()
    ship = base_ship * o.shipping_mode.map(MODE_MULT).to_numpy() * reg_mult \
        * (1 + 0.25 * (o.quantity - 1)) * rng.lognormal(0, 0.08, n_orders) \
        * np.where(o.fulfilment == "Merchant", 1.10, 1.0)
    o["shipping_cost"] = ship.round(2)
    o["fulfilment_cost"] = np.where(o.fulfilment == "FBA", 32 + 0.022 * gross,
                                    22 + 0.015 * gross).round(2)
    ad_rate = o.category.map({c: v[4] for c, v in CATEGORIES.items()}).to_numpy()
    o["marketing_cost"] = (gross * ad_rate * rng.gamma(2, 0.5, n_orders)).round(2)

    # ---- outcome: returns (unknown at order time)
    base_ret = o.category.map({c: v[3] for c, v in CATEGORIES.items()}).to_numpy()
    p_ret = (0.35 * base_ret + 0.45 * o.customer_return_rate + 0.0006 * o.discount_pct
             + 0.01 * (o.fulfilment == "Merchant")).clip(0, 0.7)
    returned = rng.random(n_orders) < p_ret
    o["returned"] = returned.astype(int)

    # ---- P&L
    cogs = o.unit_cost * o.quantity
    o["cogs"] = cogs.round(2)
    o["net_revenue"] = (gross * (1 - o.returned)).round(2)
    o["cogs_recognised"] = (cogs * np.where(returned, 0.40, 1.0)).round(2)   # 40% written off on returns
    o["return_shipping_cost"] = (0.8 * o.shipping_cost * o.returned).round(2)
    o["profit"] = (o.net_revenue - o.cogs_recognised - o.fulfilment_cost - o.shipping_cost
                   - o.marketing_cost - o.return_shipping_cost).round(2)
    o["profit_margin_pct"] = (o.profit / gross * 100).round(2)
    o["is_loss_making"] = (o.profit < 0).astype(int)

    cols = ["order_id", "order_date", "customer_id", "customer_segment", "customer_return_rate",
            "category", "region", "fulfilment", "shipping_mode", "quantity", "list_price",
            "discount_pct", "selling_price", "unit_cost", "gross_sales", "shipping_cost",
            "fulfilment_cost", "marketing_cost", "returned", "net_revenue", "cogs", "cogs_recognised",
            "return_shipping_cost", "profit", "profit_margin_pct", "is_loss_making"]
    return o[cols]


def load_or_create() -> pd.DataFrame:
    if not DATA_FILE.exists():
        generate().to_csv(DATA_FILE, index=False)
    return pd.read_csv(DATA_FILE, parse_dates=["order_date"])


if __name__ == "__main__":
    df = generate()
    df.to_csv(DATA_FILE, index=False)
    print(f"Saved {len(df):,} orders -> {DATA_FILE}")
