"""Business analysis: KPIs, profit drivers, charts and what-if scenarios."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FESTIVE_MONTHS, FIG, GREY, NAVY, ORANGE, RED, TEAL

DISC_BINS = [-0.1, 10, 20, 30, 40, 100]
DISC_LABELS = ["0-10%", "10-20%", "20-30%", "30-40%", "40%+"]


def _style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"axes.titleweight": "bold", "axes.titlesize": 12, "axes.spines.top": False,
                         "axes.spines.right": False})


def _save(name):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(FIG / name, dpi=150, bbox_inches="tight"); plt.close()


def _m(x):  # INR millions
    return float(x) / 1e6


# ----------------------------------------------------------------------------- tables
def kpis(df: pd.DataFrame) -> dict:
    net = df.net_revenue.sum()
    return {"orders": int(len(df)), "gross_sales_m": round(_m(df.gross_sales.sum()), 2),
            "net_revenue_m": round(_m(net), 2), "profit_m": round(_m(df.profit.sum()), 2),
            "margin_pct": round(df.profit.sum() / net * 100, 2) if net else 0.0,
            "loss_making_pct": round(df.is_loss_making.mean() * 100, 1),
            "return_rate_pct": round(df.returned.mean() * 100, 1),
            "avg_profit_per_order": round(df.profit.mean(), 1)}


def by_dim(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col, observed=True).agg(
        orders=("profit", "size"), net_revenue=("net_revenue", "sum"), profit=("profit", "sum"),
        loss_pct=("is_loss_making", "mean"), return_pct=("returned", "mean"),
        avg_discount=("discount_pct", "mean"), profit_per_order=("profit", "mean"))
    g["margin_pct"] = g.profit / g.net_revenue * 100
    g["loss_pct"] *= 100; g["return_pct"] *= 100
    return g.sort_values("profit", ascending=False)


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    m = df.assign(month=df.order_date.dt.to_period("M").dt.to_timestamp()).groupby("month").agg(
        net_revenue=("net_revenue", "sum"), profit=("profit", "sum"), orders=("profit", "size"),
        avg_discount=("discount_pct", "mean"))
    m["margin_pct"] = m.profit / m.net_revenue * 100
    return m


def discount_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(band=pd.cut(df.discount_pct, DISC_BINS, labels=DISC_LABELS))
    g = d.groupby(["category", "band"], observed=True).agg(p=("profit", "sum"), r=("net_revenue", "sum"))
    return (g.p / g.r * 100).unstack("band").reindex(columns=DISC_LABELS)


def waterfall_parts(df: pd.DataFrame) -> list:
    refunds = (df.gross_sales * df.returned).sum()
    return [("Gross sales", df.gross_sales.sum()), ("Refunds", -refunds),
            ("COGS", -df.cogs_recognised.sum()), ("Fulfilment", -df.fulfilment_cost.sum()),
            ("Shipping", -(df.shipping_cost.sum() + df.return_shipping_cost.sum())),
            ("Marketing", -df.marketing_cost.sum())]


def discount_cap_scenario(df: pd.DataFrame, cap: float, volume_loss: float = 0.15) -> dict:
    """What if discounts were capped at `cap`%?  Affected orders are repriced and a share
    `volume_loss` of them is assumed to be lost (price elasticity assumption)."""
    aff = df.discount_pct > cap
    new_sp = df.list_price * (1 - cap / 100)
    new_gross = new_sp * df.quantity
    new_net = new_gross * (1 - df.returned)
    new_profit = (new_net - df.cogs_recognised - df.fulfilment_cost - df.shipping_cost
                  - df.marketing_cost - df.return_shipping_cost)
    expected = np.where(aff, (1 - volume_loss) * new_profit, df.profit)
    base, new = float(df.profit.sum()), float(expected.sum())
    return {"cap": cap, "volume_loss": volume_loss, "orders_affected": int(aff.sum()),
            "pct_orders_affected": round(float(aff.mean() * 100), 1),
            "baseline_profit_m": round(_m(base), 2), "new_profit_m": round(_m(new), 2),
            "uplift_m": round(_m(new - base), 2), "uplift_pct": round((new - base) / abs(base) * 100, 1)}


# ----------------------------------------------------------------------------- charts
def run_eda(df: pd.DataFrame) -> dict:
    _style()
    out = {"kpis": kpis(df)}

    # 01 monthly trend
    m = monthly(df)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(m.index, m.net_revenue / 1e6, width=22, color="#D5DAE0", label="Net revenue (INR M)")
    for dt in m.index:
        if dt.month in FESTIVE_MONTHS:
            ax.axvspan(dt - pd.Timedelta(days=15), dt + pd.Timedelta(days=15), color=ORANGE, alpha=.18)
    ax.set_ylabel("Net revenue (INR M)"); ax.grid(axis="x", visible=False)
    ax2 = ax.twinx(); ax2.plot(m.index, m.margin_pct, color=NAVY, marker="o", lw=2, label="Profit margin (%)")
    ax2.set_ylabel("Profit margin (%)"); ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax.set_title("Revenue peaks in festive months (shaded) - but margin dips as discounting rises")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    _save("01_monthly_trend.png")
    fest = df.order_date.dt.month.isin(FESTIVE_MONTHS)
    out["festive_margin_pct"] = round(df[fest].profit.sum() / df[fest].net_revenue.sum() * 100, 2)
    out["regular_margin_pct"] = round(df[~fest].profit.sum() / df[~fest].net_revenue.sum() * 100, 2)
    out["festive_avg_discount"] = round(df[fest].discount_pct.mean(), 1)
    out["regular_avg_discount"] = round(df[~fest].discount_pct.mean(), 1)

    # 02 waterfall
    parts = waterfall_parts(df)
    labels = [p[0] for p in parts] + ["Profit"]
    vals = [p[1] / 1e6 for p in parts]
    fig, ax = plt.subplots(figsize=(10, 4.6)); run = 0
    for i, v in enumerate(vals):
        bottom = run if i else 0
        ax.bar(i, v, bottom=bottom, color=NAVY if i == 0 else RED, width=0.6)
        top = max(bottom, bottom + v)
        ax.text(i, top + 4, f"{v:+,.1f}" if i else f"{v:,.1f}", ha="center", fontsize=9)
        run += v
    ax.bar(len(vals), run, color=ORANGE, width=0.6)
    ax.text(len(vals), run + 4, f"{run:,.1f}\n({run / vals[0] * 100:.1f}% of sales)", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylim(0, vals[0] * 1.15); ax.grid(axis="x", visible=False)
    ax.set_ylabel("INR millions"); ax.set_title("Where does each rupee of sales go? Profit waterfall")
    _save("02_profit_waterfall.png")
    out["waterfall"] = {p[0]: round(p[1] / 1e6, 2) for p in parts}

    # 03 category profit & margin
    cat = by_dim(df, "category").sort_values("profit")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    ax[0].barh(cat.index, cat.profit / 1e6, color=[TEAL if v >= 0 else RED for v in cat.profit])
    ax[0].set(title="Total profit (INR M)")
    ax[1].barh(cat.index, cat.margin_pct, color=[ORANGE if v >= 5 else GREY for v in cat.margin_pct])
    ax[1].axvline(out["kpis"]["margin_pct"], ls="--", color=NAVY); ax[1].set(title="Profit margin (%) - dashed = company average")
    _save("03_category_profitability.png")
    out["by_category"] = {k: {kk: round(float(vv), 2) for kk, vv in v.items()} for k, v in cat.sort_values("profit", ascending=False).iterrows()}

    # 04 discount x category margin heatmap
    hm = discount_heatmap(df).loc[cat.index[::-1]]
    plt.figure(figsize=(8, 4.8))
    sns.heatmap(hm, annot=True, fmt=".0f", cmap="RdYlGn", center=0, cbar_kws={"label": "Margin %"})
    plt.title("Margin (%) by category and discount depth"); plt.xlabel("Discount band"); plt.ylabel("")
    _save("04_discount_margin_heatmap.png")

    # 05 loss rate vs discount
    d = df.assign(band=pd.cut(df.discount_pct, DISC_BINS, labels=DISC_LABELS))
    lb = d.groupby("band", observed=True).agg(loss=("is_loss_making", "mean"), share=("profit", "size"),
                                              profit=("profit", "sum"))
    lb["loss"] *= 100; lb["share"] = lb.share / lb.share.sum() * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(lb.index, lb.loss, color=[TEAL, TEAL, ORANGE, RED, RED])
    for i, (l, s) in enumerate(zip(lb.loss, lb.share)):
        ax.text(i, l + 1.5, f"{l:.0f}%\n({s:.0f}% of orders)", ha="center", fontsize=8)
    ax.set(ylabel="Loss-making orders (%)", title="Deeper discounts = far more loss-making orders", ylim=(0, 105))
    _save("05_loss_rate_by_discount.png")
    out["loss_rate_by_discount"] = {k: round(float(v), 1) for k, v in lb.loss.items()}
    out["profit_by_discount_m"] = {k: round(_m(v), 2) for k, v in lb.profit.items()}

    # 06 operations: region / fulfilment / shipping mode
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for a, col, title in zip(ax, ["region", "fulfilment", "shipping_mode"],
                             ["Profit per order by region", "By fulfilment", "By shipping mode"]):
        t = by_dim(df, col).sort_values("profit_per_order")
        weak = t.profit_per_order.min() < 0.9 * t.profit_per_order.max()   # only flag meaningful gaps
        cols = [RED if (weak and v == t.profit_per_order.min()) else TEAL for v in t.profit_per_order]
        a.bar(t.index, t.profit_per_order, color=cols)
        for x, v in zip(t.index, t.profit_per_order):
            a.text(x, v + 5, f"{v:,.0f}", ha="center", fontsize=8)
        a.set_title(title); a.set_ylabel("INR / order"); a.tick_params(axis="x", rotation=25)
    _save("06_operations.png")
    out["by_region"] = {k: round(float(v), 1) for k, v in by_dim(df, "region").profit_per_order.items()}
    out["by_shipping_mode"] = {k: round(float(v), 1) for k, v in by_dim(df, "shipping_mode").profit_per_order.items()}
    out["by_fulfilment"] = {k: round(float(v), 1) for k, v in by_dim(df, "fulfilment").profit_per_order.items()}
    out["by_segment_margin"] = {k: round(float(v), 2) for k, v in by_dim(df, "customer_segment").margin_pct.items()}

    # 07 returns
    r = by_dim(df, "category").sort_values("return_pct")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(r.index, r.return_pct, color=ORANGE); ax.set(title="Return rate by category (%)", xlabel="% of orders returned")
    _save("07_returns.png")
    ret = df[df.returned == 1]
    out["returns"] = {"rate_pct": round(df.returned.mean() * 100, 1), "cash_loss_m": round(-_m(ret.profit.sum()), 2),
                      "worst_category": r.index[-1], "worst_category_rate": round(float(r.return_pct.iloc[-1]), 1)}

    # headline pricing leakage: loss on non-returned orders priced too low
    nr = df[(df.returned == 0) & (df.profit < 0)]
    out["pricing_leakage_m"] = round(-_m(nr.profit.sum()), 2)
    loss = df[df.is_loss_making == 1]
    out["loss_explained_pct"] = round(float(((loss.discount_pct > 20) | (loss.returned == 1)).mean() * 100), 1)
    out["loss_orders_over_30_discount_pct"] = round(float((df[df.is_loss_making == 1].discount_pct > 30).mean() * 100), 1)

    out["scenarios"] = [discount_cap_scenario(df, c) for c in (20, 25, 30, 35)]
    return out
