"""Interactive dashboard:  streamlit run app.py"""
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import analysis as an
from src.config import FIG, NAVY, ORANGE, RED, REPORTS, TEAL
from src.generate_data import CATEGORIES, MODE_MULT, REGIONS, load_or_create
from src.model import FEATURES, add_features

st.set_page_config(page_title="Amazon Profit Analytics", page_icon="📦", layout="wide")


@st.cache_data
def load() -> pd.DataFrame:
    return load_or_create()


@st.cache_resource
def load_model():
    p = REPORTS / "loss_classifier.joblib"
    return joblib.load(p) if p.exists() else None


df_all, model = load(), load_model()
inr = lambda v: f"INR {v:,.1f}M"

st.title("📦 Amazon Marketplace Profit Analytics")
st.caption("Order-level P&L, discount economics and ML loss-risk prediction  |  synthetic data")

# ------------------------------------------------------------------ filters
st.sidebar.header("Filters")
dmin, dmax = df_all.order_date.min().date(), df_all.order_date.max().date()
rng = st.sidebar.date_input("Order date", (dmin, dmax), min_value=dmin, max_value=dmax)
if len(rng) != 2:
    st.info("Pick an end date to apply the range."); st.stop()
cats = st.sidebar.multiselect("Category", sorted(df_all.category.unique()), sorted(df_all.category.unique()))
regs = st.sidebar.multiselect("Region", sorted(df_all.region.unique()), sorted(df_all.region.unique()))
ful = st.sidebar.multiselect("Fulfilment", sorted(df_all.fulfilment.unique()), sorted(df_all.fulfilment.unique()))
seg = st.sidebar.multiselect("Customer segment", sorted(df_all.customer_segment.unique()), sorted(df_all.customer_segment.unique()))
df = df_all[df_all.order_date.dt.date.between(*rng) & df_all.category.isin(cats) & df_all.region.isin(regs)
            & df_all.fulfilment.isin(ful) & df_all.customer_segment.isin(seg)]
if df.empty:
    st.warning("No orders match these filters."); st.stop()

t1, t2, t3, t4, t5 = st.tabs(["📊 Overview", "🏷️ Pricing & discounts", "🚚 Operations & returns",
                              "🤖 Loss-risk predictor", "🧠 Model & AI summary"])

# ------------------------------------------------------------------ overview
with t1:
    k = an.kpis(df)
    c = st.columns(6)
    c[0].metric("Orders", f"{k['orders']:,}")
    c[1].metric("Net revenue", inr(k["net_revenue_m"]))
    c[2].metric("Profit", inr(k["profit_m"]))
    c[3].metric("Profit margin", f"{k['margin_pct']:.1f}%")
    c[4].metric("Loss-making orders", f"{k['loss_making_pct']:.1f}%")
    c[5].metric("Return rate", f"{k['return_rate_pct']:.1f}%")

    m = an.monthly(df)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=m.index, y=m.net_revenue / 1e6, name="Net revenue (INR M)", marker_color="#C9D1DB")
    fig.add_scatter(x=m.index, y=m.margin_pct, name="Margin (%)", mode="lines+markers",
                    line=dict(color=NAVY, width=3), secondary_y=True)
    fig.update_layout(title="Monthly revenue and margin", legend=dict(orientation="h", y=1.12), height=380)
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    parts = an.waterfall_parts(df)
    wf = go.Figure(go.Waterfall(
        x=[p[0] for p in parts] + ["Profit"], y=[p[1] / 1e6 for p in parts] + [None],
        measure=["absolute"] + ["relative"] * (len(parts) - 1) + ["total"],
        decreasing=dict(marker=dict(color=RED)), totals=dict(marker=dict(color=ORANGE)),
        increasing=dict(marker=dict(color=TEAL)), connector=dict(line=dict(color="#999"))))
    wf.update_layout(title="Profit waterfall (INR M)", height=380, showlegend=False)
    a.plotly_chart(wf, width="stretch")
    cat = an.by_dim(df, "category").reset_index()
    fc = px.bar(cat.sort_values("profit"), x="profit", y="category", orientation="h", color="margin_pct",
                color_continuous_scale="RdYlGn", title="Profit by category (colour = margin %)")
    fc.update_layout(height=380, xaxis_title="Profit (INR)", yaxis_title="")
    b.plotly_chart(fc, width="stretch")

# ------------------------------------------------------------------ pricing
with t2:
    a, b = st.columns(2)
    hm = an.discount_heatmap(df)
    a.plotly_chart(px.imshow(hm, text_auto=".0f", color_continuous_scale="RdYlGn", zmin=-60, zmax=60,
                             aspect="auto", title="Margin (%) by category and discount depth",
                             labels=dict(color="Margin %")).update_layout(height=420), width="stretch")
    d = df.assign(band=pd.cut(df.discount_pct, an.DISC_BINS, labels=an.DISC_LABELS))
    lb = d.groupby("band", observed=True).agg(loss=("is_loss_making", "mean"), orders=("profit", "size")).reset_index()
    lb["loss"] *= 100
    b.plotly_chart(px.bar(lb, x="band", y="loss", text=lb.loss.round(0).astype(int).astype(str) + "%",
                          hover_data=["orders"], color="loss", color_continuous_scale="RdYlGn_r",
                          title="Loss-making orders (%) by discount band").update_layout(height=420, coloraxis_showscale=False),
                   width="stretch")

    st.subheader("💡 What-if: cap the maximum discount")
    s1, s2 = st.columns(2)
    cap = s1.slider("Maximum discount allowed (%)", 10, 50, 25, step=5)
    vloss = s2.slider("Share of affected orders lost to higher prices (%)", 0, 50, 15, step=5) / 100
    sc = an.discount_cap_scenario(df, cap, vloss)
    r = st.columns(4)
    r[0].metric("Orders affected", f"{sc['pct_orders_affected']:.1f}%")
    r[1].metric("Current profit", inr(sc["baseline_profit_m"]))
    r[2].metric("Profit with cap", inr(sc["new_profit_m"]), f"{sc['uplift_m']:+.2f}M")
    r[3].metric("Uplift", f"{sc['uplift_pct']:+.1f}%")
    st.caption("Assumption-based simulation: affected orders are repriced at the cap; the chosen share is assumed lost.")

# ------------------------------------------------------------------ operations
with t3:
    o = st.columns(3)
    for col, dim, title in zip(o, ["region", "fulfilment", "shipping_mode"],
                               ["Profit per order by region", "By fulfilment", "By shipping mode"]):
        t = an.by_dim(df, dim).reset_index().sort_values("profit_per_order")
        col.plotly_chart(px.bar(t, x=dim, y="profit_per_order", text=t.profit_per_order.round(0),
                                title=title, color_discrete_sequence=[TEAL]).update_layout(height=340, yaxis_title="INR / order"),
                         width="stretch")
    a, b = st.columns(2)
    r = an.by_dim(df, "category").reset_index().sort_values("return_pct")
    a.plotly_chart(px.bar(r, x="return_pct", y="category", orientation="h", title="Return rate (%) by category",
                          color_discrete_sequence=[ORANGE]).update_layout(height=360, yaxis_title=""), width="stretch")
    seg_t = (df.groupby(["category", "shipping_mode"]).agg(orders=("profit", "size"), profit=("profit", "sum"),
             loss_pct=("is_loss_making", "mean")).reset_index().query("orders >= 30"))
    seg_t["loss_pct"] = (seg_t.loss_pct * 100).round(1); seg_t["profit"] = seg_t.profit.round(0)
    b.markdown("**Worst category x shipping segments by total profit**")
    b.dataframe(seg_t.sort_values("profit").head(8), hide_index=True, width="stretch")

# ------------------------------------------------------------------ predictor
with t4:
    if model is None:
        st.info("Run `python main.py` first to train the model.")
    else:
        st.write("Estimate whether a prospective order will **lose money**, before it ships.")
        c1, c2, c3 = st.columns(3)
        category = c1.selectbox("Category", list(CATEGORIES))
        region = c1.selectbox("Region", list(REGIONS))
        mode = c1.selectbox("Shipping mode", list(MODE_MULT))
        fulf = c2.selectbox("Fulfilment", ["FBA", "Merchant"])
        segment = c2.selectbox("Customer segment", ["Prime", "Standard"])
        qty = c2.slider("Quantity", 1, 4, 1)
        median = CATEGORIES[category][1]
        lp = c3.number_input("List price (INR)", 99, 100000, int(median), step=100)
        disc = c3.slider("Discount (%)", 0, 60, 20)
        ret = c3.slider("Customer's past return rate (%)", 0, 40, 6) / 100
        month = c3.selectbox("Order month", list(range(1, 13)), index=9)

        gross = lp * (1 - disc / 100) * qty
        ship = CATEGORIES[category][5] * MODE_MULT[mode] * REGIONS[region][1] * (1 + .25 * (qty - 1)) * (1.10 if fulf == "Merchant" else 1)
        row = pd.DataFrame([{
            "category": category, "region": region, "fulfilment": fulf, "shipping_mode": mode,
            "customer_segment": segment, "quantity": qty, "list_price": lp, "discount_pct": disc,
            "unit_cost": lp * CATEGORIES[category][2], "shipping_cost": ship,
            "fulfilment_cost": (32 + .022 * gross) if fulf == "FBA" else (22 + .015 * gross),
            "marketing_cost": gross * CATEGORIES[category][4], "customer_return_rate": ret, "order_month": month}])
        feat = add_features(row)
        p = float(model.predict_proba(feat[FEATURES])[0, 1])
        g = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%"},
                                   title={"text": "Probability this order loses money"},
                                   gauge={"axis": {"range": [0, 100]}, "bar": {"color": NAVY},
                                          "steps": [{"range": [0, 35], "color": "#CDEBD3"},
                                                    {"range": [35, 65], "color": "#FCE8B2"},
                                                    {"range": [65, 100], "color": "#F4C7C3"}]}))
        g.update_layout(height=300, margin=dict(t=60, b=0))
        a, b = st.columns([1, 1])
        a.plotly_chart(g, width="stretch")
        b.metric("Margin before returns", f"{feat.pre_return_margin_pct.iloc[0]:.1f}%")
        if p > 0.65:
            b.error("High risk: reduce the discount or reroute shipping before accepting this order.")
        elif p > 0.35:
            b.warning("Borderline: margin is thin, so a return would make this order unprofitable.")
        else:
            b.success("Low risk: healthy margin even after fulfilment, shipping and marketing costs.")
        b.caption("Shipping, fulfilment and ad costs are estimated from category benchmarks.")

# ------------------------------------------------------------------ model & summary
with t5:
    mp = REPORTS / "metrics.json"
    if mp.exists():
        res = json.loads(mp.read_text())["model"]
        best = res["classification"][res["best_classifier"]]
        c = st.columns(4)
        c[0].metric("Test accuracy", f"{best['test_accuracy']:.1%}", f"{(best['test_accuracy'] - res['baseline_accuracy']) * 100:+.1f} pts vs baseline")
        c[1].metric("Precision", f"{best['test_precision']:.1%}")
        c[2].metric("Recall", f"{best['test_recall']:.1%}")
        c[3].metric("ROC-AUC", f"{best['test_auc']:.3f}")
        st.caption(f"Chronological hold-out: {res['split']['test_period']} | best model: {res['best_classifier']} | "
                   f"{res['missed_losses']['pct_caused_by_returns']}% of missed losses were caused by customer returns")
        st.dataframe(pd.DataFrame(res["classification"]).T, width="stretch")
        i1, i2, i3 = st.columns(3)
        for col, f in zip((i1, i2, i3), ("08_roc_pr_curves.png", "09_confusion_matrix.png", "10_feature_importance.png")):
            if (FIG / f).exists():
                col.image(str(FIG / f), width="stretch")
    s = REPORTS / "executive_summary.md"
    if s.exists():
        st.divider(); st.markdown(s.read_text(encoding="utf-8"))
