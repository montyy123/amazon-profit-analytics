"""Run after `python main.py`:  pytest -q"""
import json

import numpy as np
import pytest

from src.config import REPORTS
from src.generate_data import generate
from src.model import FEATURES, FORBIDDEN, add_features


@pytest.fixture(scope="module")
def df():
    return generate(n_orders=3000, n_customers=1200)


def test_no_missing_values(df):
    assert df.isna().sum().sum() == 0


def test_order_ids_unique(df):
    assert df.order_id.is_unique


def test_pnl_identity_holds(df):
    recomputed = (df.net_revenue - df.cogs_recognised - df.fulfilment_cost
                  - df.shipping_cost - df.marketing_cost - df.return_shipping_cost)
    assert np.allclose(recomputed, df.profit, atol=0.05)


def test_returned_orders_have_no_revenue(df):
    assert (df.loc[df.returned == 1, "net_revenue"] == 0).all()


def test_no_target_leakage_in_features():
    assert not set(FEATURES) & FORBIDDEN


def test_feature_engineering_has_no_outcome_columns(df):
    out = add_features(df)
    assert set(FEATURES) <= set(out.columns)


def test_model_meets_accuracy_target():
    m = json.loads((REPORTS / "metrics.json").read_text())["model"]
    best = m["classification"][m["best_classifier"]]
    assert best["test_accuracy"] >= 0.90
    assert best["test_accuracy"] > m["baseline_accuracy"] + 0.10   # must beat the naive baseline clearly
