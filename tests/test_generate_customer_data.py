import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

from src.generate_customer_data import generate_customer_data

EXPECTED_COLUMNS = [
    "trip_id",
    "customer_id",
    "trip_time",
    "origin_zone",
    "destination_zone",
    "origin_x",
    "origin_y",
    "destination_x",
    "destination_y",
    "distance_km",
    "duration_minute",
    "rain",
    "rush_hour",
    "base_price",
    "delta_price",
    "final_price",
    "customer_accept_prob",
    "customer_accept",
]

RUSH_HOURS = {7, 8, 9, 16, 17, 18}


def test_customer_dataset_schema():
    df = generate_customer_data(seed=42, n_samples=5000, n_customers=200)

    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 5000
    assert df.isna().sum().sum() == 0
    assert df["trip_id"].is_unique
    assert df["customer_id"].nunique() == 200


def test_generate_customer_data_is_deterministic():
    first = generate_customer_data(seed=42)
    second = generate_customer_data(seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_customer_distance_duration_base_price():
    df = generate_customer_data(seed=42)

    expected_distance = np.sqrt(
        (df["destination_x"] - df["origin_x"]) ** 2
        + (df["destination_y"] - df["origin_y"]) ** 2
    )
    np.testing.assert_allclose(df["distance_km"], expected_distance, atol=0.01)
    assert (df["distance_km"] >= 0.5).all()

    expected_base = 10000 + 6000 * df["distance_km"] + 500 * df["duration_minute"]
    np.testing.assert_allclose(df["base_price"], expected_base, atol=1.0)
    np.testing.assert_allclose(df["final_price"], df["base_price"] + df["delta_price"], atol=1.0)


def test_linear_regression_recovers_price_effects():
    df = generate_customer_data(seed=42)
    model = LinearRegression()
    model.fit(df[["rain", "rush_hour"]], df["delta_price"])

    beta0 = model.intercept_
    beta1, beta2 = model.coef_

    # Beta0 is around 2000, beta1 around 8000, beta2 around 12000 based on generation logic
    assert 1000 <= beta0 <= 3000
    assert 6500 <= beta1 <= 9500
    assert 10500 <= beta2 <= 13500


def test_logistic_regression_recovers_customer_acceptance_behavior():
    df = generate_customer_data(seed=42)
    model = LogisticRegression(solver='lbfgs')
    model.fit(df[["final_price", "rain", "rush_hour"]], df["customer_accept"])

    alpha_price, alpha_rain, alpha_rush = model.coef_[0]

    # Customers hate high prices
    assert alpha_price < 0
    
    # Customers accept higher prices during rain or rush hour
    assert alpha_rain > 0
    assert alpha_rush > 0

    # Model should have reasonable predictive power (AUC > 0.55)
    pred_prob = model.predict_proba(df[["final_price", "rain", "rush_hour"]])[:, 1]
    auc = roc_auc_score(df["customer_accept"], pred_prob)
    assert auc > 0.55
