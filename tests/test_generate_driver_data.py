import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.generate_driver_data import generate_driver_data


EXPECTED_COLUMNS = [
    "trip_id",
    "rain",
    "rush_hour",
    "base_price",
    "delta_price",
    "price",
    "driver_accept_prob",
    "driver_accept",
]


def test_generate_driver_data_has_expected_shape_and_columns():
    df = generate_driver_data(n_samples=500, seed=123)

    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 500
    assert df.isna().sum().sum() == 0
    assert df["trip_id"].is_unique


def test_generate_driver_data_is_deterministic_for_same_seed():
    first = generate_driver_data(n_samples=250, seed=42)
    second = generate_driver_data(n_samples=250, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_binary_columns_only_contain_zero_or_one():
    df = generate_driver_data(n_samples=500, seed=123)

    for column in ["rain", "rush_hour", "driver_accept"]:
        assert set(df[column].unique()).issubset({0, 1})


def test_price_identity_and_positive_values():
    df = generate_driver_data(n_samples=500, seed=123)

    pd.testing.assert_series_equal(
        df["price"],
        df["base_price"] + df["delta_price"],
        check_names=False,
    )
    assert (df["base_price"] > 0).all()
    assert (df["delta_price"] > 0).all()
    assert (df["price"] > df["base_price"]).all()
    assert df["driver_accept_prob"].between(0, 1).all()


def test_linear_price_coefficients_are_positive_without_interaction():
    df = generate_driver_data(n_samples=3000, seed=42)
    model = LinearRegression()
    model.fit(df[["rain", "rush_hour"]], df["delta_price"])

    rain_coef, rush_coef = model.coef_
    assert rain_coef > 0
    assert rush_coef > 0
    assert 2500 <= rain_coef <= 4500
    assert 3500 <= rush_coef <= 6000


def test_logistic_acceptance_price_coefficient_is_positive():
    df = generate_driver_data(n_samples=3000, seed=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(df[["price", "rain", "rush_hour"]], df["driver_accept"])

    price_coef = model.coef_[0][0]
    assert price_coef > 0


def test_acceptance_rate_increases_across_price_bins():
    df = generate_driver_data(n_samples=3000, seed=42)
    df = df.assign(price_bin=pd.qcut(df["price"], q=4, labels=False))
    rates = df.groupby("price_bin", observed=True)["driver_accept"].mean()

    assert rates.iloc[-1] > rates.iloc[0]
