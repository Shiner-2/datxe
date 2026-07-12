import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.generate_driver_data import generate_driver_data


EXPECTED_COLUMNS = [
    "trip_id",
    "driver_id",
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
    "driver_accept_prob",
    "driver_accept",
]

RUSH_HOURS = {7, 8, 9, 16, 17, 18}


def _fit_rmse(train, test, features):
    train_x = pd.get_dummies(train[features], drop_first=True, dtype=float)
    test_x = pd.get_dummies(test[features], drop_first=True, dtype=float)
    test_x = test_x.reindex(columns=train_x.columns, fill_value=0)

    model = LinearRegression()
    model.fit(train_x, train["delta_price"])
    predicted = model.predict(test_x)
    return mean_squared_error(test["delta_price"], predicted) ** 0.5


def test_default_dataset_matches_latest_synthetic_plan_schema():
    df = generate_driver_data(seed=42)

    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 5000
    assert df.isna().sum().sum() == 0
    assert df["trip_id"].is_unique
    assert df["driver_id"].nunique() == 50


def test_generate_driver_data_is_deterministic_for_same_seed():
    first = generate_driver_data(seed=42)
    second = generate_driver_data(seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_trip_time_drives_rush_hour_flag():
    df = generate_driver_data(seed=42)
    hours = pd.to_datetime(df["trip_time"]).dt.hour
    expected_rush_hour = hours.isin(RUSH_HOURS).astype(int)

    pd.testing.assert_series_equal(df["rush_hour"], expected_rush_hour, check_names=False)


def test_distance_duration_base_and_final_price_formulas():
    df = generate_driver_data(seed=42)

    expected_distance = np.sqrt(
        (df["destination_x"] - df["origin_x"]) ** 2
        + (df["destination_y"] - df["origin_y"]) ** 2
    )
    np.testing.assert_allclose(df["distance_km"], expected_distance, atol=0.01)
    assert (df["distance_km"] >= 0.5).all()
    assert (df["duration_minute"] >= 1).all()

    expected_base = 10000 + 6000 * df["distance_km"] + 500 * df["duration_minute"]
    np.testing.assert_allclose(df["base_price"], expected_base, atol=1.0)
    np.testing.assert_allclose(df["final_price"], df["base_price"] + df["delta_price"], atol=1.0)
    assert (df["final_price"] > 0).all()


def test_each_driver_has_all_rain_and_rush_hour_groups():
    df = generate_driver_data(seed=42)
    counts = df.groupby(["driver_id", "rain", "rush_hour"]).size().unstack(["rain", "rush_hour"])

    assert counts.notna().all().all()
    assert (counts >= 1).all().all()


def test_baseline_ols_recovers_rain_and_rush_hour_effects():
    df = generate_driver_data(seed=42)
    model = LinearRegression()
    model.fit(df[["rain", "rush_hour"]], df["delta_price"])

    beta0 = model.intercept_
    beta1, beta2 = model.coef_

    assert 1000 <= beta0 <= 3000
    assert 6500 <= beta1 <= 9500
    assert 10500 <= beta2 <= 13500


def test_extended_models_improve_price_prediction():
    df = generate_driver_data(seed=42).sort_values("trip_time")
    split_index = int(len(df) * 0.8)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    rmse_m0 = _fit_rmse(train, test, ["rain", "rush_hour"])
    rmse_m1 = _fit_rmse(train, test, ["rain", "rush_hour", "origin_zone", "destination_zone"])
    rmse_m2 = _fit_rmse(
        train,
        test,
        ["rain", "rush_hour", "origin_zone", "destination_zone", "driver_id"],
    )

    assert rmse_m1 < rmse_m0
    assert rmse_m2 < rmse_m1


def test_logistic_regression_recovers_driver_acceptance_behavior():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    df = generate_driver_data(seed=42)
    model = LogisticRegression(solver='lbfgs')
    model.fit(df[["final_price", "rain", "rush_hour"]], df["driver_accept"])

    alpha_price, alpha_rain, alpha_rush = model.coef_[0]

    # Drivers prefer high prices
    assert alpha_price > 0
    
    # Drivers hate rain or rush hour
    assert alpha_rain < 0
    assert alpha_rush < 0

    pred_prob = model.predict_proba(df[["final_price", "rain", "rush_hour"]])[:, 1]
    auc = roc_auc_score(df["driver_accept"], pred_prob)
    assert auc > 0.55
