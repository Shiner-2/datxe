from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_PATH = Path("data/customer_synthetic_trips.csv")
RUSH_HOURS = np.array([7, 8, 9, 16, 17, 18])
NORMAL_HOURS = np.array([hour for hour in range(24) if hour not in set(RUSH_HOURS)])


def _centered_effects(rng: np.random.Generator, labels: list[str], sigma: float) -> dict[str, float]:
    values = rng.normal(loc=0, scale=sigma, size=len(labels))
    values = values - values.mean()
    return dict(zip(labels, values, strict=True))


def _sample_coordinates(
    rng: np.random.Generator, n_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    origin_x = rng.uniform(0, 10, n_samples)
    origin_y = rng.uniform(0, 10, n_samples)
    destination_x = rng.uniform(0, 10, n_samples)
    destination_y = rng.uniform(0, 10, n_samples)
    distance = np.sqrt((destination_x - origin_x) ** 2 + (destination_y - origin_y) ** 2)

    too_short = distance < 0.5
    while too_short.any():
        count = int(too_short.sum())
        destination_x[too_short] = rng.uniform(0, 10, count)
        destination_y[too_short] = rng.uniform(0, 10, count)
        distance = np.sqrt((destination_x - origin_x) ** 2 + (destination_y - origin_y) ** 2)
        too_short = distance < 0.5

    return origin_x, origin_y, destination_x, destination_y, distance


def _zone_from_x(values: np.ndarray) -> np.ndarray:
    zone_number = np.floor(values).astype(int) + 1
    zone_number = np.clip(zone_number, 1, 10)
    return np.array([f"zone_{zone:02d}" for zone in zone_number])


def _balanced_customer_weather_time(
    rng: np.random.Generator, n_samples: int, n_customers: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    customers = [f"customer_{c_id:03d}" for c_id in range(1, n_customers + 1)]
    required_rows: list[tuple[str, int, int]] = []

    for customer in customers:
        for rain in [0, 1]:
            for rush_hour in [0, 1]:
                required_rows.append((customer, rain, rush_hour))

    if n_samples < len(required_rows):
        raise ValueError("n_samples must be at least 4 * n_customers for balanced groups.")

    remaining = n_samples - len(required_rows)
    extra_customer_ids = rng.choice(customers, size=remaining, replace=True)
    extra_rain = rng.binomial(1, 0.25, remaining)
    extra_hours = rng.integers(0, 24, remaining)
    extra_rush = np.isin(extra_hours, RUSH_HOURS).astype(int)

    customer_ids = np.array([row[0] for row in required_rows] + list(extra_customer_ids))
    rain = np.array([row[1] for row in required_rows] + list(extra_rain))
    rush_hour = np.array([row[2] for row in required_rows] + list(extra_rush))

    order = rng.permutation(n_samples)
    return customer_ids[order], rain[order], rush_hour[order]


def _sample_trip_times(rng: np.random.Generator, rush_hour: np.ndarray) -> pd.Series:
    n_samples = len(rush_hour)
    days = rng.integers(0, 90, n_samples)
    minutes = rng.integers(0, 60, n_samples)

    hours = np.empty(n_samples, dtype=int)
    rush_mask = rush_hour == 1
    hours[rush_mask] = rng.choice(RUSH_HOURS, size=int(rush_mask.sum()))
    hours[~rush_mask] = rng.choice(NORMAL_HOURS, size=int((~rush_mask).sum()))

    trip_times = (
        pd.Timestamp("2024-01-01")
        + pd.to_timedelta(days, unit="D")
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
    )
    return pd.Series(trip_times).dt.strftime("%Y-%m-%d %H:%M:%S")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_customer_data(n_samples: int = 5000, seed: int = 42, n_customers: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_ids, rain, rush_hour = _balanced_customer_weather_time(rng, n_samples, n_customers)
    trip_time = _sample_trip_times(rng, rush_hour)

    origin_x, origin_y, destination_x, destination_y, distance_km = _sample_coordinates(rng, n_samples)

    origin_x = np.round(origin_x, 3)
    origin_y = np.round(origin_y, 3)
    destination_x = np.round(destination_x, 3)
    destination_y = np.round(destination_y, 3)
    distance_km = np.round(
        np.sqrt((destination_x - origin_x) ** 2 + (destination_y - origin_y) ** 2), 3
    )

    origin_zone = _zone_from_x(origin_x)
    destination_zone = _zone_from_x(destination_x)

    duration_noise = rng.normal(loc=0, scale=2, size=n_samples)
    duration_minute = 3 + 2.5 * distance_km + 8 * rush_hour + 3 * rain + duration_noise
    duration_minute = np.round(np.maximum(duration_minute, 1), 2)

    base_price = np.round(10000 + 6000 * distance_km + 500 * duration_minute, 0)

    origin_effects = _centered_effects(rng, [f"zone_{zone:02d}" for zone in range(1, 11)], sigma=1500)
    destination_effects = _centered_effects(rng, [f"zone_{zone:02d}" for zone in range(1, 11)], sigma=1000)

    # Price model
    beta0 = 2000
    beta1 = 8000
    beta2 = 12000
    epsilon_price = rng.normal(loc=0, scale=3000, size=n_samples)

    delta_price = (
        beta0
        + beta1 * rain
        + beta2 * rush_hour
        + np.array([origin_effects[zone] for zone in origin_zone])
        + np.array([destination_effects[zone] for zone in destination_zone])
        + epsilon_price
    )
    delta_price = np.round(delta_price, 0)
    final_price = base_price + delta_price

    # Customer acceptance model
    customer_effects = _centered_effects(rng, sorted(set(customer_ids)), sigma=0.5)

    alpha0 = 2.5
    alpha_price = -0.00003  # Negative effect: Higher price -> Lower acceptance
    alpha_rain = 1.2        # Positive effect: Will accept higher prices in rain
    alpha_rush = 0.8        # Positive effect: Will accept higher prices in rush hour
    epsilon_accept = rng.normal(loc=0, scale=0.5, size=n_samples)

    accept_logit = (
        alpha0
        + alpha_price * final_price
        + alpha_rain * rain
        + alpha_rush * rush_hour
        + np.array([customer_effects[cid] for cid in customer_ids])
        + epsilon_accept
    )
    
    customer_accept_prob = sigmoid(accept_logit)
    customer_accept = rng.binomial(1, customer_accept_prob)

    df = pd.DataFrame(
        {
            "trip_id": np.arange(1, n_samples + 1),
            "customer_id": customer_ids,
            "trip_time": trip_time,
            "origin_zone": origin_zone,
            "destination_zone": destination_zone,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "destination_x": destination_x,
            "destination_y": destination_y,
            "distance_km": distance_km,
            "duration_minute": duration_minute,
            "rain": rain,
            "rush_hour": rush_hour,
            "base_price": base_price,
            "delta_price": delta_price,
            "final_price": final_price,
            "customer_accept_prob": np.round(customer_accept_prob, 4),
            "customer_accept": customer_accept,
        }
    )
    return df


def save_customer_data(
    output_path: Path = DEFAULT_OUTPUT_PATH, n_samples: int = 5000, seed: int = 42, n_customers: int = 200
) -> pd.DataFrame:
    df = generate_customer_data(n_samples=n_samples, seed=seed, n_customers=n_customers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic ride-hailing data for customers.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-customers", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = save_customer_data(
        output_path=args.output,
        n_samples=args.n_samples,
        seed=args.seed,
        n_customers=args.n_customers,
    )
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
