from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_PATH = Path("data/driver_synthetic_trips.csv")


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def generate_driver_data(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    rain = rng.binomial(1, 0.28, n_samples)
    rush_hour = rng.binomial(1, 0.36, n_samples)

    base_price = rng.normal(loc=42000, scale=8500, size=n_samples)
    base_price = np.clip(base_price, 18000, 90000)

    beta0 = 2500
    beta1 = 3500
    beta2 = 4800
    price_noise = rng.normal(loc=0, scale=1700, size=n_samples)

    outlier_mask = rng.random(n_samples) < 0.025
    price_noise[outlier_mask] += rng.normal(loc=4500, scale=1800, size=outlier_mask.sum())

    delta_price = beta0 + beta1 * rain + beta2 * rush_hour + price_noise
    delta_price = np.clip(delta_price, 700, None)

    base_price = np.round(base_price, 0)
    delta_price = np.round(delta_price, 0)
    price = base_price + delta_price

    centered_price = (price - price.mean()) / 10000

    alpha0 = 0.15
    alpha_price = 0.75
    alpha_rain = -0.18
    alpha_rush = -0.10
    acceptance_noise = rng.normal(loc=0, scale=0.30, size=n_samples)

    logit = (
        alpha0
        + alpha_price * centered_price
        + alpha_rain * rain
        + alpha_rush * rush_hour
        + acceptance_noise
    )
    driver_accept_prob = sigmoid(logit)
    driver_accept = rng.binomial(1, driver_accept_prob)

    df = pd.DataFrame(
        {
            "trip_id": np.arange(1, n_samples + 1),
            "rain": rain,
            "rush_hour": rush_hour,
            "base_price": base_price,
            "delta_price": delta_price,
            "price": price,
            "driver_accept_prob": np.round(driver_accept_prob, 4),
            "driver_accept": driver_accept,
        }
    )
    return df


def save_driver_data(
    output_path: Path = DEFAULT_OUTPUT_PATH, n_samples: int = 3000, seed: int = 42
) -> pd.DataFrame:
    df = generate_driver_data(n_samples=n_samples, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic driver acceptance trip data.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = save_driver_data(output_path=args.output, n_samples=args.n_samples, seed=args.seed)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
