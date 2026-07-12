from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_PATH = Path("data/driver_synthetic_trips.csv")
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


def _balanced_driver_weather_time(
    rng: np.random.Generator, n_samples: int, n_drivers: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    drivers = [f"driver_{driver_id:02d}" for driver_id in range(1, n_drivers + 1)]
    required_rows: list[tuple[str, int, int]] = []

    for driver in drivers:
        for rain in [0, 1]:
            for rush_hour in [0, 1]:
                required_rows.append((driver, rain, rush_hour))

    if n_samples < len(required_rows):
        raise ValueError("n_samples must be at least 4 * n_drivers for balanced groups.")

    remaining = n_samples - len(required_rows)
    extra_driver_ids = rng.choice(drivers, size=remaining, replace=True)
    extra_rain = rng.binomial(1, 0.25, remaining)
    extra_hours = rng.integers(0, 24, remaining)
    extra_rush = np.isin(extra_hours, RUSH_HOURS).astype(int)

    driver_ids = np.array([row[0] for row in required_rows] + list(extra_driver_ids))
    rain = np.array([row[1] for row in required_rows] + list(extra_rain))
    rush_hour = np.array([row[2] for row in required_rows] + list(extra_rush))

    order = rng.permutation(n_samples)
    return driver_ids[order], rain[order], rush_hour[order]


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


def generate_driver_data(n_samples: int = 5000, seed: int = 42, n_drivers: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    driver_ids, rain, rush_hour = _balanced_driver_weather_time(rng, n_samples, n_drivers)
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

    driver_effects = _centered_effects(rng, sorted(set(driver_ids)), sigma=2000)
    origin_effects = _centered_effects(rng, [f"zone_{zone:02d}" for zone in range(1, 11)], sigma=1500)
    destination_effects = _centered_effects(rng, [f"zone_{zone:02d}" for zone in range(1, 11)], sigma=1000)

    beta0 = 2000
    beta1 = 8000
    beta2 = 12000
    epsilon = rng.normal(loc=0, scale=3000, size=n_samples)

    delta_price = (
        beta0
        + beta1 * rain
        + beta2 * rush_hour
        + np.array([driver_effects[driver] for driver in driver_ids])
        + np.array([origin_effects[zone] for zone in origin_zone])
        + np.array([destination_effects[zone] for zone in destination_zone])
        + epsilon
    )
    delta_price = np.round(delta_price, 0)
    final_price = base_price + delta_price

    df = pd.DataFrame(
        {
            "trip_id": np.arange(1, n_samples + 1),
            "driver_id": driver_ids,
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
        }
    )
    return df


def save_driver_data(
    output_path: Path = DEFAULT_OUTPUT_PATH, n_samples: int = 5000, seed: int = 42, n_drivers: int = 50
) -> pd.DataFrame:
    df = generate_driver_data(n_samples=n_samples, seed=seed, n_drivers=n_drivers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic ride-hailing price data.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-drivers", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = save_driver_data(
        output_path=args.output,
        n_samples=args.n_samples,
        seed=args.seed,
        n_drivers=args.n_drivers,
    )
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
