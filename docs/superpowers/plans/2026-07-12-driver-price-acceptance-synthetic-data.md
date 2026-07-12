# Driver Price Acceptance Synthetic Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible synthetic trip-level data plus a notebook that analyzes how rain and rush hour affect ride-hailing price, then how price affects driver acceptance.

**Architecture:** Keep the project small and script-first. `src/generate_driver_data.py` owns deterministic data generation and CSV export; `tests/test_generate_driver_data.py` verifies the statistical assumptions; `notebooks/driver_price_acceptance.ipynb` presents the report workflow using the generated CSV.

**Tech Stack:** Python 3, NumPy, pandas, scikit-learn, matplotlib, seaborn, pytest, Jupyter/nbconvert.

---

## File Structure

- Create `requirements.txt`: runtime and test dependencies.
- Create `src/__init__.py`: make `src` importable for tests.
- Create `src/generate_driver_data.py`: deterministic synthetic data generator and CLI.
- Create `tests/test_generate_driver_data.py`: unit/statistical checks for generated data.
- Create `data/driver_synthetic_trips.csv`: generated dataset from the CLI.
- Create `notebooks/driver_price_acceptance.ipynb`: runnable report notebook.

## Task 1: Project Dependencies

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Create dependency file**

```text
numpy>=1.26
pandas>=2.2
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
jupyter>=1.0
nbconvert>=7.16
nbformat>=5.10
pytest>=8.0
```

- [ ] **Step 2: Install dependencies**

Run:

```bash
python -m pip install -r requirements.txt
```

Expected: command exits with status `0`.

- [ ] **Step 3: Commit dependencies**

```bash
git add requirements.txt
git commit -m "chore: add analysis dependencies"
```

## Task 2: Generator Tests

**Files:**
- Create: `src/__init__.py`
- Create: `tests/test_generate_driver_data.py`
- Later implementation target: `src/generate_driver_data.py`

- [ ] **Step 1: Create package marker**

Create `src/__init__.py` as an empty file.

- [ ] **Step 2: Write failing tests**

Create `tests/test_generate_driver_data.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
pytest tests/test_generate_driver_data.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.generate_driver_data'`.

- [ ] **Step 4: Commit failing tests**

```bash
git add src/__init__.py tests/test_generate_driver_data.py
git commit -m "test: define driver data generator expectations"
```

## Task 3: Synthetic Data Generator

**Files:**
- Create: `src/generate_driver_data.py`
- Modify: `tests/test_generate_driver_data.py` only if an assertion is mathematically wrong after inspecting output.

- [ ] **Step 1: Implement generator**

Create `src/generate_driver_data.py`:

```python
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

    base_price = np.round(base_price, 0)
    delta_price = np.round(delta_price, 0)
    price = base_price + delta_price

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


def save_driver_data(output_path: Path = DEFAULT_OUTPUT_PATH, n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
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
```

- [ ] **Step 2: Run generator tests**

Run:

```bash
pytest tests/test_generate_driver_data.py -v
```

Expected: all tests PASS.

- [ ] **Step 3: Generate CSV**

Run:

```bash
python src/generate_driver_data.py --output data/driver_synthetic_trips.csv --n-samples 3000 --seed 42
```

Expected output:

```text
Saved 3000 rows to data/driver_synthetic_trips.csv
```

- [ ] **Step 4: Inspect generated CSV header**

Run:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('data/driver_synthetic_trips.csv')
print(df.head())
print(df.shape)
print(df.isna().sum().sum())
PY
```

Expected: first rows print, shape is `(3000, 8)`, missing count is `0`.

- [ ] **Step 5: Commit generator and data**

```bash
git add src/generate_driver_data.py data/driver_synthetic_trips.csv
git commit -m "feat: generate driver synthetic trip data"
```

## Task 4: Analysis Notebook

**Files:**
- Create: `notebooks/driver_price_acceptance.ipynb`

- [ ] **Step 1: Create notebook**

Run this structured notebook writer:

```bash
python - <<'PY'
from pathlib import Path

import nbformat as nbf

notebook_path = Path("notebooks/driver_price_acceptance.ipynb")
notebook_path.parent.mkdir(parents=True, exist_ok=True)

markdown_intro = """# Driver Price Acceptance Analysis

Notebook nay phan tich synthetic data cho bai toan gia xe cong nghe, tap trung vao tai xe. Hai feature chinh la `rain` va `rush_hour`. Muc tieu la du doan phu phi, sau do danh gia tac dong cua gia den kha nang tai xe nhan chuyen.
"""

cells = [
    nbf.v4.new_markdown_cell(markdown_intro),
    nbf.v4.new_code_cell(
        '''from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

sns.set_theme(style="whitegrid")
DATA_PATH = Path("../data/driver_synthetic_trips.csv")
df = pd.read_csv(DATA_PATH)
df.head()'''
    ),
    nbf.v4.new_code_cell('''df.describe(include="all")'''),
    nbf.v4.new_code_cell(
        '''df[["rain", "rush_hour", "driver_accept"]].mean().rename(
    {
        "rain": "Ti le chuyen khi troi mua",
        "rush_hour": "Ti le chuyen trong gio cao diem",
        "driver_accept": "Ti le tai xe nhan chuyen",
    }
)'''
    ),
    nbf.v4.new_code_cell(
        '''price_summary = (
    df.groupby(["rain", "rush_hour"], as_index=False)
    .agg(
        avg_base_price=("base_price", "mean"),
        avg_delta_price=("delta_price", "mean"),
        avg_price=("price", "mean"),
        acceptance_rate=("driver_accept", "mean"),
        n_trips=("trip_id", "count"),
    )
)
price_summary'''
    ),
    nbf.v4.new_code_cell(
        '''fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(data=df, x="rain", y="delta_price", ax=axes[0], errorbar="sd")
axes[0].set_title("Phu phi trung binh theo trang thai mua")
axes[0].set_xlabel("Rain")
axes[0].set_ylabel("Delta price")

sns.barplot(data=df, x="rush_hour", y="delta_price", ax=axes[1], errorbar="sd")
axes[1].set_title("Phu phi trung binh theo gio cao diem")
axes[1].set_xlabel("Rush hour")
axes[1].set_ylabel("Delta price")
plt.tight_layout()'''
    ),
    nbf.v4.new_code_cell(
        '''linear_model = LinearRegression()
linear_model.fit(df[["rain", "rush_hour"]], df["delta_price"])

linear_coefficients = pd.DataFrame(
    {
        "feature": ["intercept", "rain", "rush_hour"],
        "coefficient": [linear_model.intercept_, *linear_model.coef_],
    }
)
linear_coefficients'''
    ),
    nbf.v4.new_code_cell(
        '''logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(df[["price", "rain", "rush_hour"]], df["driver_accept"])

predicted_accept = logistic_model.predict(df[["price", "rain", "rush_hour"]])
predicted_prob = logistic_model.predict_proba(df[["price", "rain", "rush_hour"]])[:, 1]

logistic_coefficients = pd.DataFrame(
    {
        "feature": ["intercept", "price", "rain", "rush_hour"],
        "coefficient": [logistic_model.intercept_[0], *logistic_model.coef_[0]],
    }
)

metrics = {
    "accuracy": accuracy_score(df["driver_accept"], predicted_accept),
    "roc_auc": roc_auc_score(df["driver_accept"], predicted_prob),
}

logistic_coefficients, metrics'''
    ),
    nbf.v4.new_code_cell(
        '''df_bins = df.assign(price_bin=pd.qcut(df["price"], q=6))
acceptance_by_price = (
    df_bins.groupby("price_bin", observed=True)
    .agg(
        avg_price=("price", "mean"),
        acceptance_rate=("driver_accept", "mean"),
        n_trips=("trip_id", "count"),
    )
    .reset_index()
)
acceptance_by_price'''
    ),
    nbf.v4.new_code_cell(
        '''plt.figure(figsize=(8, 4))
sns.lineplot(data=acceptance_by_price, x="avg_price", y="acceptance_rate", marker="o")
plt.title("Ti le tai xe nhan chuyen theo muc gia")
plt.xlabel("Gia trung binh")
plt.ylabel("Ti le nhan chuyen")
plt.ylim(0, 1)
plt.tight_layout()'''
    ),
    nbf.v4.new_code_cell(
        '''rain_effect = linear_coefficients.loc[linear_coefficients["feature"] == "rain", "coefficient"].iloc[0]
rush_effect = linear_coefficients.loc[linear_coefficients["feature"] == "rush_hour", "coefficient"].iloc[0]
price_effect = logistic_coefficients.loc[logistic_coefficients["feature"] == "price", "coefficient"].iloc[0]

conclusion = f"""
Ket luan:
- Khi troi mua, phu phi du doan tang trung binh khoang {rain_effect:,.0f} VND.
- Trong gio cao diem, phu phi du doan tang trung binh khoang {rush_effect:,.0f} VND.
- He so price trong mo hinh Logistic Regression la {price_effect:.6f} va lon hon 0.
- Dieu nay phu hop voi gia dinh: gia chuyen xe cao hon lam xac suat tai xe nhan chuyen tang len.
"""
print(conclusion)'''
    ),
]

notebook = nbf.v4.new_notebook(cells=cells)
nbf.write(notebook, notebook_path)
print(f"Wrote {notebook_path}")
PY
```

Expected output:

```text
Wrote notebooks/driver_price_acceptance.ipynb
```

The notebook contains cells equivalent to this Python content in order:

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

sns.set_theme(style="whitegrid")
DATA_PATH = Path("../data/driver_synthetic_trips.csv")
df = pd.read_csv(DATA_PATH)
df.head()
```

```python
df.describe(include="all")
```

```python
df[["rain", "rush_hour", "driver_accept"]].mean().rename(
    {
        "rain": "Ti le chuyen khi troi mua",
        "rush_hour": "Ti le chuyen trong gio cao diem",
        "driver_accept": "Ti le tai xe nhan chuyen",
    }
)
```

```python
price_summary = (
    df.groupby(["rain", "rush_hour"], as_index=False)
    .agg(
        avg_base_price=("base_price", "mean"),
        avg_delta_price=("delta_price", "mean"),
        avg_price=("price", "mean"),
        acceptance_rate=("driver_accept", "mean"),
        n_trips=("trip_id", "count"),
    )
)
price_summary
```

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(data=df, x="rain", y="delta_price", ax=axes[0], errorbar="sd")
axes[0].set_title("Phu phi trung binh theo trang thai mua")
axes[0].set_xlabel("Rain")
axes[0].set_ylabel("Delta price")

sns.barplot(data=df, x="rush_hour", y="delta_price", ax=axes[1], errorbar="sd")
axes[1].set_title("Phu phi trung binh theo gio cao diem")
axes[1].set_xlabel("Rush hour")
axes[1].set_ylabel("Delta price")
plt.tight_layout()
```

```python
linear_model = LinearRegression()
linear_model.fit(df[["rain", "rush_hour"]], df["delta_price"])

linear_coefficients = pd.DataFrame(
    {
        "feature": ["intercept", "rain", "rush_hour"],
        "coefficient": [linear_model.intercept_, *linear_model.coef_],
    }
)
linear_coefficients
```

```python
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(df[["price", "rain", "rush_hour"]], df["driver_accept"])

predicted_accept = logistic_model.predict(df[["price", "rain", "rush_hour"]])
predicted_prob = logistic_model.predict_proba(df[["price", "rain", "rush_hour"]])[:, 1]

logistic_coefficients = pd.DataFrame(
    {
        "feature": ["intercept", "price", "rain", "rush_hour"],
        "coefficient": [logistic_model.intercept_[0], *logistic_model.coef_[0]],
    }
)

metrics = {
    "accuracy": accuracy_score(df["driver_accept"], predicted_accept),
    "roc_auc": roc_auc_score(df["driver_accept"], predicted_prob),
}

logistic_coefficients, metrics
```

```python
df_bins = df.assign(price_bin=pd.qcut(df["price"], q=6))
acceptance_by_price = (
    df_bins.groupby("price_bin", observed=True)
    .agg(
        avg_price=("price", "mean"),
        acceptance_rate=("driver_accept", "mean"),
        n_trips=("trip_id", "count"),
    )
    .reset_index()
)
acceptance_by_price
```

```python
plt.figure(figsize=(8, 4))
sns.lineplot(data=acceptance_by_price, x="avg_price", y="acceptance_rate", marker="o")
plt.title("Ti le tai xe nhan chuyen theo muc gia")
plt.xlabel("Gia trung binh")
plt.ylabel("Ti le nhan chuyen")
plt.ylim(0, 1)
plt.tight_layout()
```

```python
rain_effect = linear_coefficients.loc[linear_coefficients["feature"] == "rain", "coefficient"].iloc[0]
rush_effect = linear_coefficients.loc[linear_coefficients["feature"] == "rush_hour", "coefficient"].iloc[0]
price_effect = logistic_coefficients.loc[logistic_coefficients["feature"] == "price", "coefficient"].iloc[0]

conclusion = f"""
Ket luan:
- Khi troi mua, phu phi du doan tang trung binh khoang {rain_effect:,.0f} VND.
- Trong gio cao diem, phu phi du doan tang trung binh khoang {rush_effect:,.0f} VND.
- He so price trong mo hinh Logistic Regression la {price_effect:.6f} va lon hon 0.
- Dieu nay phu hop voi gia dinh: gia chuyen xe cao hon lam xac suat tai xe nhan chuyen tang len.
"""
print(conclusion)
```

- [ ] **Step 2: Run notebook from top to bottom**

Run:

```bash
jupyter nbconvert --to notebook --execute notebooks/driver_price_acceptance.ipynb --output driver_price_acceptance.executed.ipynb --output-dir notebooks
```

Expected: command exits with status `0` and creates `notebooks/driver_price_acceptance.executed.ipynb`.

- [ ] **Step 3: Remove executed copy from version control scope**

Run:

```bash
rm notebooks/driver_price_acceptance.executed.ipynb
```

Expected: file is removed; original notebook remains.

- [ ] **Step 4: Commit notebook**

```bash
git add notebooks/driver_price_acceptance.ipynb
git commit -m "docs: add driver price acceptance notebook"
```

## Task 5: Final Verification

**Files:**
- Read: `requirements.txt`
- Read: `src/generate_driver_data.py`
- Read: `tests/test_generate_driver_data.py`
- Read: `data/driver_synthetic_trips.csv`
- Read: `notebooks/driver_price_acceptance.ipynb`

- [ ] **Step 1: Run full test suite**

Run:

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 2: Regenerate data and verify deterministic output**

Run:

```bash
python src/generate_driver_data.py --output data/driver_synthetic_trips.csv --n-samples 3000 --seed 42
git diff -- data/driver_synthetic_trips.csv
```

Expected: generator prints `Saved 3000 rows to data/driver_synthetic_trips.csv`; `git diff` prints no changes.

- [ ] **Step 3: Execute notebook**

Run:

```bash
jupyter nbconvert --to notebook --execute notebooks/driver_price_acceptance.ipynb --output driver_price_acceptance.executed.ipynb --output-dir notebooks
rm notebooks/driver_price_acceptance.executed.ipynb
```

Expected: nbconvert exits with status `0`; executed copy is removed.

- [ ] **Step 4: Check final git status**

Run:

```bash
git status --short
```

Expected: no uncommitted tracked changes. If generated or planned files are intentionally uncommitted, commit them with a focused message before completion.
