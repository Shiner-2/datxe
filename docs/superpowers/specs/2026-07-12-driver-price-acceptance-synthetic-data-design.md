# Driver Price Acceptance Synthetic Data Design

## Goal

Build a small, reproducible synthetic dataset and analysis workflow for a ride-hailing pricing problem. The first scope is driver behavior only: estimate how weather and rush hour affect trip price, then evaluate how price affects the driver's trip acceptance probability.

The workflow should produce both a CSV dataset and a notebook suitable for reporting.

## Scope

Included:

- Synthetic trip-level data for driver acceptance analysis.
- Two primary binary features: `rain` and `rush_hour`.
- A linear price model using independent effects only.
- A logistic driver acceptance model where higher price increases acceptance probability.
- Light noise and mild outliers so the data is realistic but still interpretable.

Excluded for now:

- Customer acceptance modeling.
- Interaction term `rain * rush_hour`.
- Individual driver heterogeneity.
- Real API, database, or production prediction service.

## Data Outputs

The project will generate:

- `data/driver_synthetic_trips.csv`: synthetic trip-level dataset.
- `notebooks/driver_price_acceptance.ipynb`: analysis notebook with modeling, charts, and conclusions.
- `src/generate_driver_data.py`: deterministic data generator using a fixed random seed.

Each row in the CSV represents one trip request and contains:

- `trip_id`: unique trip identifier.
- `rain`: `1` if the trip occurs during rain, else `0`.
- `rush_hour`: `1` if the trip occurs during rush hour, else `0`.
- `base_price`: base fare before weather and rush-hour surcharge.
- `delta_price`: extra price above the base fare.
- `price`: final trip price.
- `driver_accept_prob`: latent probability used to simulate driver acceptance.
- `driver_accept`: binary outcome, `1` if the driver accepts the trip, else `0`.

## Price Generation

The synthetic price model is:

```text
delta_price = beta0 + beta1 * rain + beta2 * rush_hour + noise
price = base_price + delta_price
```

Where:

- `beta0 > 0`: average surcharge when there is no rain and no rush hour.
- `beta1 > 0`: additional surcharge caused by rain.
- `beta2 > 0`: additional surcharge caused by rush hour.
- `noise`: random variation that represents unmodeled real-world factors.

There is no `rain * rush_hour` interaction term in this phase. Rain and rush hour contribute independently and additively to price.

## Driver Acceptance Generation

Driver acceptance will be generated with a logistic probability:

```text
driver_accept_prob = sigmoid(alpha0 + alpha_price * price + alpha_rain * rain + alpha_rush * rush_hour + noise)
driver_accept ~ Bernoulli(driver_accept_prob)
```

The key assumption is:

```text
alpha_price > 0
```

This means higher trip price increases the probability that a driver accepts the trip. Rain and rush hour may have their own smaller direct effects because they can make trips less comfortable or more difficult, but price is the main evaluation variable.

## Analysis Workflow

The notebook will:

1. Load the generated CSV.
2. Show basic dataset statistics and feature distributions.
3. Plot average `price` and `delta_price` by `rain` and `rush_hour`.
4. Fit Linear Regression:

```text
delta_price ~ rain + rush_hour
```

5. Interpret the estimated coefficients as the independent price effects of rain and rush hour.
6. Fit Logistic Regression:

```text
driver_accept ~ price + rain + rush_hour
```

7. Interpret the `price` coefficient as the effect of price on driver acceptance.
8. Plot driver acceptance rate by price bins.
9. Summarize the conclusion in Vietnamese for report use.

## Expected Results

The generated data and fitted models should support these conclusions:

- Rain increases the predicted trip price.
- Rush hour increases the predicted trip price.
- Because the driver acceptance model has a positive price coefficient, higher price increases the driver's probability of accepting a trip.
- The relationship should remain visible even with realistic random noise and mild outliers.

## Validation

Validation will check:

- The generator is deterministic with a fixed seed.
- The CSV contains the expected columns and no missing values.
- Estimated Linear Regression coefficients for `rain` and `rush_hour` are positive.
- Estimated Logistic Regression coefficient for `price` is positive.
- The notebook runs from top to bottom without manual edits.

