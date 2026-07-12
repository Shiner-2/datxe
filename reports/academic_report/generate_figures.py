from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    driver = pd.read_csv(DATA_DIR / "driver_synthetic_trips.csv")
    customer = pd.read_csv(DATA_DIR / "customer_synthetic_trips.csv")
    return driver, customer


def _context_labels(df: pd.DataFrame) -> pd.Series:
    labels = np.select(
        [
            (df["rain"] == 0) & (df["rush_hour"] == 0),
            (df["rain"] == 1) & (df["rush_hour"] == 0),
            (df["rain"] == 0) & (df["rush_hour"] == 1),
            (df["rain"] == 1) & (df["rush_hour"] == 1),
        ],
        ["Không mưa\nBình thường", "Mưa\nBình thường", "Không mưa\nCao điểm", "Mưa\nCao điểm"],
        default="Không xác định",
    )
    return pd.Series(labels, index=df.index, name="context")


def _save(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_final_price_distribution(driver: pd.DataFrame, customer: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(driver["final_price"], bins=35, stat="density", alpha=0.45, label="Tài xế", ax=ax)
    sns.histplot(customer["final_price"], bins=35, stat="density", alpha=0.45, label="Khách hàng", ax=ax)
    ax.set_title("Phân phối giá cuối cùng trong hai bộ dữ liệu")
    ax.set_xlabel("Giá cuối cùng (VND)")
    ax.set_ylabel("Mật độ")
    ax.legend()
    ax.grid(alpha=0.25)
    return _save(fig, output_dir, "final_price_distribution.png")


def _plot_acceptance_by_context(driver: pd.DataFrame, customer: pd.DataFrame, output_dir: Path) -> Path:
    driver_plot = driver.assign(context=_context_labels(driver), group="Tài xế").groupby(
        ["context", "group"], as_index=False
    )["driver_accept"].mean()
    driver_plot = driver_plot.rename(columns={"driver_accept": "accept_rate"})

    customer_plot = customer.assign(context=_context_labels(customer), group="Khách hàng").groupby(
        ["context", "group"], as_index=False
    )["customer_accept"].mean()
    customer_plot = customer_plot.rename(columns={"customer_accept": "accept_rate"})

    plot_df = pd.concat([driver_plot, customer_plot], ignore_index=True)
    order = ["Không mưa\nBình thường", "Mưa\nBình thường", "Không mưa\nCao điểm", "Mưa\nCao điểm"]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="context", y="accept_rate", hue="group", order=order, ax=ax)
    ax.set_title("Tỷ lệ chấp nhận theo ngữ cảnh")
    ax.set_xlabel("Ngữ cảnh")
    ax.set_ylabel("Tỷ lệ chấp nhận")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(title="Nhóm")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, output_dir, "acceptance_by_context.png")


def _plot_delta_price_by_context(driver: pd.DataFrame, customer: pd.DataFrame, output_dir: Path) -> Path:
    driver_plot = driver.assign(context=_context_labels(driver), group="Tài xế").groupby(
        ["context", "group"], as_index=False
    )["delta_price"].mean()
    customer_plot = customer.assign(context=_context_labels(customer), group="Khách hàng").groupby(
        ["context", "group"], as_index=False
    )["delta_price"].mean()
    plot_df = pd.concat([driver_plot, customer_plot], ignore_index=True)
    order = ["Không mưa\nBình thường", "Mưa\nBình thường", "Không mưa\nCao điểm", "Mưa\nCao điểm"]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="context", y="delta_price", hue="group", order=order, ax=ax)
    ax.set_title("Phụ phí trung bình theo ngữ cảnh")
    ax.set_xlabel("Ngữ cảnh")
    ax.set_ylabel("Delta price trung bình (VND)")
    ax.legend(title="Nhóm")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, output_dir, "delta_price_by_context.png")


def _acceptance_by_price(df: pd.DataFrame, target: str, group: str) -> pd.DataFrame:
    bins = np.arange(10_000, 135_001, 10_000)
    binned = df.assign(price_bin=pd.cut(df["final_price"], bins=bins, include_lowest=True))
    grouped = binned.groupby("price_bin", observed=True)[target].mean().reset_index(name="accept_rate")
    grouped["price_mid"] = grouped["price_bin"].apply(lambda interval: interval.mid).astype(float)
    grouped["group"] = group
    return grouped


def _plot_acceptance_by_price_bin(driver: pd.DataFrame, customer: pd.DataFrame, output_dir: Path) -> Path:
    plot_df = pd.concat(
        [
            _acceptance_by_price(driver, "driver_accept", "Tài xế"),
            _acceptance_by_price(customer, "customer_accept", "Khách hàng"),
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=plot_df, x="price_mid", y="accept_rate", hue="group", marker="o", ax=ax)
    ax.set_title("Tỷ lệ chấp nhận theo khoảng giá")
    ax.set_xlabel("Trung điểm khoảng giá (VND)")
    ax.set_ylabel("Tỷ lệ chấp nhận")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(alpha=0.25)
    return _save(fig, output_dir, "acceptance_by_price_bin.png")


def _fit_logistic(df: pd.DataFrame, target: str) -> LogisticRegression:
    x = df[["final_price", "rain", "rush_hour"]]
    y = df[target]
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(x_train, y_train)
    return model


def _predict(model: LogisticRegression, final_price: int, rain: int, rush_hour: int) -> float:
    x = pd.DataFrame({"final_price": [final_price], "rain": [rain], "rush_hour": [rush_hour]})
    return float(model.predict_proba(x)[0, 1])


def _plot_profit_heatmap(driver: pd.DataFrame, customer: pd.DataFrame, output_dir: Path) -> Path:
    customer_model = _fit_logistic(customer, "customer_accept")
    driver_model = _fit_logistic(driver, "driver_accept")

    p_cust_range = np.arange(30_000, 100_001, 2_000)
    p_driver_range = np.arange(20_000, 90_001, 2_000)
    rain = 1
    rush_hour = 1
    profit_matrix = np.full((len(p_cust_range), len(p_driver_range)), np.nan)

    best_profit = -np.inf
    best_point = (0, 0)
    for i, pc in enumerate(p_cust_range):
        for j, pd_price in enumerate(p_driver_range):
            if pc <= pd_price:
                continue
            prob_c = _predict(customer_model, int(pc), rain, rush_hour)
            prob_d = _predict(driver_model, int(pd_price), rain, rush_hour)
            profit = (pc - pd_price) * prob_c * prob_d
            profit_matrix[i, j] = profit
            if profit > best_profit:
                best_profit = profit
                best_point = (j, i)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.heatmap(
        profit_matrix,
        xticklabels=p_driver_range // 1000,
        yticklabels=p_cust_range // 1000,
        cmap="YlOrRd",
        cbar_kws={"label": "Lợi nhuận kỳ vọng (VND)"},
        ax=ax,
    )
    ax.scatter(best_point[0] + 0.5, best_point[1] + 0.5, marker="*", s=220, c="royalblue", edgecolors="white")
    ax.set_title("Heatmap lợi nhuận kỳ vọng: Mưa + Cao điểm")
    ax.set_xlabel("Giá trả tài xế (nghìn VND)")
    ax.set_ylabel("Giá thu khách hàng (nghìn VND)")
    ax.invert_yaxis()
    return _save(fig, output_dir, "profit_heatmap.png")


def _plot_trip_spatial_distribution(driver: pd.DataFrame, customer: pd.DataFrame, output_dir: Path) -> Path:
    sample_driver = driver.sample(n=900, random_state=42).assign(group="Tài xế")
    sample_customer = customer.sample(n=900, random_state=42).assign(group="Khách hàng")
    plot_df = pd.concat([sample_driver, sample_customer], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
    for ax, point_prefix, title in [
        (axes[0], "origin", "Điểm xuất phát"),
        (axes[1], "destination", "Điểm kết thúc"),
    ]:
        sns.scatterplot(
            data=plot_df,
            x=f"{point_prefix}_x",
            y=f"{point_prefix}_y",
            hue="group",
            alpha=0.45,
            s=18,
            linewidth=0,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Tọa độ x")
        ax.set_ylabel("Tọa độ y")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.grid(alpha=0.25)
    axes[1].legend_.remove()
    axes[0].legend(title="Nhóm")
    fig.suptitle("Phân bố không gian của các chuyến đi", y=1.02)
    return _save(fig, output_dir, "trip_spatial_distribution.png")


def _plot_distance_duration_relationship(driver: pd.DataFrame, customer: pd.DataFrame, output_dir: Path) -> Path:
    sample = pd.concat(
        [
            driver.sample(n=900, random_state=7).assign(group="Tài xế"),
            customer.sample(n=900, random_state=8).assign(group="Khách hàng"),
        ],
        ignore_index=True,
    )
    sample["context"] = _context_labels(sample)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.scatterplot(
        data=sample,
        x="distance_km",
        y="duration_minute",
        hue="context",
        alpha=0.55,
        s=18,
        linewidth=0,
        ax=ax,
    )
    sns.regplot(data=sample, x="distance_km", y="duration_minute", scatter=False, color="black", ax=ax)
    ax.set_title("Quan hệ giữa quãng đường và thời lượng chuyến đi")
    ax.set_xlabel("Quãng đường (km)")
    ax.set_ylabel("Thời lượng (phút)")
    ax.legend(title="Ngữ cảnh", fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, output_dir, "distance_duration_relationship.png")


def _fit_driver_rmse(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[float, float]:
    train_x = pd.get_dummies(train[features], drop_first=True, dtype=float)
    test_x = pd.get_dummies(test[features], drop_first=True, dtype=float).reindex(columns=train_x.columns, fill_value=0)
    model = LinearRegression()
    model.fit(train_x, train["delta_price"])
    predicted = model.predict(test_x)
    return mean_squared_error(test["delta_price"], predicted) ** 0.5, r2_score(test["delta_price"], predicted)


def _plot_driver_model_comparison(driver: pd.DataFrame, output_dir: Path) -> Path:
    sorted_df = driver.sort_values("trip_time")
    split_index = int(len(sorted_df) * 0.8)
    train = sorted_df.iloc[:split_index]
    test = sorted_df.iloc[split_index:]
    specs = [
        ("M0\nMưa + cao điểm", ["rain", "rush_hour"]),
        ("M1\nThêm khu vực", ["rain", "rush_hour", "origin_zone", "destination_zone"]),
        ("M2\nThêm tài xế", ["rain", "rush_hour", "origin_zone", "destination_zone", "driver_id"]),
    ]
    rows = []
    for name, features in specs:
        rmse, r2 = _fit_driver_rmse(train, test, features)
        rows.append({"model": name, "RMSE": rmse, "R2": r2})
    plot_df = pd.DataFrame(rows)

    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    sns.barplot(data=plot_df, x="model", y="RMSE", color="#5B8FF9", ax=ax1)
    ax1.set_title("So sánh mô hình dự đoán phụ phí phía tài xế")
    ax1.set_xlabel("Mô hình")
    ax1.set_ylabel("RMSE trên test set (VND)")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(plot_df["model"], plot_df["R2"], color="#D62728", marker="o", linewidth=2)
    ax2.set_ylabel("R² trên test set")
    ax2.set_ylim(0.6, 0.9)
    return _save(fig, output_dir, "driver_model_comparison.png")


def _plot_predicted_curve(model: LogisticRegression, prices: np.ndarray, contexts: list[dict[str, int | str]]) -> pd.DataFrame:
    rows = []
    for ctx in contexts:
        for price in prices:
            rows.append(
                {
                    "price": price,
                    "probability": _predict(model, int(price), int(ctx["rain"]), int(ctx["rush_hour"])),
                    "context": str(ctx["name"]),
                }
            )
    return pd.DataFrame(rows)


def _plot_customer_demand_curves(customer: pd.DataFrame, output_dir: Path) -> Path:
    model = _fit_logistic(customer, "customer_accept")
    prices = np.arange(25_000, 105_001, 2_500)
    contexts = [
        {"name": "Bình thường", "rain": 0, "rush_hour": 0},
        {"name": "Mưa", "rain": 1, "rush_hour": 0},
        {"name": "Cao điểm", "rain": 0, "rush_hour": 1},
        {"name": "Mưa + cao điểm", "rain": 1, "rush_hour": 1},
    ]
    plot_df = _plot_predicted_curve(model, prices, contexts)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=plot_df, x="price", y="probability", hue="context", linewidth=2, ax=ax)
    ax.set_title("Đường cầu dự đoán của khách hàng")
    ax.set_xlabel("Giá thu khách hàng (VND)")
    ax.set_ylabel("Xác suất đặt xe dự đoán")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(alpha=0.25)
    ax.legend(title="Ngữ cảnh")
    return _save(fig, output_dir, "customer_demand_curves.png")


def _plot_driver_supply_curves(driver: pd.DataFrame, output_dir: Path) -> Path:
    model = _fit_logistic(driver, "driver_accept")
    prices = np.arange(25_000, 105_001, 2_500)
    contexts = [
        {"name": "Bình thường", "rain": 0, "rush_hour": 0},
        {"name": "Mưa", "rain": 1, "rush_hour": 0},
        {"name": "Cao điểm", "rain": 0, "rush_hour": 1},
        {"name": "Mưa + cao điểm", "rain": 1, "rush_hour": 1},
    ]
    plot_df = _plot_predicted_curve(model, prices, contexts)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=plot_df, x="price", y="probability", hue="context", linewidth=2, ax=ax)
    ax.set_title("Đường cung dự đoán của tài xế")
    ax.set_xlabel("Giá trả tài xế (VND)")
    ax.set_ylabel("Xác suất nhận chuyến dự đoán")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(alpha=0.25)
    ax.legend(title="Ngữ cảnh")
    return _save(fig, output_dir, "driver_supply_curves.png")


def _optimal_pricing_summary(driver: pd.DataFrame, customer: pd.DataFrame) -> pd.DataFrame:
    customer_model = _fit_logistic(customer, "customer_accept")
    driver_model = _fit_logistic(driver, "driver_accept")
    contexts = [
        {"context": "Bình thường", "rain": 0, "rush_hour": 0},
        {"context": "Mưa", "rain": 1, "rush_hour": 0},
        {"context": "Cao điểm", "rain": 0, "rush_hour": 1},
        {"context": "Mưa + cao điểm", "rain": 1, "rush_hour": 1},
    ]
    p_cust_range = np.arange(30_000, 100_001, 2_000)
    p_driver_range = np.arange(20_000, 90_001, 2_000)
    rows = []
    for ctx in contexts:
        best = {"expected_profit": -np.inf}
        for pc in p_cust_range:
            for pd_price in p_driver_range:
                if pc <= pd_price:
                    continue
                prob_c = _predict(customer_model, int(pc), int(ctx["rain"]), int(ctx["rush_hour"]))
                prob_d = _predict(driver_model, int(pd_price), int(ctx["rain"]), int(ctx["rush_hour"]))
                match_rate = prob_c * prob_d
                expected_profit = (pc - pd_price) * match_rate
                if expected_profit > best["expected_profit"]:
                    best = {
                        "context": ctx["context"],
                        "p_customer": pc,
                        "p_driver": pd_price,
                        "margin": pc - pd_price,
                        "match_rate": match_rate,
                        "expected_profit": expected_profit,
                    }
        rows.append(best)
    return pd.DataFrame(rows)


def _plot_optimal_prices_by_context(driver: pd.DataFrame, customer: pd.DataFrame, output_dir: Path) -> Path:
    summary = _optimal_pricing_summary(driver, customer)
    price_df = summary.melt(
        id_vars="context",
        value_vars=["p_customer", "p_driver"],
        var_name="price_type",
        value_name="price",
    )
    price_df["price_type"] = price_df["price_type"].map({"p_customer": "Thu khách", "p_driver": "Trả tài xế"})

    fig, ax1 = plt.subplots(figsize=(9.5, 5.2))
    sns.barplot(data=price_df, x="context", y="price", hue="price_type", ax=ax1)
    ax1.set_title("Cặp giá tối ưu theo ngữ cảnh")
    ax1.set_xlabel("Ngữ cảnh")
    ax1.set_ylabel("Mức giá tối ưu (VND)")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(title="Loại giá", loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(summary["context"], summary["expected_profit"], color="#D62728", marker="o", linewidth=2)
    ax2.set_ylabel("Lợi nhuận kỳ vọng tối đa (VND)")
    return _save(fig, output_dir, "optimal_prices_by_context.png")


def generate_all_figures(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    output_path = Path(output_dir)
    driver, customer = _load_data()
    return [
        _plot_trip_spatial_distribution(driver, customer, output_path),
        _plot_distance_duration_relationship(driver, customer, output_path),
        _plot_final_price_distribution(driver, customer, output_path),
        _plot_driver_model_comparison(driver, output_path),
        _plot_acceptance_by_context(driver, customer, output_path),
        _plot_customer_demand_curves(customer, output_path),
        _plot_driver_supply_curves(driver, output_path),
        _plot_delta_price_by_context(driver, customer, output_path),
        _plot_optimal_prices_by_context(driver, customer, output_path),
        _plot_acceptance_by_price_bin(driver, customer, output_path),
        _plot_profit_heatmap(driver, customer, output_path),
    ]


def main() -> None:
    for path in generate_all_figures():
        print(path)


if __name__ == "__main__":
    main()
