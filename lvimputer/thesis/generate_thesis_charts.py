from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "imgs"
DATA_PATH = ROOT / "data" / "dataset1.csv"
NOTEBOOK_PATH = ROOT / "lstm_vae_imputation_torch.ipynb"
SEED = 42
MISSING_RATE = 0.30
SCENARIOS = ["START", "MIDDLE", "END", "RANDOM"]

BLUE = "#4267ac"
ORANGE = "#d95f45"
GREEN = "#3f7a5a"
PURPLE = "#7f4b86"
RED = "#b64c39"
GRAY = "#5f6f81"
LIGHT_GRID = "#d5dde6"


STAT_ORDER = [
    "Hybrid LSTM-VAE",
    "Mean",
    "Median",
    "Mode",
    "Random",
    "Hot-deck",
    "LOCF",
    "LOCB",
    "Forward fill",
    "Linear interpolation",
]

ML_ORDER = [
    "Hybrid LSTM-VAE",
    "KNN Regressor",
    "Ridge Regression",
    "HistGradientBoosting",
    "IterativeImputer (BR)",
]

STAT_MAE = {
    "Hybrid LSTM-VAE": [5.5709, 4.5137, 5.0384, 0.3524],
    "Mean": [9.2347, 7.0395, 5.8609, 7.3714],
    "Median": [9.1957, 6.6843, 6.3201, 7.2299],
    "Mode": [11.0104, 6.2012, 8.3491, 8.6309],
    "Random": [10.8356, 9.5952, 9.5612, 9.8093],
    "Hot-deck": [10.8356, 9.5952, 9.5612, 9.8093],
    "LOCF": [13.6231, 11.7102, 5.9014, 0.5226],
    "LOCB": [13.6231, 12.7757, 5.9014, 0.5231],
    "Forward fill": [13.6231, 11.7102, 5.9014, 0.5226],
    "Linear interpolation": [13.6231, 4.6072, 5.9014, 0.3571],
}

STAT_RMSE = {
    "Hybrid LSTM-VAE": [8.4133, 5.4412, 6.3708, 0.8826],
    "Mean": [11.5840, 7.9770, 6.9491, 8.9767],
    "Median": [11.8297, 7.8048, 7.5575, 9.1268],
    "Mode": [14.3724, 7.8215, 10.2349, 11.4638],
    "Random": [13.7920, 12.2936, 11.9564, 12.5566],
    "Hot-deck": [13.7920, 12.2936, 11.9564, 12.5566],
    "LOCF": [16.7863, 14.0247, 7.1699, 1.3634],
    "LOCB": [16.7863, 14.4487, 7.1699, 1.2813],
    "Forward fill": [16.7863, 14.0247, 7.1699, 1.3634],
    "Linear interpolation": [16.7863, 5.5114, 7.1699, 0.9214],
}

ML_MAE = {
    "Hybrid LSTM-VAE": [5.5709, 4.5137, 5.0384, 0.3524],
    "KNN Regressor": [13.6231, 14.1719, 14.4671, 3.2709],
    "Ridge Regression": [13.6231, 14.1898, 14.4819, 1.7082],
    "HistGradientBoosting": [13.7788, 8.1973, 11.0963, 0.8732],
    "IterativeImputer (BR)": [9.2346, 7.0269, 5.8575, 0.7717],
}

ML_RMSE = {
    "Hybrid LSTM-VAE": [8.4133, 5.4412, 6.3708, 0.8826],
    "KNN Regressor": [16.7863, 16.0915, 15.9828, 4.5291],
    "Ridge Regression": [16.7863, 16.1095, 15.9893, 2.4936],
    "HistGradientBoosting": [16.9188, 11.0179, 13.0327, 1.8788],
    "IterativeImputer (BR)": [11.5841, 7.9645, 6.9472, 1.6647],
}

RUNTIME = {
    "Hybrid LSTM-VAE": [2.1139, 0.1020, 1.8241, 0.1086],
    "Mean": [0.0005, 0.0003, 0.0004, 0.0002],
    "Median": [0.0004, 0.0003, 0.0004, 0.0003],
    "Mode": [0.0016, 0.0011, 0.0013, 0.0013],
    "Forward fill": [0.0012, 0.0008, 0.0010, 0.0012],
    "Linear interpolation": [0.0073, 0.0046, 0.0049, 0.0061],
    "KNN Regressor": [2.5203, 0.7712, 0.7229, 0.8390],
    "Ridge Regression": [0.0653, 0.0866, 0.0699, 0.0994],
    "HistGradientBoosting": [1.5530, 1.4957, 1.4340, 1.6974],
    "IterativeImputer (BR)": [29.8333, 28.8665, 25.4975, 27.4472],
}


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.family": "Segoe UI",
            "axes.edgecolor": "#25313d",
            "axes.labelcolor": "#17202a",
            "xtick.color": "#17202a",
            "ytick.color": "#17202a",
            "axes.titleweight": "semibold",
        }
    )


def save_fig(fig: plt.Figure, name: str) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG_DIR / name, bbox_inches="tight")
    plt.close(fig)


def load_temperature() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def build_masks(n: int) -> dict[str, np.ndarray]:
    missing_count = max(1, int(round(n * MISSING_RATE)))
    masks: dict[str, np.ndarray] = {}
    for scenario in SCENARIOS:
        mask = np.ones(n, dtype=bool)
        if scenario == "START":
            mask[:missing_count] = False
        elif scenario == "MIDDLE":
            start = max(0, (n - missing_count) // 2)
            mask[start : start + missing_count] = False
        elif scenario == "END":
            mask[-missing_count:] = False
        elif scenario == "RANDOM":
            rng = np.random.default_rng(SEED)
            hidden = rng.choice(n, size=missing_count, replace=False)
            mask[hidden] = False
        masks[scenario] = mask
    return masks


def comparison_segment(scenario: str, mask: np.ndarray, n: int, segment_len: int = 500) -> tuple[int, int]:
    segment_len = min(segment_len, n)
    if scenario == "START":
        start = 0
    elif scenario == "MIDDLE":
        start = max(0, (n - segment_len) // 2)
    elif scenario == "END":
        start = max(0, n - segment_len)
    else:
        missing_idx = np.flatnonzero(~mask)
        center = int(np.median(missing_idx)) if missing_idx.size else 0
        start = max(0, center - segment_len // 2)
        start = min(start, max(0, n - segment_len))
    return start, min(n, start + segment_len)


def simple_baselines(values: np.ndarray, index: pd.DatetimeIndex, mask: np.ndarray) -> dict[str, np.ndarray]:
    observed = values[mask]
    out: dict[str, np.ndarray] = {}
    out["Mean"] = np.where(mask, values, observed.mean()).astype(float)
    out["Median"] = np.where(mask, values, np.median(observed)).astype(float)

    ffill = values.astype(float).copy()
    ffill[~mask] = np.nan
    ffill = pd.Series(ffill, index=index).ffill().bfill().to_numpy()
    out["Forward fill"] = ffill

    interp = values.astype(float).copy()
    interp[~mask] = np.nan
    interp = pd.Series(interp, index=index).interpolate(method="linear").bfill().ffill().to_numpy()
    out["Linear interpolation"] = interp
    return out


def plot_raw_temperature(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.plot(df["datetime"], df["temperature"], color=BLUE, linewidth=0.7)
    ax.set_title("Raw temperature time series")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Temperature")
    ax.grid(True, alpha=0.28, color=LIGHT_GRID)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    save_fig(fig, "thesis-raw-temperature-timeseries.png")


def plot_missingness_positions(df: pd.DataFrame, masks: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    x = df["datetime"]
    y = df["temperature"].to_numpy()
    for ax, scenario in zip(axes, SCENARIOS):
        mask = masks[scenario]
        ax.plot(x, y, color="#9aa8b5", linewidth=0.65, label="Ground truth")
        ax.scatter(x[~mask], y[~mask], s=5, color=RED, alpha=0.55, label="Hidden")
        ax.set_title(f"{scenario} missingness")
        ax.grid(True, alpha=0.24, color=LIGHT_GRID)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    axes[0].legend(loc="upper right")
    fig.supylabel("Temperature")
    fig.suptitle("Synthetic missingness positions over the same sequence", y=1.01, fontweight="semibold")
    fig.tight_layout()
    save_fig(fig, "thesis-missingness-positions.png")


def plot_metric_panels(
    mae: dict[str, list[float]],
    rmse: dict[str, list[float]],
    methods: list[str],
    title: str,
    name: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=False)
    axes = axes.ravel()
    for i, (ax, scenario) in enumerate(zip(axes, SCENARIOS)):
        labels = methods
        x = np.arange(len(labels))
        width = 0.37
        mae_vals = [mae[m][i] for m in labels]
        rmse_vals = [rmse[m][i] for m in labels]
        colors_mae = [PURPLE if m == "Hybrid LSTM-VAE" else BLUE for m in labels]
        colors_rmse = [RED if m == "Hybrid LSTM-VAE" else ORANGE for m in labels]
        ax.bar(x - width / 2, mae_vals, width, label="MAE", color=colors_mae)
        ax.bar(x + width / 2, rmse_vals, width, label="RMSE", color=colors_rmse)
        ax.set_title(f"{scenario} missingness")
        ax.set_ylabel("Error")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25, color=LIGHT_GRID)
        ax.legend(fontsize=8)
    fig.suptitle(title, y=1.02, fontweight="semibold")
    fig.tight_layout()
    save_fig(fig, name)


def plot_baseline_segments(df: pd.DataFrame, masks: dict[str, np.ndarray]) -> None:
    values = df["temperature"].to_numpy(dtype=float)
    index = pd.DatetimeIndex(df["datetime"])
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.8), sharey=True)
    axes = axes.ravel()
    for ax, scenario in zip(axes, SCENARIOS):
        mask = masks[scenario]
        start, end = comparison_segment(scenario, mask, len(values))
        segment_index = index[start:end]
        observed = values.copy()
        observed[~mask] = np.nan
        baselines = simple_baselines(values, index, mask)
        ax.plot(segment_index, values[start:end], color="black", linewidth=1.4, label="Ground truth")
        ax.plot(segment_index, observed[start:end], color=GRAY, linewidth=1.0, alpha=0.8, label="Observed")
        ax.plot(segment_index, baselines["Mean"][start:end], color="#8a8f98", linewidth=1.0, label="Mean")
        ax.plot(segment_index, baselines["Median"][start:end], color="#6d6a9f", linewidth=1.0, label="Median")
        ax.plot(segment_index, baselines["Forward fill"][start:end], color=GREEN, linewidth=1.0, label="Forward fill")
        ax.plot(segment_index, baselines["Linear interpolation"][start:end], color=ORANGE, linewidth=1.15, label="Linear interpolation")
        ax.set_title(f"{scenario} missingness")
        ax.grid(True, alpha=0.24, color=LIGHT_GRID)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    axes[0].legend(fontsize=8, ncol=2)
    fig.supylabel("Temperature")
    fig.suptitle("Representative baseline imputations by missingness scenario", y=1.01, fontweight="semibold")
    fig.tight_layout()
    save_fig(fig, "thesis-baseline-segment-grid.png")


def plot_runtime() -> None:
    methods = list(RUNTIME)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=True)
    axes = axes.ravel()
    for i, (ax, scenario) in enumerate(zip(axes, SCENARIOS)):
        vals = [RUNTIME[m][i] for m in methods]
        colors = [PURPLE if m == "Hybrid LSTM-VAE" else BLUE for m in methods]
        ax.bar(np.arange(len(methods)), vals, color=colors)
        ax.set_yscale("log")
        ax.set_title(f"{scenario} missingness")
        ax.set_ylabel("Seconds (log scale)")
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.grid(axis="y", which="both", alpha=0.25, color=LIGHT_GRID)
    fig.suptitle("Imputation-only runtime by method and scenario", y=1.02, fontweight="semibold")
    fig.tight_layout()
    save_fig(fig, "thesis-runtime-comparison.png")


def notebook_pngs(cell_index: int) -> list[Image.Image]:
    if not NOTEBOOK_PATH.exists():
        return []
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    if cell_index >= len(nb.get("cells", [])):
        return []
    images: list[Image.Image] = []
    for output in nb["cells"][cell_index].get("outputs", []):
        data = output.get("data", {})
        if "image/png" not in data:
            continue
        encoded = data["image/png"]
        if isinstance(encoded, list):
            encoded = "".join(encoded)
        raw = base64.b64decode(encoded)
        images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
    return images


def save_image(image: Image.Image, name: str) -> Path:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    path = IMG_DIR / name
    image.save(path)
    return path


def contact_sheet(images: list[Image.Image], name: str, columns: int = 2, target_width: int = 1100) -> None:
    if not images:
        return
    resized: list[Image.Image] = []
    for img in images:
        scale = target_width / img.width
        resized.append(img.resize((target_width, int(img.height * scale)), Image.Resampling.LANCZOS))
    pad = 36
    rows = int(np.ceil(len(resized) / columns))
    cell_h = max(img.height for img in resized)
    sheet_w = columns * target_width + (columns + 1) * pad
    sheet_h = rows * cell_h + (rows + 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    for idx, img in enumerate(resized):
        row = idx // columns
        col = idx % columns
        x = pad + col * (target_width + pad)
        y = pad + row * (cell_h + pad)
        sheet.paste(img, (x, y))
    save_image(sheet, name)


def extract_notebook_outputs() -> None:
    training = notebook_pngs(10)
    if training:
        save_image(training[0], "thesis-training-loss.png")

    eval_images = notebook_pngs(11)
    if len(eval_images) >= 6:
        save_image(eval_images[0], "thesis-hybrid-error-by-scenario.png")
        names = ["start", "middle", "end", "random"]
        for label, img in zip(names, eval_images[1:5]):
            save_image(img, f"thesis-hybrid-vs-stat-{label}.png")
        save_image(eval_images[5], "thesis-hybrid-reconstructions.png")
        contact_sheet(eval_images[1:5], "thesis-hybrid-vs-stat-grid.png", columns=2, target_width=1050)

    ml_images = notebook_pngs(12)
    if len(ml_images) >= 5:
        names = ["start", "middle", "end", "random"]
        for label, img in zip(names, ml_images[:4]):
            save_image(img, f"thesis-hybrid-vs-ml-{label}.png")
        contact_sheet(ml_images[:4], "thesis-hybrid-vs-ml-visual-grid.png", columns=2, target_width=1050)
        save_image(ml_images[4], "thesis-ml-baseline-errors-notebook.png")


def main() -> None:
    setup_matplotlib()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_temperature()
    masks = build_masks(len(df))

    plot_raw_temperature(df)
    plot_missingness_positions(df, masks)
    plot_baseline_segments(df, masks)
    plot_metric_panels(
        STAT_MAE,
        STAT_RMSE,
        STAT_ORDER,
        "Hybrid LSTM-VAE versus statistical/time-aware baselines",
        "thesis-stat-baseline-errors.png",
    )
    plot_metric_panels(
        ML_MAE,
        ML_RMSE,
        ML_ORDER,
        "Hybrid LSTM-VAE versus machine-learning baselines",
        "thesis-ml-baseline-errors.png",
    )
    plot_runtime()
    extract_notebook_outputs()
    print(f"Wrote thesis charts to {IMG_DIR}")


if __name__ == "__main__":
    main()
