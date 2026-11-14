import numpy as np
import pandas as pd

BIN_EDGES = [0, 10, 30, 100, np.inf]
BIN_LABELS = ["0–<10", "10–<30", "30–<100", "≥100"]
FRP_FIELDS = ["sum_frp_modis", "sum_frp_viirs"]


def frp_bin(modis):
    return pd.cut(modis, bins=BIN_EDGES, labels=BIN_LABELS, right=False)


def summarize_ratio(pairs, draws: int = 1_000, seed: int = 61):
    modis, viirs = pairs[FRP_FIELDS].sum().to_numpy(dtype=float)
    k = float(viirs / modis) if modis > 0 else None
    weeks = pairs.acq_date.dt.to_period("W").astype(str)
    blocks = pairs.groupby(weeks)[FRP_FIELDS].sum().to_numpy(dtype=float)
    interval = [None, None]
    if k is not None and len(blocks) >= 2:
        rng = np.random.default_rng(seed)
        ratios = []
        for _ in range(draws):
            sample = blocks[rng.integers(0, len(blocks), len(blocks))].sum(axis=0)
            if sample[0] > 0:
                ratios.append(sample[1] / sample[0])
        if ratios:
            interval = np.quantile(ratios, [0.025, 0.975]).tolist()
    return {"pairs": len(pairs), "week_blocks": len(blocks), "sum_frp_modis": modis,
            "sum_frp_viirs": viirs, "k": k, "ci95_low": interval[0], "ci95_high": interval[1]}


def contrast_ratio(left, right, comparisons=1, draws=8_000, seed=79):
    def by_week(pairs):
        return pairs.groupby(pairs.acq_date.dt.to_period("W"))[FRP_FIELDS].sum()

    blocks = by_week(left).join(by_week(right), how="outer", lsuffix="_a", rsuffix="_b").fillna(0)
    if len(blocks) < 2:
        return [None, None, None, None]
    values = blocks.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        am, av, bm, bv = values[rng.integers(0, len(values), len(values))].sum(axis=0)
        if am > 0 and bm > 0:
            differences.append(av / am - bv / bm)
    if not differences:
        return [None, None, None, None]
    adjusted_tail = 0.025 / max(comparisons, 1)
    return np.quantile(differences, [0.025, 0.975, adjusted_tail, 1 - adjusted_tail]).tolist()


def error_metrics(observed, predicted):
    error = np.asarray(predicted) - np.asarray(observed)
    return {"mae_mw": float(np.abs(error).mean()), "bias_mw": float(error.mean())}


def fit_power(pairs):
    frame = pairs.loc[pairs.sum_frp_modis.gt(0) & pairs.sum_frp_viirs.gt(0), FRP_FIELDS]
    if len(frame) < 2 or frame.sum_frp_modis.nunique() < 2:
        return None
    exponent, log_scale = np.polyfit(np.log(frame.sum_frp_modis), np.log(frame.sum_frp_viirs), 1)
    return {"scale": float(np.exp(log_scale)), "exponent": float(exponent)}
