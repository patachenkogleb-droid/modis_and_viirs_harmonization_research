from collections import Counter

import numpy as np
import pandas as pd
from pyproj import Transformer

from experiments.run import HOLDOUT, TRAIN, MissingInputs, run
from src.calibration.harmonizer import error_metrics, fit_power, frp_bin, summarize_ratio
from src.grid.spatial_grid import ORIGIN_X, ORIGIN_Y, add_grid
from src.landcover.landcover_mapper import IGBP_GROUPS, attach_landcover
from src.pairing.match_sensors import aggregate_passes

LAT_EDGES = [-np.inf, 36, 40, 44, np.inf]
LAT_LABELS = ["<36", "36–<40", "40–<44", "≥44"]
MIN_PAIRS, MIN_WEEKS, MIN_CELLS = 100, 12, 8
MIN_COVER_SHARE = .60
PHASE = ("daynight",)
FRP = ("daynight", "frp_band")
COVER = ("daynight", "strong_cover")
LAT = ("daynight", "latitude_band")
FRP_LAT = (*FRP, "latitude_band")
FRP_COVER = (*FRP, "strong_cover")
ALL_FACTORS = (*FRP_COVER, "latitude_band")
MODELS = {
    "phase": (PHASE,),
    "phase_frp": (PHASE, FRP),
    "phase_cover": (PHASE, COVER),
    "phase_lat": (PHASE, LAT),
    "phase_frp_lat": (PHASE, FRP, FRP_LAT),
    "phase_frp_cover": (PHASE, FRP, FRP_COVER),
    "phase_frp_cover_lat": (PHASE, FRP, FRP_COVER, ALL_FACTORS),
}
COMPARATORS = {
    "static": "raw", "phase": "static", "phase_power": "phase", "phase_frp": "phase",
    "phase_cover": "phase", "phase_lat": "phase", "phase_frp_lat": "phase_frp",
    "phase_frp_cover": "phase_frp", "phase_frp_cover_lat": "phase_frp_cover",
}
_INVERSE = Transformer.from_crs(6933, 4326, always_xy=True)


def annotate(frame, context):
    frame = attach_landcover(frame, context.data / "landcover", context.grid_size, audit=True)
    x = (frame.cell_x.to_numpy() + .5) * context.grid_size + ORIGIN_X
    y = (frame.cell_y.to_numpy() + .5) * context.grid_size + ORIGIN_Y
    _, latitude = _INVERSE.transform(x, y)
    strong = frame.grouped_share.ge(MIN_COVER_SHARE) & frame.grouped_landcover.isin(IGBP_GROUPS.values())
    return frame.assign(
        latitude_band=pd.cut(latitude, LAT_EDGES, labels=LAT_LABELS, right=False),
        frp_band=frp_bin(frame.sum_frp_modis),
        strong_cover=frame.grouped_landcover.where(strong),
    )


def fit(frame):
    total = frame.sum_frp_modis.sum()
    if total <= 0:
        raise MissingInputs("Positive Aqua FRP is required to fit daily calibration")
    k = float(frame.sum_frp_viirs.sum() / total)
    metadata = frame.assign(
        week=frame.acq_date.dt.to_period("W"),
        cell=pd.factorize(pd.MultiIndex.from_frame(frame[["cell_x", "cell_y"]]))[0],
    )
    levels = tuple(dict.fromkeys(keys for model in MODELS.values() for keys in model))
    tables = {}
    for keys in levels:
        table = metadata.groupby(list(keys), observed=True).agg(
            pairs=("sum_frp_modis", "size"), weeks=("week", "nunique"), cells=("cell", "nunique"),
            aqua=("sum_frp_modis", "sum"), snpp=("sum_frp_viirs", "sum"),
        )
        table["supported"] = (table.pairs.ge(MIN_PAIRS) & table.weeks.ge(MIN_WEEKS)
                              & table.cells.ge(MIN_CELLS) & table.aqua.gt(0))
        table["k"] = (table.snpp / table.aqua.where(table.aqua.gt(0))).where(table.supported)
        tables[keys] = table
    powers = {phase: fit_power(group) for phase, group in frame.groupby("daynight")}
    return k, tables, powers


def predict(frame, model):
    k, tables, powers = model
    modis = frame.sum_frp_modis
    scales = {keys: frame.join(table["k"].rename("_k"), on=list(keys))["_k"]
              for keys, table in tables.items()}
    predictions = {"raw": modis, "static": modis * k}
    for name, levels in MODELS.items():
        scale = scales[levels[0]].fillna(k)
        for keys in levels[1:]:
            scale = scales[keys].fillna(scale)
        predictions[name] = modis * scale
    power = predictions["phase"].copy()
    for phase, params in powers.items():
        if params and params["exponent"] > 0:
            active = frame.daynight.eq(phase)
            power.loc[active] = params["scale"] * np.power(modis.loc[active], params["exponent"])
    predictions["phase_power"] = power
    return frame.assign(**predictions), list(predictions)


def phases(frame):
    return [(phase, frame.loc[frame.daynight.eq(phase)]) for phase in ("D", "N")] + [("all", frame)]


def paired_daily(frame, methods):
    return frame.groupby("acq_date")[["sum_frp_viirs", *methods]].sum().rename(
        columns={"sum_frp_viirs": "observed"}
    )


def reported_daily(aqua, snpp, methods):
    predicted = aqua.groupby("acq_date")[methods].sum()
    observed = snpp.groupby("acq_date")["sum_frp"].sum().rename("observed")
    return predicted.join(observed, how="inner")


def mae_difference_interval(observed, candidate, reference):
    diffs = (candidate - observed).abs() - (reference - observed).abs()
    blocks = diffs.groupby(diffs.index.to_period("W")).agg(["sum", "size"]).to_numpy()
    if len(blocks) < 2:
        return [None, None]
    rng = np.random.default_rng(229)
    samples = rng.integers(0, len(blocks), size=(2_000, len(blocks)))
    totals = blocks[samples].sum(axis=1)
    return np.quantile(totals[:, 0] / totals[:, 1], [.025, .975]).tolist()


def evaluate(pairs, aqua, snpp, model, period):
    matched, methods = predict(pairs, model)
    all_aqua, _ = predict(aqua, model)
    views = []
    for (phase, m), (_, a), (_, v) in zip(phases(matched), phases(all_aqua), phases(snpp)):
        views.append(("matched", phase, paired_daily(m, methods)))
        views.append(("reported", phase, reported_daily(a, v, methods)))
    scores = []
    series = []
    for scope, phase, daily in views:
        if daily.empty:
            continue
        series.append(daily.reset_index().assign(scope=scope, phase=phase, period=period))
        for name in methods:
            baseline = COMPARATORS.get(name)
            metrics = error_metrics(daily.observed, daily[name])
            delta = None
            bounds = [None, None]
            if baseline:
                delta = metrics["mae_mw"] - error_metrics(daily.observed, daily[baseline])["mae_mw"]
                if period == "holdout":
                    bounds = mae_difference_interval(daily.observed, daily[name], daily[baseline])
            scores.append(dict(period=period, scope=scope, phase=phase, method=name, days=len(daily),
                               **metrics, compare_to=baseline, delta_mae_mw=delta,
                               ci95_low_mw=bounds[0], ci95_high_mw=bounds[1]))
    return scores, series


def detection_coverage(aqua, snpp):
    coverage = []
    for (phase, a), (_, v) in zip(phases(aqua), phases(snpp)):
        a_days = set(a.acq_date.unique())
        v_days = set(v.acq_date.unique())
        calendar = set(pd.date_range(f"{HOLDOUT[0]}-01-01", f"{HOLDOUT[1]}-12-31"))
        coverage.append(dict(phase=phase, aqua_passes=len(a), snpp_passes=len(v),
                             both_days=len(a_days & v_days), aqua_only_days=len(a_days - v_days),
                             snpp_only_days=len(v_days - a_days), neither_days=len(calendar - (a_days | v_days))))
    return coverage


def landcover_audit(frame, period):
    cell_years = frame.assign(year=frame.acq_date.dt.year).drop_duplicates(["year", "cell_x", "cell_y"])
    return {
        "period": period, "cell_years": len(cell_years), "pairs": len(frame),
        "changed_cell_years": int(cell_years.landcover.ne(cell_years.grouped_landcover).sum()),
        "changed_pairs": int(frame.landcover.ne(frame.grouped_landcover).sum()),
        "strong_cell_years": int(cell_years.strong_cover.notna().sum()),
        "strong_pairs": int(frame.strong_cover.notna().sum()),
        "strong_aqua_frp_share": float(frame.loc[frame.strong_cover.notna(), "sum_frp_modis"].sum()
                                      / frame.sum_frp_modis.sum()),
        "median_grouped_share": float(cell_years.grouped_share.median()),
        "missing_cell_years": int(cell_years.valid_share.lt(.5).sum()),
    }


def latitude_rows(frame, period):
    rows = []
    for phase, subset in phases(frame):
        for band, group in subset.groupby("latitude_band", observed=True):
            rows.append(dict(period=period, phase=phase, latitude_band=band,
                             cells=len(group[["cell_x", "cell_y"]].drop_duplicates()),
                             median_offset_minutes=float(group.delta_minutes.median()),
                             **summarize_ratio(group)))
    return rows


def load_period(context, years):
    points = context.load(*years)
    pairs, coverage = context.pair(points)
    if pairs.empty:
        raise MissingInputs(f"No matches in {years[0]}–{years[1]}")
    passes = aggregate_passes(add_grid(points, context.grid_size))
    aqua = passes.loc[passes.sensor.eq("aqua")].rename(columns={"sum_frp": "sum_frp_modis"})
    snpp = passes.loc[passes.sensor.eq("snpp"), ["acq_date", "daynight", "sum_frp"]]
    return annotate(pairs, context), annotate(aqua, context), snpp, coverage


def experiment(context):
    train, train_aqua, train_snpp, train_coverage = load_period(context, TRAIN)
    holdout, aqua, snpp, test_coverage = load_period(context, HOLDOUT)
    early = train.loc[train.acq_date.dt.year.le(2014)]
    validation = train.loc[train.acq_date.dt.year.ge(2015)]
    val_scores, val_series = evaluate(validation, train_aqua.loc[train_aqua.acq_date.dt.year.ge(2015)],
                                     train_snpp.loc[train_snpp.acq_date.dt.year.ge(2015)], fit(early), "validation")
    model = fit(train)
    test_scores, test_series = evaluate(holdout, aqua, snpp, model, "holdout")
    scores = val_scores + test_scores
    rankings = pd.DataFrame(val_scores).sort_values("mae_mw").drop_duplicates(["scope", "phase"])
    winners = rankings[["scope", "phase", "method"]].to_dict("records")
    latitude = latitude_rows(train, "training") + latitude_rows(holdout, "holdout")
    audit = [landcover_audit(train, "training"), landcover_audit(holdout, "holdout")]
    coverage = detection_coverage(aqua, snpp)
    context.write_table("e7_daily_metrics.csv", scores)
    pd.concat(val_series + test_series, ignore_index=True).to_csv(context.output / "e7_daily_series.csv", index=False)
    context.write_table("e7_latitude_bands.csv", latitude)
    context.write_table("e7_landcover_audit.csv", audit)
    context.write_table("e7_detection_coverage.csv", coverage)
    fitted = [table.reset_index().assign(level="+".join(keys)) for keys, table in model[1].items()]
    pd.concat(fitted, ignore_index=True).to_csv(context.output / "e7_fitted_scales.csv", index=False)
    for scope in ("matched", "reported"):
        daytime = [row for row in test_scores if row["scope"] == scope and row["phase"] == "D"]
        if daytime:
            context.plot(f"e7_daily_{scope}.png", [row["method"].replace("_", " ") for row in daytime],
                         [row["mae_mw"] for row in daytime], "Daily FRP sum MAE (MW)", "Model",
                         x_rotation=35)
    return {
        "status": "RUN" if context.archive_complete(TRAIN[0], HOLDOUT[1]) else "PARTIAL",
        "static_k_both_phases": model[0], "validation_winners": winners,
        "training_pair_coverage": train_coverage, "holdout_pair_coverage": test_coverage,
        "holdout_detection_coverage": coverage, "landcover_audit": audit,
        "group_support": {"+".join(keys): dict(Counter(table.supported.map({True: "yes", False: "no"})))
                          for keys, table in model[1].items()},
        "scores": scores,
        "scope": "UTC dates. Matched pairs and all reported detections on dates with both sensors.",
    }


if __name__ == "__main__":
    run([7])
