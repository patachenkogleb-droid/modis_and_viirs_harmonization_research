import numpy as np

from experiments.run import HOLDOUT, TRAIN, MissingInputs, run
from src.calibration.harmonizer import BIN_LABELS, error_metrics, fit_power, frp_bin, summarize_ratio
from src.landcover.landcover_mapper import IGBP_GROUPS, attach_landcover

MIN_GROUP_PAIRS = 30
MIN_GROUP_WEEKS = 4


def supported_scale(frame):
    modis = frame.sum_frp_modis.sum()
    if (len(frame) >= MIN_GROUP_PAIRS and modis > 0
            and frame.acq_date.dt.to_period("W").nunique() >= MIN_GROUP_WEEKS):
        return float(frame.sum_frp_viirs.sum() / modis)
    return None


def fit_models(train):
    modis = train.sum_frp_modis.sum()
    if modis <= 0:
        raise MissingInputs("Training MODIS FRP sum must be positive")
    static = float(train.sum_frp_viirs.sum() / modis)
    labels = frp_bin(train.sum_frp_modis)
    bin_scales = {label: supported_scale(train.loc[labels.eq(label)]) for label in BIN_LABELS}
    land_scales = {}
    if "landcover" in train:
        for label in sorted(set(IGBP_GROUPS.values())):
            scale = supported_scale(train.loc[train.landcover.eq(label)])
            if scale is not None:
                land_scales[label] = scale
    return {"static": static, "binned": {label: static if scale is None else scale
                                          for label, scale in bin_scales.items()},
            "power": fit_power(train), "landcover": land_scales}


def predict(frame, model):
    modis = frame.sum_frp_modis
    static = model["static"]
    predictions = {
        "raw_modis": modis,
        "static": modis * static,
        "binned": modis * frp_bin(modis).astype("string").map(model["binned"]).fillna(static),
    }
    power = model["power"]
    if power and power["exponent"] > 0:
        predictions["power"] = power["scale"] * np.power(modis, power["exponent"])
    landcover = model["landcover"]
    if landcover and frame.landcover.isin(landcover).any():
        predictions["landcover"] = modis * frame.landcover.map(landcover).fillna(static)
    return predictions


def weekly_metrics(frame, predictions):
    columns = ["sum_frp_viirs", *predictions]
    points = frame.assign(**predictions)
    weeks = points.acq_date.dt.to_period("W").dt.start_time
    totals = points.groupby(weeks)[columns].sum()
    totals["pairs"] = points.groupby(weeks).size()
    metrics = [dict(method=method, weeks=len(totals),
                    **error_metrics(totals.sum_frp_viirs, totals[method]))
               for method in predictions]
    return totals.reset_index(names="week_start"), metrics


def experiment(context):
    train = context.load(*TRAIN, daynight="D")
    holdout = context.load(*HOLDOUT, daynight="D")
    train, train_coverage = context.pair(train)
    holdout, test_coverage = context.pair(holdout)
    if train.empty or holdout.empty:
        raise MissingInputs("Training and holdout both require matched detections")
    raster_dir = context.data / "landcover"
    if any(raster_dir.glob("LC_Type1_*.tif")):
        train = attach_landcover(train, raster_dir, context.grid_size)
        holdout = attach_landcover(holdout, raster_dir, context.grid_size)

    early = train.loc[train.acq_date.dt.year.le(2014)]
    validation = train.loc[train.acq_date.dt.year.ge(2015)]
    validation_rows = []
    selected = None
    validation_weekly = []
    selected_weekly = None
    if early.sum_frp_modis.sum() > 0 and not validation.empty:
        validation_predictions = predict(validation, fit_models(early))
        for method, predicted in validation_predictions.items():
            validation_rows.append(dict(method=method, pairs=len(validation),
                                        **error_metrics(validation.sum_frp_viirs, predicted)))
        selected = min(validation_rows, key=lambda row: row["mae_mw"])["method"]
        _, validation_weekly = weekly_metrics(validation, validation_predictions)
        selected_weekly = min(validation_weekly, key=lambda row: row["mae_mw"])["method"]
        context.write_table("e6_training_validation.csv", validation_rows)
        context.write_table("e6_training_validation_weekly.csv", validation_weekly)

    model = fit_models(train)
    predictions = predict(holdout, model)
    methods = list(predictions)
    weekly, weekly_results = weekly_metrics(holdout, predictions)
    weekly.to_csv(context.output / "e6_weekly_series.csv", index=False)
    context.write_table("e6_weekly_backtest.csv", weekly_results)
    context.plot("e6_weekly_backtest.png", methods, [row["mae_mw"] for row in weekly_results],
                 "MAE of weekly FRP sums (MW)", "Model")
    holdout = holdout.assign(**predictions)
    rows = []
    groups = [("all", holdout), *holdout.groupby(holdout.acq_date.dt.year)]
    for year, group in groups:
        for method in methods:
            rows.append(dict(year=year if year == "all" else int(year), method=method,
                             pairs=len(group), **error_metrics(group.sum_frp_viirs, group[method])))
    context.write_table("e6_backtest.csv", rows)
    context.plot("e6_backtest.png", methods, [row["mae_mw"] for row in rows if row["year"] == "all"],
                 "MAE (MW)", "Model")
    train_years = sorted(train.acq_date.dt.year.unique().tolist())
    holdout_years = sorted(holdout.acq_date.dt.year.unique().tolist())
    complete = train_years == list(range(TRAIN[0], TRAIN[1] + 1)) and holdout_years == list(
        range(HOLDOUT[0], HOLDOUT[1] + 1))
    return {"status": "RUN" if complete and context.archive_complete(TRAIN[0], HOLDOUT[1]) else "PARTIAL", "summary": rows,
            "models": model, "static_training": summarize_ratio(train),
            "training_validation": validation_rows, "selected_on_training_validation": selected,
            "validation_weekly": validation_weekly, "selected_on_training_weekly": selected_weekly,
            "weekly_summary": weekly_results,
            "train_years": train_years, "holdout_years": holdout_years,
            "train_coverage": train_coverage, "holdout_coverage": test_coverage,
            "scope": "Evaluation on matched detection pairs. Raw MODIS is eligible for training selection."}


if __name__ == "__main__":
    run([6])
