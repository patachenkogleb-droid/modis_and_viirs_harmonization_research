import numpy as np
import pandas as pd

from experiments.run import HOLDOUT, TRAIN, MissingInputs, run
from src.calibration.harmonizer import error_metrics
from src.grid.spatial_grid import add_grid
from src.pairing.match_sensors import PASS_KEYS, aggregate_passes, match_aggregated_passes
from src.utils.benchmark import measure

SENSORS = {"aqua": "time_aqua", "snpp": "time_snpp"}
LABELS = {
    "matched": "Matched",
    "same_cell_competing": "Same cell, competing",
    "neighbor_within_window": "Adjacent cell",
    "same_cell_other_time": "Same cell, other time",
    "region_within_window": "Elsewhere in region",
    "no_close_record": "No close record",
}
WINDOW_MINUTES = 60


def mark_matches(passes, pairs):
    keys = [*PASS_KEYS, "time"]
    groups = []
    for sensor, time_column in SENSORS.items():
        assigned = pairs[[*PASS_KEYS, time_column]].rename(columns={time_column: "time"})
        observations = passes.loc[passes.sensor.eq(sensor)]
        marked = observations.merge(assigned.assign(matched=True), on=keys, how="left",
                                    validate="one_to_one")
        marked["matched"] = marked.matched.eq(True)
        groups.append(marked)
    return pd.concat(groups, ignore_index=True)


def nearest_minutes(left, right, keys):
    matched = pd.merge_asof(left, right, left_on="time", right_on="other_time",
                            by=keys, direction="nearest")
    return (matched.time - matched.other_time).abs().dt.total_seconds().to_numpy() / 60


def classify_unpaired(observations):
    parts = []
    for sensor in SENSORS:
        left = observations.loc[observations.sensor.eq(sensor) & ~observations.matched].sort_values("time").copy()
        right = observations.loc[observations.sensor.ne(sensor), [*PASS_KEYS, "time"]].rename(
            columns={"time": "other_time"}).sort_values("other_time")
        same_cell = nearest_minutes(left, right, PASS_KEYS)
        neighbor = np.full(len(left), np.inf)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                shifted = right.assign(cell_x=right.cell_x + dx, cell_y=right.cell_y + dy)
                neighbor = np.fmin(neighbor, nearest_minutes(left, shifted, PASS_KEYS))
        region = nearest_minutes(left, right, ["acq_date", "daynight"])
        left["evidence"] = np.select(
            [same_cell < WINDOW_MINUTES, neighbor < WINDOW_MINUTES,
             np.isfinite(same_cell), region < WINDOW_MINUTES],
            ["same_cell_competing", "neighbor_within_window", "same_cell_other_time",
             "region_within_window"],
            default="no_close_record",
        )
        parts.append(left)
    return pd.concat([observations.loc[observations.matched].assign(evidence="matched"), *parts],
                     ignore_index=True)


def evidence_rows(observations):
    output = []
    for (phase, sensor), group in observations.groupby(["daynight", "sensor"]):
        total_frp = group.sum_frp.sum()
        for label in LABELS:
            selected = group.loc[group.evidence.eq(label), "sum_frp"]
            output.append(dict(
                phase=phase, sensor=sensor, evidence=label, passes=len(selected),
                pass_fraction=len(selected) / len(group), frp_mw=float(selected.sum()),
                frp_fraction=float(selected.sum() / total_frp) if total_frp else None,
                median_frp_mw=float(selected.median()) if len(selected) else None,
                p90_frp_mw=float(selected.quantile(.9)) if len(selected) else None,
            ))
    return output


def daily_totals(observations, phase):
    phase_data = observations if phase == "all" else observations.loc[observations.daynight.eq(phase)]
    sums = {}
    for sensor in SENSORS:
        source = phase_data.loc[phase_data.sensor.eq(sensor)]
        total = source.groupby("acq_date").sum_frp.sum()
        # On a date with records from this sensor, no matched contribution means zero
        # matched FRP. This is an accounting identity, not a zero for missing observations.
        common = source.loc[source.matched].groupby("acq_date").sum_frp.sum().reindex(total.index, fill_value=0)
        sums[sensor] = pd.DataFrame({sensor: total, f"matched_{sensor}": common})
    days = sums["aqua"].join(sums["snpp"], how="inner")
    for sensor in SENSORS:
        days[f"unpaired_{sensor}"] = days[sensor] - days[f"matched_{sensor}"]
    return days


def ratio(numerator, denominator):
    total = denominator.sum()
    if total <= 0:
        raise MissingInputs("Positive Aqua FRP is required for the aggregate comparison")
    return float(numerator.sum() / total)


def compare_daily(train, holdout):
    coefficients, metrics, series = [], [], []
    for phase in ("D", "N", "all"):
        reference = daily_totals(train, phase)
        test = daily_totals(holdout, phase)
        if reference.empty or test.empty:
            continue
        k_matched = ratio(reference.matched_snpp, reference.matched_aqua)
        k_regional = ratio(reference.snpp, reference.aqua)
        coefficients.append(dict(
            phase=phase, training_days=len(reference), holdout_days=len(test),
            k_training_matched=k_matched, k_training_regional=k_regional,
            k_holdout_matched=ratio(test.matched_snpp, test.matched_aqua),
            k_holdout_regional=ratio(test.snpp, test.aqua),
            mean_aqua_minus_snpp_matched_mw=float((test.matched_aqua - test.matched_snpp).mean()),
            mean_unpaired_aqua_mw=float(test.unpaired_aqua.mean()),
            mean_unpaired_snpp_mw=float(test.unpaired_snpp.mean()),
        ))
        predictions = {"raw": test.aqua, "matched_multiplier": test.aqua * k_matched,
                       "regional_multiplier": test.aqua * k_regional}
        for name, prediction in predictions.items():
            metrics.append(dict(phase=phase, method=name, days=len(test),
                                **error_metrics(test.snpp, prediction)))
        series.append(test.assign(phase=phase, **predictions).reset_index())
    if not series:
        raise MissingInputs("Both sensors need to report on at least one common date")
    return coefficients, metrics, pd.concat(series, ignore_index=True)


def load_observations(context, years):
    points = context.load(*years)
    with measure("grid_and_aggregate") as record:
        passes = aggregate_passes(add_grid(points, context.grid_size))
    record.update(rows=len(points), passes=len(passes))
    context.metrics.append(record)
    with measure("pairing") as record:
        pairs, coverage = match_aggregated_passes(passes, WINDOW_MINUTES)
        marked = mark_matches(passes, pairs)
    record.update(coverage)
    context.metrics.append(record)
    return passes, marked, coverage


def experiment(context):
    _, train, train_coverage = load_observations(context, TRAIN)
    passes, holdout, test_coverage = load_observations(context, HOLDOUT)
    with measure("unpaired_evidence") as record:
        classified = classify_unpaired(holdout)
    record["rows"] = len(classified)
    context.metrics.append(record)
    evidence = evidence_rows(classified)
    coefficients, metrics, daily = compare_daily(train, holdout)
    sensitivity = []
    for minutes in (30, 60, 90):
        coverage = test_coverage if minutes == WINDOW_MINUTES else match_aggregated_passes(passes, minutes)[1]
        sensitivity.append(dict(window_minutes=minutes, **coverage))
    context.write_table("e8_unpaired_evidence.csv", evidence)
    context.write_table("e8_daily_coefficients.csv", coefficients)
    context.write_table("e8_daily_metrics.csv", metrics)
    context.write_table("e8_pair_window_sensitivity.csv", sensitivity)
    daily.to_csv(context.output / "e8_daily_decomposition.csv", index=False)
    daytime = [row for row in metrics if row["phase"] == "D"]
    if daytime:
        context.plot("e8_daily_calibration.png", [row["method"].replace("_", " ") for row in daytime],
                     [row["mae_mw"] for row in daytime], "Daily FRP sum MAE (MW)", "Model")
    return {
        "status": "RUN" if context.archive_complete(TRAIN[0], HOLDOUT[1]) else "PARTIAL",
        "training_coverage": train_coverage, "holdout_coverage": test_coverage,
        "evidence": evidence, "coefficients": coefficients, "daily_metrics": metrics,
        "window_sensitivity": sensitivity,
        "scope": "Reported vegetation detections, same UTC date and phase. Nearby detections are proxies, "
                 "and cannot establish clear coverage or identify a missed fire.",
    }


if __name__ == "__main__":
    run([8])
