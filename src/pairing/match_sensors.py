import pandas as pd

PASS_KEYS = ["cell_x", "cell_y", "acq_date", "daynight"]
PAIR_COLUMNS = PASS_KEYS + ["time_aqua", "time_snpp", "sum_frp_modis", "sum_frp_viirs",
                            "rows_aqua", "rows_snpp", "delta_minutes", "signed_minutes"]


def aggregate_passes(frame, gap_minutes: int = 10, *, with_scan=False):
    keys = PASS_KEYS + ["sensor"]
    ordered = frame.sort_values(keys + ["timestamp"]).copy()
    differences = ordered.groupby(keys, sort=False)["timestamp"].diff()
    ordered["new_pass"] = differences.gt(pd.Timedelta(minutes=gap_minutes))
    ordered["pass_number"] = ordered.groupby(keys, sort=False)["new_pass"].cumsum()
    measures = dict(time=("timestamp", "median"), sum_frp=("frp", "sum"), rows=("frp", "size"))
    if with_scan:
        measures["scan_aqua"] = ("scan", "median")
    return ordered.groupby(keys + ["pass_number"], as_index=False, sort=False).agg(**measures)


def export_passes_csv(frame, destination):
    passes = aggregate_passes(frame)
    passes["time_epoch_ms"] = passes.pop("time").astype("datetime64[ms]").astype("int64")
    passes.to_csv(destination, index=False)


def match_passes(frame, max_minutes: float = 60):
    return match_aggregated_passes(aggregate_passes(frame), max_minutes)


def match_aggregated_passes(passes, max_minutes: float = 60):
    aqua = passes.loc[passes.sensor.eq("aqua")]
    snpp = passes.loc[passes.sensor.eq("snpp")]
    candidates = aqua.merge(snpp, on=PASS_KEYS, suffixes=("_aqua", "_snpp"))
    candidates["signed_minutes"] = (
        candidates.time_snpp - candidates.time_aqua
    ).dt.total_seconds() / 60
    candidates["delta_minutes"] = candidates.signed_minutes.abs()
    candidates = candidates.loc[candidates.delta_minutes.lt(max_minutes)].sort_values(
        ["delta_minutes", "time_aqua", "time_snpp", *PASS_KEYS]
    )
    competing = candidates.duplicated(PASS_KEYS, keep=False)
    selected = list(candidates.index[~competing])
    for _, group in candidates.loc[competing].groupby(PASS_KEYS, sort=False):
        aqua_used = set()
        snpp_used = set()
        for row in group.itertuples():
            if row.pass_number_aqua in aqua_used or row.pass_number_snpp in snpp_used:
                continue
            aqua_used.add(row.pass_number_aqua)
            snpp_used.add(row.pass_number_snpp)
            selected.append(row.Index)
    paired = candidates.loc[selected].rename(columns={
        "sum_frp_aqua": "sum_frp_modis", "sum_frp_snpp": "sum_frp_viirs"
    })
    counts = passes.sensor.value_counts()
    coverage = {
        "aqua_passes": int(counts.get("aqua", 0)),
        "snpp_passes": int(counts.get("snpp", 0)),
        "matched_pairs": len(paired),
        "unpaired_aqua": int(counts.get("aqua", 0)) - len(paired),
        "unpaired_snpp": int(counts.get("snpp", 0)) - len(paired),
    }
    return paired.loc[:, PAIR_COLUMNS].reset_index(drop=True), coverage
