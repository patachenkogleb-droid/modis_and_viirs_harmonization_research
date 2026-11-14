"""E10: fit reported daily FRP against MAE and export calendar lookup factors."""
import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import duckdb
import numpy as np
import pandas as pd

from experiments.regional_daily_inputs import GRID, PHASES, REGIONS, daily_summary, read_year
from experiments.regional_daily_models import MIN_DAYS, MIN_WEEKS, bootstrap, week
from experiments.run import Context

FILENAME = "frp_modis_viirs_coefficients.csv"
CANDIDATES = ("identity", "ratio_all", "ratio_dn", "mae_all", "mae_dn", "mae_joint_dn")
ONE = dict.fromkeys(PHASES, 1.)
REGIONAL_CONTROL = (2017, 2020)
FRESH_CONTROL = (2021, 2025)


def new_western_days(data):
    if not Context(data, GRID, None).archive_complete(*FRESH_CONTROL):
        raise ValueError("The 2021–2025 western Area API manifest is incomplete")
    rel, bbox = REGIONS["western_us"]
    days, audits = [], []
    with duckdb.connect(config={"threads": "4"}) as db:
        for year in range(FRESH_CONTROL[0], FRESH_CONTROL[1] + 1):
            print(f"Preparing western_us {year}", flush=True)
            points, audit = read_year(db, data / rel, bbox, year)
            # Reported sums are additive; no pairing or grid is needed for this bbox.
            observations = points.rename(columns={"frp": "sum_frp"}).assign(matched=False)
            daily = daily_summary(observations)
            days.append(daily.assign(region="western_us", year=year))
            audits.append(audit.assign(year=year))
    return pd.concat(days, ignore_index=True), pd.concat(audits, ignore_index=True)


def exact_mae_factor(aqua, snpp):
    """argmin sum |k*aqua - snpp|, weighted median with weights Aqua FRP."""
    a, s = np.asarray(aqua), np.asarray(snpp)
    valid = a > 0
    a, s = a[valid], s[valid]
    ratios = s / a
    order = np.argsort(ratios, kind="stable")
    weights = np.cumsum(a[order])
    return float(ratios[order[np.searchsorted(weights, .5 * weights[-1])]])


def fit(days):
    data = {phase: days.loc[days.phase.eq(phase) & days.both] for phase in PHASES}
    for phase, frame in data.items():
        if len(frame) < MIN_DAYS or week(frame.acq_date).nunique() < MIN_WEEKS or not frame.aqua.sum() > 0:
            raise ValueError(f"Insufficient support in {phase}")
    ratio = {phase: float(q.snpp.sum() / q.aqua.sum()) for phase, q in data.items()}
    mae = {phase: exact_mae_factor(q.aqua, q.snpp) for phase, q in data.items()}
    q = data["all"]
    a_d, a_n = (q[f"aqua_{phase}"].fillna(0).to_numpy() for phase in ("D", "N"))
    target = q.snpp.to_numpy()
    # Fixed grid for k_N/k_D. For every ratio r, the optimal k_D is exact.
    best = (float("inf"), 1., 1.)
    for r in np.exp2(np.linspace(-1, 1, 33)):
        feature = a_d + r * a_n
        k_d = exact_mae_factor(feature, target)
        trial = (np.abs(feature * k_d - target).mean(), k_d, float(r * k_d))
        if trial < best:
            best = trial
    return {
        "identity": ONE,
        "ratio_all": dict.fromkeys(PHASES, ratio["all"]),
        "ratio_dn": ratio,
        "mae_all": dict.fromkeys(PHASES, mae["all"]),
        "mae_dn": mae,
        "mae_joint_dn": {"D": best[1], "N": best[2], "all": mae["all"]},
    }


def forecasts(frame, phase, factors, mode):
    if mode == "split" and phase == "all":
        return pd.concat([frame[f"aqua_{p}"] * factors[p] for p in ("D", "N")],
                         axis=1).sum(axis=1, min_count=1)
    return frame.aqua * factors[phase]


def evaluate(days, models):
    metrics = []
    for phase in PHASES:
        frame = days.loc[days.phase.eq(phase) & days.both]
        by_week = week(frame.acq_date)
        raw_abs_error = (frame.aqua - frame.snpp).abs()
        for name, factors in models.items():
            modes = ("split", "total_only") if phase == "all" else ("phase",)
            for mode in modes:
                pred = forecasts(frame, phase, factors, mode)
                err = pred - frame.snpp
                delta = err.abs() - raw_abs_error
                ci_low, ci_high = bootstrap(delta.groupby(by_week).agg(["sum", "size"]))
                metrics.append(dict(
                    phase=phase, mode=mode, model=name, days=len(frame), weeks=by_week.nunique(),
                    mae_mw=err.abs().mean(), bias_mw=err.mean(), delta_mae_mw=delta.mean(),
                    delta_ci95_low_mw=ci_low, delta_ci95_high_mw=ci_high))
    return pd.DataFrame(metrics)


def select_model(metrics, mode):
    view = metrics.loc[metrics.phase.eq("all") & metrics["mode"].eq(mode)].set_index("model")
    # In total-only mode, the split profiles repeat the two corresponding all factors.
    candidates = CANDIDATES if mode == "split" else ("identity", "ratio_all", "mae_all")
    return min(candidates, key=lambda model: view.loc[model, "mae_mw"])


def confirmed(rows):
    return (not rows.empty and rows.days.ge(MIN_DAYS).all() and rows.weeks.ge(MIN_WEEKS).all()
            and rows.delta_ci95_high_mw.lt(0).all())


def calendar_table(metrics, parameters, winners):
    columns = ["region", "phase", "k", "method", "status", "fit_first_year", "fit_last_year",
               "control_first_year", "control_last_year"]
    rows = [dict(region="_default", phase="all", k=1., method="identity", status="fallback")]
    for region in REGIONS:
        # Cerrado's E9 reported factors were pre-existing benchmarks with 2017–2020 evidence.
        # E10 candidates need independent confirmation; it is available only in the western US.
        options = {}
        observed = metrics.loc[metrics.region.eq(region)]
        if region == "cerrado":
            control = observed.loc[observed.period.eq("regional_control")]
            split = control.loc[control.model.eq("ratio_dn") &
                                (control["mode"].eq("split") | control.phase.ne("all"))]
            total = control.loc[control.model.eq("ratio_all") & control["mode"].eq("total_only")]
            if len(split) == 3 and confirmed(split):
                options.update({p: ("ratio_dn", "validated_e9", 2020) for p in ("D", "N")})
            if len(total) == 1 and confirmed(total):
                options["all"] = ("ratio_all", "validated_e9", 2020)
        if region == "western_us":
            controls = observed.loc[observed.period.isin(["regional_control", "fresh_control"])]
            name = winners[region, "split"]
            split = controls.loc[controls.model.eq(name) &
                                 (controls["mode"].eq("split") | controls.phase.ne("all"))]
            name_total = winners[region, "total_only"]
            total = controls.loc[controls.model.eq(name_total) & controls["mode"].eq("total_only")]
            if len(split) == 6 and confirmed(split):
                options.update({p: (name, "validated_e10", 2025) for p in ("D", "N")})
            if len(total) == 2 and confirmed(total):
                options["all"] = (name_total, "validated_e10", 2025)
        for phase in PHASES:
            row = dict(region=region, phase=phase, k=1., method="identity", status="fallback")
            if phase in options:
                name, status, control_last = options[phase]
                row.update(k=parameters[region][name][phase], method=name, status=status,
                           fit_first_year=2012, fit_last_year=2016, control_first_year=2017,
                           control_last_year=control_last)
            rows.append(row)
    table = pd.DataFrame(rows, columns=columns)
    return table.astype({field: "Int64" for field in columns[5:]})


def run(data, output, calendar):
    history = pd.read_parquet(data / "outputs" / "e9" / "reported_observations.parquet")
    fresh, audit = new_western_days(data)
    metrics, selection, parameters = [], [], {}
    winners = {}
    for region in REGIONS:
        local = history.loc[history.region.eq(region)]
        if set(local.year) != set(range(2012, 2021)):
            raise ValueError(f"Incomplete E9 daily archive for {region}")
        validation_fit = fit(local.loc[local.year.le(2014)])
        validation = evaluate(local.loc[local.year.between(2015, 2016)], validation_fit)
        for mode in ("split", "total_only"):
            winners[region, mode] = select_model(validation, mode)
            selection.append(dict(region=region, mode=mode, selected=winners[region, mode]))
        parameters[region] = fit(local.loc[local.year.le(2016)])
        control = evaluate(local.loc[local.year.between(*REGIONAL_CONTROL)], parameters[region])
        metrics += [validation.assign(region=region, period="validation", eval_first=2015, eval_last=2016),
                    control.assign(region=region, period="regional_control", eval_first=2017, eval_last=2020)]
        if region == "western_us":
            metrics.append(evaluate(fresh, parameters[region]).assign(
                region=region, period="fresh_control", eval_first=2021, eval_last=2025))
            for year, group in fresh.groupby("year"):
                metrics.append(evaluate(group, parameters[region]).assign(
                    region=region, period="fresh_year", eval_first=year, eval_last=year))
    result = pd.concat(metrics, ignore_index=True)
    result["fit_last"] = np.where(result.period.eq("validation"), 2014, 2016)
    assert result.fit_last.lt(result.eval_first).all()
    rows = [dict(region=r, model=m, phase=p, k=k)
            for r, models in parameters.items() for m, factors in models.items() for p, k in factors.items()]
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(selection).to_csv(output / "selection.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "candidate_coefficients.csv", index=False)
    audit.to_csv(output / "western_archive_audit.csv", index=False)
    table = calendar_table(result, parameters, winners)
    if table.duplicated(["region", "phase"]).any() or not table.k.ge(0).all():
        raise ValueError("Invalid calendar lookup keys or factors")
    calendar.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=calendar.parent, delete=False) as temp:
        pending = Path(temp.name)
    try:
        table.to_csv(pending, index=False, lineterminator="\n", float_format="%.12g")
        pending.replace(calendar)
    finally:
        pending.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calendar", type=Path, default=Path(FILENAME))
    args = parser.parse_args()
    run(args.data, args.output or args.data / "outputs" / "e10", args.calendar)


if __name__ == "__main__":
    main()
