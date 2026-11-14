"""Calibration candidates, chronological/spatial validation and E9 outputs."""


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.regional_daily_inputs import PHASES, REGIONS, YEARS

LEVELS = {
    "static": (),
    "matched_phase": ("daynight",),
    "phase_frp": ("daynight", "frp_band"),
    "phase_frp_cover": ("daynight", "frp_band", "strong_cover"),
    "phase_frp_lat": ("daynight", "frp_band", "latitude_band"),
}
PARENT = dict(static="raw", matched_phase="static", phase_frp="matched_phase",
              phase_frp_cover="phase_frp", phase_frp_lat="phase_frp",
              report_static="raw", report_phase="report_static", from_pairs="raw", historical_west="raw")
MODELS = {"matched": ("raw", *LEVELS), "reported": ("raw", "report_static", "report_phase")}
MW = ["aqua", "snpp"]
PAIR_MW = {"sum_frp_modis": "aqua", "sum_frp_viirs": "snpp"}
MIN_PAIRS, MIN_WEEKS, MIN_CELLS, MIN_DAYS = 100, 12, 8, 100


def week(dates):
    return dates.dt.to_period("W-SUN").dt.start_time


def bootstrap(blocks):
    """Percentile interval for a ratio of additive quantities, resampling UTC weeks."""
    blocks = np.asarray(blocks, dtype=float)
    if len(blocks) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(229)
    sampled = blocks[rng.integers(0, len(blocks), size=(2000, len(blocks)))].sum(axis=1)
    ratios = sampled[sampled[:, 1] > 0, 0] / sampled[sampled[:, 1] > 0, 1]
    return tuple(np.quantile(ratios, [.025, .975])) if len(ratios) else (np.nan, np.nan)


def fit_table(frame, keys, scope, level, *, pooled, intervals):
    frame = frame.assign(_week=week(frame.acq_date))
    strata = frame.groupby(list(keys), observed=True) if keys else [((), frame)]
    rows = []
    for label, group in strata:
        label = label if isinstance(label, tuple) else (label,)
        weeks = group._week.nunique()
        days = group[["region", "acq_date"]].drop_duplicates().shape[0]
        cells = group[["cell_x", "cell_y"]].drop_duplicates().shape[0] if scope == "matched" else None
        regions = "|".join(sorted(group.region.unique()))
        n_regions = group.region.nunique()
        aqua, snpp = group[MW].sum()
        minimum = len(group) >= MIN_PAIRS and cells >= MIN_CELLS if scope == "matched" else days >= MIN_DAYS
        supported = minimum and weeks >= MIN_WEEKS and aqua > 0 and (not pooled or n_regions >= 2)
        estimate = snpp / aqua if aqua > 0 else np.nan
        lo, hi = (np.nan, np.nan)
        if intervals and aqua > 0:
            lo, hi = bootstrap(group.groupby("_week")[["snpp", "aqua"]].sum())
        rows.append(dict(
            **dict(zip(keys, label)), model=level, scope=scope, k=estimate if supported else np.nan,
            k_unfiltered=estimate, ci95_low=lo, ci95_high=hi, supported=supported,
            support_regions=regions, n_support_regions=n_regions, days=days, week_blocks=weeks,
            pairs=len(group) if scope == "matched" else None, cells=cells,
            sum_aqua_mw=aqua, sum_snpp_mw=snpp, fallback=PARENT[level]))
    return pd.DataFrame(rows)


def fit(pairs, daily, regions, last, intervals):
    p = pairs.loc[pairs.region.isin(regions) & pairs.year.le(last)].rename(columns=PAIR_MW)
    d = daily.loc[daily.region.isin(regions) & daily.year.le(last) & daily.both]
    assert not p.empty and not d.empty and p.year.max() == d.year.max() == last
    settings = dict(pooled=len(regions) > 1, intervals=intervals)
    tables = {level: fit_table(p, keys, "matched", level, **settings) for level, keys in LEVELS.items()}
    tables["report_static"] = fit_table(d.loc[d.phase.eq("all")], (), "reported",
                                       "report_static", **settings)
    tables["report_phase"] = fit_table(d.loc[d.phase.ne("all")].rename(columns={"phase": "daynight"}),
                                      ("daynight",), "reported", "report_phase", **settings)
    return tables


def constant(table):
    k = table.k.iloc[0] if len(table) else np.nan
    return float(k) if pd.notna(k) else 1.


def lookup(frame, table, keys):
    if table.empty:
        return pd.Series(np.nan, index=frame.index)
    return frame.join(table.set_index(list(keys)).k, on=list(keys), validate="many_to_one").k


def matched_predictions(pairs, tables, historical=None):
    points = pairs.rename(columns=PAIR_MW).copy()
    points["raw"] = points.aqua
    points["static"] = points.aqua * constant(tables["static"])
    amounts = []
    for level, keys in LEVELS.items():
        if not keys:
            continue
        k = lookup(points, tables[level], keys)
        points[level] = (points.aqua * k).fillna(points[PARENT[level]])
        amounts.append(fallback_rows(points.assign(fallback=k.isna()), level, "matched"))
    methods = list(MODELS["matched"])
    if historical is not None:
        points["historical_west"] = points.aqua * points.daynight.map(historical)
        methods.append("historical_west")
    daily = []
    for phase in PHASES:
        current = points if phase == "all" else points.loc[points.daynight.eq(phase)]
        sums = current.groupby(["region", "acq_date"], as_index=False)[
            [*MW, *methods]].sum()
        daily.append(sums.assign(phase=phase, both=True, has_aqua=True, has_snpp=True))
    return pd.concat(daily, ignore_index=True), pd.concat(amounts, ignore_index=True)


def fallback_rows(points, model, scope, group_field="daynight"):
    out = []
    for phase in PHASES:
        p = points if phase == "all" else points.loc[points[group_field].eq(phase)]
        back = p.loc[p.fallback]
        out.append(dict(phase=phase, model=model, scope=scope, units=len(p), fallback_units=len(back),
                        aqua_mw=p.aqua.sum(), fallback_aqua_mw=back.aqua.sum()))
    return pd.DataFrame(out)


def reported_predictions(daily, tables):
    d = daily.copy()
    d["raw"] = d.aqua
    reg_k = constant(tables["report_static"])
    d["report_static"] = reg_k * d.aqua
    for model, phase_level, default in (
        ("report_phase", "report_phase", reg_k),
        ("from_pairs", "matched_phase", constant(tables["static"])),
    ):
        scales = tables[phase_level].set_index("daynight").k
        d[model] = d.aqua * d.phase.map(scales).fillna(default)
        all_days = d.phase.eq("all")
        d.loc[all_days, model] = pd.concat(
            [d.loc[all_days, f"aqua_{phase}"] * (scales.get(phase) if pd.notna(scales.get(phase)) else default)
             for phase in ("D", "N")], axis=1).sum(axis=1, min_count=1)
    amounts = []
    for model, level in (("report_phase", "report_phase"), ("from_pairs", "matched_phase")):
        scales = tables[level].set_index("daynight").k
        channels = d.loc[d.phase.ne("all") & d.has_aqua].copy()
        amounts.append(fallback_rows(channels.assign(fallback=channels.phase.map(scales).isna()),
                                     model, "reported", group_field="phase"))
    return d, pd.concat(amounts, ignore_index=True)


def evaluate(daily, models):
    rows = []
    for phase, current in daily.loc[daily.both].groupby("phase", sort=False):
        current = current.assign(_week=week(current.acq_date))
        for method in models:
            errors = current[method] - current.snpp
            abs_errors = errors.abs()
            info = dict(phase=phase, model=method, days=len(current),
                        week_blocks=current._week.nunique(), mae_mw=abs_errors.mean(),
                        bias_mw=errors.mean(), simpler=PARENT.get(method))
            for key, ref in (("raw", "raw"), ("simpler", PARENT.get(method))):
                if ref is None:
                    continue
                delta = abs_errors - (current[ref] - current.snpp).abs()
                lo, hi = bootstrap(delta.groupby(current._week).agg(["sum", "size"]))
                info[f"delta_{key}_mw"] = delta.mean()
                info[f"delta_{key}_ci95_low"] = lo
                info[f"delta_{key}_ci95_high"] = hi
            rows.append(info)
    return pd.DataFrame(rows)


def predict(pairs, reported, tables, historical=None):
    m, mf = matched_predictions(pairs, tables, historical)
    r, rf = reported_predictions(reported, tables)
    return {"matched": m, "reported": r}, pd.concat([mf, rf], ignore_index=True)


def diagnostics(pairs, output):
    paired = pairs.rename(columns=PAIR_MW).assign(
        _week=week(pairs.acq_date),
        scan_band=pd.cut(pairs.scan_aqua, [0, 1.5, 3, np.inf], right=False,
                         labels=["<1.5", "1.5-<3", ">=3"]),
        time_gap_band=pd.cut(pairs.delta_minutes, [0, 15, 30, 60], right=False,
                             labels=["<15", "15-<30", "30-<60"]),
        cover=pairs.strong_cover.fillna("weak_or_unknown"))
    out = []
    for factor in ("latitude_band", "cover", "frp_band", "scan_band", "time_gap_band"):
        group = paired.groupby(["region", "period", "daynight", factor], observed=True)
        table = group.agg(
            aqua_mw=("aqua", "sum"), snpp_mw=("snpp", "sum"), pairs=("aqua", "size"),
            week_blocks=("_week", "nunique"), median_latitude=("latitude", "median"),
            min_latitude=("latitude", "min"), max_latitude=("latitude", "max"),
            median_scan_km=("scan_aqua", "median"), median_gap_minutes=("delta_minutes", "median"),
            median_signed_gap_minutes=("signed_minutes", "median")).reset_index()
        out.append(table.rename(columns={factor: "bin"}).assign(factor=factor))
    pd.concat(out, ignore_index=True).assign(k=lambda t: t.snpp_mw / t.aqua_mw).to_csv(
        output / "factor_diagnostics.csv", index=False)
    cells = pairs.drop_duplicates(["region", "year", "cell_x", "cell_y"])
    mix = cells.assign(cover=cells.strong_cover.fillna("weak_or_unknown")).groupby(
        ["region", "period", "latitude_band", "cover"], observed=True).size()
    mix.rename("cell_years").reset_index().to_csv(output / "cover_latitude_support.csv", index=False)


def coverage_outputs(daily, output):
    d = daily.assign(
        day_status=np.select([daily.both, daily.has_aqua], ["both", "aqua_only"], default="snpp_only"),
        aqua_dn=np.select([daily.has_aqua_D & daily.has_aqua_N, daily.has_aqua_D, daily.has_aqua_N],
                          ["DN", "D", "N"], default="missing"),
        snpp_dn=np.select([daily.has_snpp_D & daily.has_snpp_N, daily.has_snpp_D, daily.has_snpp_N],
                          ["DN", "D", "N"], default="missing"))
    d.groupby(["region", "period", "phase", "day_status"], observed=True).size().rename(
        "days").reset_index().to_csv(output / "detection_days.csv", index=False)
    d.loc[d.phase.eq("all")].groupby(["region", "period", "aqua_dn", "snpp_dn"],
                                    observed=True).size().rename("days").reset_index().to_csv(
        output / "daynight_composition.csv", index=False)
    summary = d.loc[d.both].groupby(["region", "period", "phase"], as_index=False).agg(
        days=("aqua", "size"), aqua_mw=("aqua", "mean"), snpp_mw=("snpp", "mean"),
        matched_difference_mw=("matched_difference_mw", "mean"),
        unpaired_aqua_mw=("aqua_unpaired", "mean"), unpaired_snpp_mw=("snpp_unpaired", "mean"),
        difference_mw=("difference_mw", "mean"))
    summary.to_csv(output / "decomposition_summary.csv", index=False)
    return d


def sources_for(design, target):
    if design == "local":
        return (target,)
    if design == "west_transfer":
        return ("western_us",)
    return tuple(r for r in REGIONS if r != target)


def selection_rows(validation, design, target):
    """Use only source-region validation dates; for LORO, rotate inside the source set."""
    out = []
    for scope, methods in MODELS.items():
        v = validation.loc[validation.scope.eq(scope) & validation.phase.eq("all")]
        base = v.loc[v.model.eq("raw")].set_index("region").mae_mw
        vals = v.assign(relative_mae=v.mae_mw / v.region.map(base)).groupby("model").relative_mae.mean()
        winner = min(methods, key=lambda model: vals[model])
        for model in methods:
            out.append(dict(design=design, target_region=target, scope=scope, model=model,
                            validation_regions="|".join(sorted(base.index)),
                            relative_mae=vals[model], selected=model == winner))
    return out


def pilot_check(west_pairs, reported, pilot, output):
    from experiments.e7_daily_calibration import LAT_EDGES, LAT_LABELS, fit as e7_fit, predict as e7_predict

    # E7/E8 included the few pairs with one zero FRP. Recreate that convention only here.
    p = west_pairs.assign(latitude_band=pd.cut(west_pairs.latitude, LAT_EDGES, labels=LAT_LABELS,
                                               right=False))
    train = p.loc[p.year.le(2016)]
    test = p.loc[p.year.ge(2017)]
    history = reported.loc[reported.region.eq("western_us") & reported.both]
    e7 = pd.read_csv(pilot / "e7_daily_metrics.csv")
    e8 = pd.read_csv(pilot / "e8_daily_coefficients.csv")
    e8_metrics = pd.read_csv(pilot / "e8_daily_metrics.csv")
    prediction, methods = e7_predict(test, e7_fit(train))
    checks = []
    for phase in PHASES:
        a = train if phase == "all" else train.loc[train.daynight.eq(phase)]
        z = prediction if phase == "all" else prediction.loc[prediction.daynight.eq(phase)]
        daily_pairs = z.groupby("acq_date")[["sum_frp_viirs", *methods]].sum()
        for method in methods:
            old = e7.loc[(e7.period == "holdout") & (e7.phase == phase) & (e7.scope == "matched")
                         & (e7.method == method)].iloc[0]
            checks.append((f"E7 matched {phase} {method} MAE", old.mae_mw,
                           (daily_pairs[method] - daily_pairs.sum_frp_viirs).abs().mean()))
            checks.append((f"E7 matched {phase} {method} days", old.days, len(daily_pairs)))
        r = history.loc[history.phase.eq(phase)]
        ref, test_reported = r.loc[r.year.le(2016)], r.loc[r.year.ge(2017)]
        old = e8.loc[e8.phase.eq(phase)].iloc[0]
        k_pairs = a.sum_frp_viirs.sum() / a.sum_frp_modis.sum()
        k_reported = ref.snpp.sum() / ref.aqua.sum()
        checks.extend([
            (f"E8 {phase} matched k", old.k_training_matched, k_pairs),
            (f"E8 {phase} reported k", old.k_training_regional, k_reported),
            (f"E8 {phase} matched difference", old.mean_aqua_minus_snpp_matched_mw,
             test_reported.matched_difference_mw.mean()),
            (f"E8 {phase} unpaired Aqua", old.mean_unpaired_aqua_mw, test_reported.aqua_unpaired.mean()),
            (f"E8 {phase} unpaired S-NPP", old.mean_unpaired_snpp_mw, test_reported.snpp_unpaired.mean()),
        ])
        for model, k in (("raw", 1.), ("matched_multiplier", k_pairs), ("regional_multiplier", k_reported)):
            row = e8_metrics.loc[e8_metrics.phase.eq(phase) & e8_metrics.method.eq(model)].iloc[0]
            checks.append((f"E8 reported {phase} {model} MAE", row.mae_mw,
                           (k * test_reported.aqua - test_reported.snpp).abs().mean()))
    report = pd.DataFrame(checks, columns=["check", "pilot", "e9_pilot_convention"])
    report["difference"] = report.e9_pilot_convention - report.pilot
    report["matches"] = np.isclose(report.e9_pilot_convention, report.pilot, rtol=1e-8, atol=1e-7)
    report.to_csv(output / "pilot_verification.csv", index=False)
    if not report.matches.all():
        raise ValueError("Western pilot check failed; see pilot_verification.csv")


def analyze(output, pilot):
    pairs, reported = [
        pd.concat([pd.read_parquet(output / "cache" / r / str(y) / f"{name}.parquet")
                   for r in REGIONS for y in YEARS], ignore_index=True)
        for name in ("pairs", "reported")]
    if pairs[["region", "cell_x", "cell_y"]].drop_duplicates().duplicated(["cell_x", "cell_y"]).any():
        raise ValueError("Cross-region paired cell overlap")
    west = pairs.loc[pairs.region.eq("western_us")]
    pilot_check(west, reported, pilot, output)
    historical = fit_table(west.loc[west.year.le(2016)].rename(columns=PAIR_MW),
                           ("daynight",), "matched", "historical_west", pooled=False, intervals=True)
    pairs = pairs.loc[pairs.eligible].copy()
    reported = coverage_outputs(reported, output)
    reported.to_parquet(output / "reported_observations.parquet", index=False)
    diagnostics(pairs, output)
    # Validation fits may be reused across targets; exclude the target before fitting or ranking.
    val_cache = {}
    scored, series, selections, coefficients, fallbacks = [], [], [], [], []
    designs = [(design, r) for design in ("local", "west_transfer", "leave_region_out") for r in REGIONS
               if design != "west_transfer" or r != "western_us"]

    def evaluate_split(tables, target, first, last, historic_k=None):
        p = pairs.loc[pairs.region.eq(target) & pairs.year.between(first, last)]
        d = reported.loc[reported.region.eq(target) & reported.year.between(first, last)]
        assert set(d.year.unique()) == set(range(first, last + 1))
        predictions, fallback = predict(p, d, tables, historic_k)
        extras = {"reported": ("from_pairs",),
                  "matched": ("historical_west",) if historic_k is not None else ()}
        scores = pd.concat([
            evaluate(predictions[scope], methods + extras[scope]).assign(region=target, scope=scope)
            for scope, methods in MODELS.items()
        ])
        return scores, predictions, fallback

    for design, target in designs:
        print(f"Evaluating {design} / {target}", flush=True)
        sources = sources_for(design, target)
        if design != "local":
            assert target not in sources
        validation = []
        for inner in sources:
            inner_sources = tuple(r for r in sources if r != inner) if len(sources) > 1 else sources
            key = (inner_sources, inner)
            if key not in val_cache:
                early = fit(pairs, reported, inner_sources, 2014, intervals=False)
                val_cache[key] = evaluate_split(early, inner, 2015, 2016)[0]
            validation.append(val_cache[key].assign(training_regions="|".join(inner_sources)))
        val = pd.concat(validation, ignore_index=True)
        selections += selection_rows(val, design, target)
        scored.append(val.assign(design=design, target_region=target, period="validation",
                                 fit_start=2012, fit_end=2014, eval_start=2015, eval_end=2016))
        final = fit(pairs, reported, sources, 2016, intervals=True)
        historic_k = historical.set_index("daynight").k if design == "west_transfer" else None
        score, prediction, fallback = evaluate_split(final, target, 2017, 2020, historic_k)
        meta = dict(design=design, target_region=target, period="holdout", fit_start=2012, fit_end=2016,
                    eval_start=2017, eval_end=2020, training_regions="|".join(sources))
        scored.append(score.assign(**meta))
        fallbacks.append(fallback.assign(**meta))
        for scope, table in prediction.items():
            series.append(table.assign(**meta, region=target, scope=scope))
        coefficients.extend(table.assign(**meta) for table in final.values())
        if historic_k is not None:
            coefficients.append(historical.assign(**meta))
    scores = pd.concat(scored, ignore_index=True)
    coeff = pd.concat(coefficients, ignore_index=True)
    selected = pd.DataFrame(selections)
    keys = ["design", "target_region", "scope", "model"]
    coeff = coeff.merge(selected[keys + ["selected"]], on=keys, how="left", validate="many_to_one")
    scores = scores.merge(selected[keys + ["selected"]], on=keys, how="left", validate="many_to_one")
    decisions = scores.loc[scores.period.eq("holdout") & scores.selected.eq(True)].copy()
    decisions["validated_vs_raw"] = decisions.delta_raw_ci95_high.lt(0)
    decisions["validated_vs_simpler"] = decisions.delta_simpler_ci95_high.lt(0)
    decisions["worse_than_simpler"] = decisions.delta_simpler_ci95_low.gt(0)
    decisions["status"] = np.select(
        [decisions.model.eq("raw"), ~decisions.validated_vs_raw, decisions.worse_than_simpler,
         decisions.validated_vs_simpler],
        ["retain_raw", "research_only", "worse_than_simpler", "validated_vs_raw_and_simpler"],
        default="validated_vs_raw_only")
    coeff.to_csv(output / "coefficients.csv", index=False)
    selected.to_csv(output / "selection.csv", index=False)
    scores.to_csv(output / "metrics.csv", index=False)
    scores.loc[scores.model.eq("historical_west")].to_csv(
        output / "historical_west_transfer.csv", index=False)
    decisions.to_csv(output / "decisions.csv", index=False)
    decisions.assign(
        application_model=decisions.model.where(
            decisions.validated_vs_raw & ~decisions.worse_than_simpler, "raw"),
        applicability=np.where(decisions.scope.eq("matched"), "known common groups only",
                               "regional FIRMS reported index"),
    ).to_csv(output / "application_rules.csv", index=False)
    daily = pd.concat(series, ignore_index=True)
    daily.to_parquet(output / "daily_series.parquet", index=False)
    fallback = pd.concat(fallbacks, ignore_index=True)
    fallback.assign(fallback_fraction=fallback.fallback_units / fallback.units,
                    fallback_aqua_fraction=fallback.fallback_aqua_mw / fallback.aqua_mw).to_csv(
        output / "fallback_audit.csv", index=False)
    macro_and_plots(scores, coeff, daily, output)
    plot_factor_modifiers(coeff, output)




def macro_and_plots(scores, coeff, daily, output):
    hold = scores.loc[scores.period.eq("holdout")].copy()
    raw = hold.loc[hold.model.eq("raw"), ["design", "region", "phase", "scope", "mae_mw"]].rename(
        columns={"mae_mw": "raw_mae_mw"})
    keys = ["design", "region", "phase", "scope"]
    hold = hold.merge(raw, on=keys, validate="many_to_one")
    hold["delta_pct_raw"] = 100 * (hold.mae_mw / hold.raw_mae_mw - 1)
    daily_blocks = dict(iter(daily.loc[daily.both].groupby(keys, sort=False)))
    macro = []
    for (design, phase, scope), part in hold.groupby(["design", "phase", "scope"]):
        selections = [(name, group) for name, group in part.groupby("model")]
        selections.append(("validation_selected", part.loc[part.selected.eq(True)]))
        for name, group in selections:
            weeks = []
            for row in group.itertuples():
                q = daily_blocks[design, row.region, phase, scope]
                error = (q[row.model] - q.snpp).abs() - (q.raw - q.snpp).abs()
                blocks = error.groupby(week(q.acq_date)).agg(["sum", "size"])
                weeks.append((row.raw_mae_mw, blocks))
            index = sorted(set().union(*(set(blocks.index) for _, blocks in weeks)))
            draws = np.random.default_rng(229).integers(0, len(index), size=(2000, len(index)))
            sampled = []
            for raw_mae, blocks in weeks:
                totals = blocks.reindex(index, fill_value=0).to_numpy()[draws].sum(axis=1)
                sampled.append(100 * totals[:, 0] / totals[:, 1] / raw_mae)
            low, high = np.quantile(np.stack(sampled).mean(axis=0), [.025, .975])
            macro.append(dict(design=design, phase=phase, scope=scope, model=name,
                              n_regions=group.region.nunique(), macro_mae_mw=group.mae_mw.mean(),
                              macro_bias_mw=group.bias_mw.mean(), macro_delta_pct_raw=group.delta_pct_raw.mean(),
                              macro_delta_pct_ci95_low=low, macro_delta_pct_ci95_high=high,
                              macro_delta_mae_mw=(group.mae_mw - group.raw_mae_mw).mean()))
    pd.DataFrame(macro).to_csv(output / "macro_metrics.csv", index=False)

    labels = {"western_us": "Western US", "eastern_us": "Eastern US",
              "southern_amazon": "Southern Amazon", "cerrado": "Cerrado"}
    designs = {"local": ("Local", "#246794"), "west_transfer": ("Western transfer", "#a75320"),
               "leave_region_out": ("Region held out", "#357b5b")}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, layout="constrained")
    winners = hold.loc[hold.selected.eq(True)]
    for col, phase in enumerate(PHASES):
        for i, scope in enumerate(MODELS):
            ax = axes[i, col]
            p = winners.loc[(winners.scope == scope) & (winners.phase == phase)]
            for offset, (design, (label, color)) in zip((-.23, 0, .23), designs.items()):
                q = p.loc[p.design.eq(design)].set_index("region").reindex(list(REGIONS))
                x = np.arange(len(REGIONS)) + offset
                y = q.delta_pct_raw.to_numpy()
                lo = (100 * q.delta_raw_ci95_low / q.raw_mae_mw).to_numpy()
                hi = (100 * q.delta_raw_ci95_high / q.raw_mae_mw).to_numpy()
                ax.vlines(x, lo, hi, color=color, lw=1.2)
                ax.scatter(x, y, label=label, color=color, s=21, zorder=3)
            ax.axhline(0, color="#888888", linewidth=1)
            ax.set_title(f"{scope} · {phase}")
            ax.grid(axis="y", alpha=.2)
            ax.set_xticks(np.arange(len(REGIONS)), labels.values(), rotation=30, ha="right")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Daily FRP backtest, 2017–2020 · Models selected in 2015–2016")
    fig.supylabel("MAE change from raw Aqua (%) · lower is better")
    fig.savefig(output / "regional_errors.png", dpi=160)
    plt.close(fig)

    band_order = ["0–<10", "10–<30", "30–<100", "≥100"]
    f = coeff.loc[coeff.design.eq("local") & coeff.model.eq("phase_frp")]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharey=True, layout="constrained")
    for i, phase in enumerate(("D", "N")):
        for j, (region, label) in enumerate(labels.items()):
            ax = axes[i, j]
            q = f.loc[f.target_region.eq(region) & f.daynight.eq(phase)].set_index("frp_band").reindex(band_order)
            x = np.arange(len(band_order))
            ax.axhline(1, color="#888888", linewidth=1, linestyle=":")
            ax.vlines(x, q.ci95_low, q.ci95_high, color="#246794")
            ax.scatter(x, q.k, color="#246794", s=18, zorder=3)
            ax.scatter(x, q.k_unfiltered.where(~q.supported.fillna(False)), facecolors="none",
                       edgecolors="#a75320", s=25, zorder=3)
            ax.set_title(f"{label} · {phase}", fontsize=10)
            ax.set_xticks(x, band_order, fontsize=8)
            for loc, row in enumerate(q.itertuples()):
                if pd.notna(row.k_unfiltered):
                    right_edge = loc == len(band_order) - 1
                    ax.annotate(f"{row.pairs:,}", (loc, row.k_unfiltered),
                                xytext=(-4 if right_edge else 4, 5), ha="right" if right_edge else "left",
                                textcoords="offset points", fontsize=7)
    fig.suptitle("Matched-group factors, 2012–2016 · Labels are counts of pairs")
    fig.supylabel("FRP ratio k = Σ S-NPP / Σ Aqua")
    fig.supxlabel("Aqua group FRP before calibration (MW) · hollow points lack support")
    fig.savefig(output / "frp_factors.png", dpi=160)
    plt.close(fig)


def plot_factor_modifiers(coeff, output):
    """Show conditional cover/latitude factors relative to the same D/N + FRP parent."""
    local = coeff.loc[coeff.design.eq("local")]
    keys = ["target_region", "daynight", "frp_band"]
    base = local.loc[local.model.eq("phase_frp"), keys + ["k"]].rename(columns={"k": "base_k"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for i, phase in enumerate(("D", "N")):
        for j, (model, factor) in enumerate((
            ("phase_frp_cover", "strong_cover"), ("phase_frp_lat", "latitude_band"),
        )):
            ax = axes[i, j]
            plot = local.loc[local.model.eq(model) & local.daynight.eq(phase)].merge(
                base, on=keys, validate="many_to_one")
            plot = plot.loc[plot.k.notna() & plot.base_k.gt(0)]
            labels = sorted(plot[factor].unique())
            positions = {label: x for x, label in enumerate(labels)}
            for offset, color, region in zip((-.24, -.08, .08, .24),
                                             ("#246794", "#a75320", "#357b5b", "#71468c"), REGIONS):
                part = plot.loc[plot.target_region.eq(region)]
                x = part[factor].map(positions).to_numpy(dtype=float) + offset
                ax.scatter(x, part.k / part.base_k, s=8 + 2 * np.sqrt(part.pairs.to_numpy(dtype=float)),
                           alpha=.55, color=color, label=region.replace("_", " "))
            ax.axhline(1, color="#888888", linestyle=":", linewidth=1)
            ax.set_title(f"{phase} · {'land cover' if j == 0 else 'absolute latitude'}")
            ax.set_xticks(range(len(labels)), [x.replace("_", " ") for x in labels],
                          rotation=26, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=.2)
    axes[0, 1].legend(fontsize=8, loc="best")
    fig.suptitle("Conditional matched factors, 2012–2016 · Marker area increases with pair count")
    fig.supylabel("Conditional k / D/N + Aqua FRP-bin k")
    fig.savefig(output / "factor_modifiers.png", dpi=160)
    plt.close(fig)
