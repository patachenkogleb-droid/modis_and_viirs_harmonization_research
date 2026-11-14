use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use chrono::{Datelike, NaiveDate};
use clap::Args;
use serde::Deserialize;

use super::models::{export, periods, train, Factor, Period, BIAS_LIMIT, PHASE_RATIO, POOLED_RATIO};
use super::{aqua, viirs, Counts, Series, REGIONS, SCOPES};

const MODELS: [&str; 3] = ["raw", "ratio_e18", "ratio_phase_e19"];
const SCALES: [&str; 3] = ["day", "week", "month"];
const COMPONENTS: [&str; 3] = ["daylight", "night", "combined"];

#[derive(Args)]
pub struct E19Args {
    /// Completed E18 output directory from a run with --western-holdout.
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Deserialize)]
struct Daily {
    region: String,
    scope: String,
    date_utc: String,
    aqua_d: u64,
    aqua_n: u64,
    snpp_d: u64,
    snpp_n: u64,
}

pub(super) fn domain_years(region: usize) -> std::ops::RangeInclusive<i32> {
    match region {
        0 => 2018..=2025,
        1..=6 => 2018..=2020,
        _ => 2020..=2020,
    }
}

pub(super) fn load_e18_daily(input: &Path) -> Result<Series> {
    ensure!(fs::read_to_string(input.join("COMPLETE"))?.trim() == "E18", "Input is not a completed E18 run");
    let mut reader = csv::Reader::from_path(input.join("daily.csv"))?;
    let mut series = Series::new();
    for record in reader.deserialize::<Daily>() {
        let r = record?;
        let region = REGIONS.iter().position(|region| region.name == r.region)
            .with_context(|| format!("Unknown region {}", r.region))?;
        let scope = SCOPES.iter().position(|scope| *scope == r.scope)
            .with_context(|| format!("Unknown scope {}", r.scope))?;
        let date = NaiveDate::parse_from_str(&r.date_utc, "%Y-%m-%d")?;
        ensure!(domain_years(region).contains(&date.year()), "Unexpected year for {}", r.region);
        let counts = [r.aqua_d, r.aqua_n, r.snpp_d, r.snpp_n];
        ensure!(series.insert((region, scope, date), counts).is_none(), "Duplicate E18 daily record");
    }
    for (r, region) in REGIONS.iter().enumerate() {
        for year in domain_years(r) {
            let start = NaiveDate::from_ymd_opt(year, 1, 1).unwrap();
            for date in start.iter_days().take_while(|d| d.year() == year) {
                let mut scopes = Vec::with_capacity(SCOPES.len());
                for (s, scope) in SCOPES.iter().enumerate() {
                    scopes.push(*series.get(&(r, s, date))
                        .with_context(|| format!("Missing {} {scope} {date}", region.name))?);
                }
                for phase in 0..4 {
                    ensure!(scopes[0][phase] == scopes[1..].iter().map(|c| c[phase]).sum::<u64>(),
                        "E18 quadrant totals differ from region at {} {date}", region.name);
                }
            }
        }
    }
    Ok(series)
}

#[derive(Clone, Copy)]
struct Metrics {
    aqua: u64,
    snpp: u64,
    absolute: f64,
    bias: f64,
}

impl Metrics {
    fn wape(self) -> Option<f64> {
        (self.snpp > 0).then(|| self.absolute / self.snpp as f64)
    }

    fn relative_bias(self) -> Option<f64> {
        (self.snpp > 0).then(|| self.bias / self.snpp as f64)
    }
}

fn components(c: Counts, factors: Factor) -> ([u64; 3], [u64; 3], [f64; 3]) {
    let day = c[0] as f64 * factors[0];
    let night = c[1] as f64 * factors[1];
    ([c[0], c[1], aqua(c)], [c[2], c[3], viirs(c)], [day, night, day + night])
}

fn score(rows: &[Period], factors: Factor) -> [Metrics; 3] {
    let mut metrics = [Metrics { aqua: 0, snpp: 0, absolute: 0.0, bias: 0.0 }; 3];
    for &(_, c) in rows {
        let (a, v, estimate) = components(c, factors);
        for (i, m) in metrics.iter_mut().enumerate() {
            let error = estimate[i] - v[i] as f64;
            m.aqua += a[i];
            m.snpp += v[i];
            m.absolute += error.abs();
            m.bias += error;
        }
    }
    metrics
}

pub(super) fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut values: Vec<_> = values.collect();
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 { (values[mid - 1] + values[mid]) / 2.0 } else { values[mid] }
}

pub fn run(args: &E19Args) -> Result<()> {
    ensure!(!args.output.exists(), "Output directory already exists");
    let series = load_e18_daily(&args.input)?;
    fs::create_dir_all(&args.output)?;
    fs::write(args.output.join("input.txt"), format!("{}\n", args.input.canonicalize()?.join("daily.csv").display()))?;
    let mut coefficients = csv::Writer::from_path(args.output.join("fold_coefficients.csv"))?;
    coefficients.write_record(["excluded_region", "training_years", "training_regions",
        "aqua_day", "aqua_night", "snpp_day", "snpp_night", "k_e18", "k_e19_day", "k_e19_night"])?;
    let mut validation = csv::Writer::from_path(args.output.join("validation.csv"))?;
    validation.write_record(["region", "year", "scope", "scale", "component", "model",
        "periods", "aqua", "snpp", "mae", "wape", "bias_count", "relative_bias"])?;
    let mut monthly = csv::Writer::from_path(args.output.join("monthly_predictions.csv"))?;
    monthly.write_record(["region", "scope", "month_utc", "component", "model", "aqua", "snpp", "estimate"])?;

    let mut controls = Vec::new();
    let mut western = Vec::new();
    let mut west_export = None;
    for (r, region) in REGIONS.iter().enumerate() {
        let (fit, n, counts) = train(&series, r)?;
        let factors = [[1.0; 2], fit[POOLED_RATIO], fit[PHASE_RATIO]];
        coefficients.serialize((region.name, "2018-2019", n, counts[0], counts[1], counts[2],
                                counts[3], factors[1][0], factors[2][0], factors[2][1]))?;
        if r == 0 {
            west_export = Some((factors[2], n, counts));
        }
        for year in domain_years(r).filter(|&year| year >= 2020) {
            for (s, scope) in SCOPES.iter().enumerate() {
                for (scale, scale_name) in SCALES.iter().enumerate() {
                    let rows = periods(&series, r, s, year, scale);
                    let metrics = factors.map(|f| score(&rows, f));
                    if scale == 2 {
                        if s == 0 {
                            if year == 2020 { controls.push(metrics) } else { western.push(metrics) }
                        }
                        for &(date, c) in &rows {
                            for (m, f) in factors.iter().enumerate() {
                                let (a, v, estimate) = components(c, *f);
                                for (p, component) in COMPONENTS.iter().enumerate() {
                                    monthly.serialize((region.name, scope, date.to_string(), component,
                                                        MODELS[m], a[p], v[p], estimate[p]))?;
                                }
                            }
                        }
                    }
                    for (m, scores) in metrics.iter().enumerate() {
                        for (p, score) in scores.iter().enumerate() {
                            validation.serialize((region.name, year, scope, scale_name, COMPONENTS[p],
                                MODELS[m], rows.len(), score.aqua, score.snpp, score.absolute / rows.len() as f64,
                                score.wape(), score.bias, score.relative_bias()))?;
                        }
                    }
                }
            }
        }
    }
    coefficients.flush()?;
    validation.flush()?;
    monthly.flush()?;

    ensure!(controls.len() == 9 && western.len() == 5, "Incomplete E19 validation");
    ensure!(controls.iter().chain(&western).all(|m| m[0][2].snpp > 0), "Missing regional S-NPP counts");
    let combined_wape = |m: &[[Metrics; 3]; 3], model: usize| m[model][2].wape().unwrap();
    let raw_median = median(controls.iter().map(|m| combined_wape(m, 0)));
    let shared_median = median(controls.iter().map(|m| combined_wape(m, 1)));
    let phase_median = median(controls.iter().map(|m| combined_wape(m, 2)));
    let wins = |cohort: &[[[Metrics; 3]; 3]], baseline| {
        cohort.iter().filter(|m| combined_wape(m, 2) < combined_wape(m, baseline)).count()
    };
    let bias_passes = |cohort: &[[[Metrics; 3]; 3]]| {
        cohort.iter().filter(|m| m[2][2].relative_bias().unwrap().abs() <= BIAS_LIMIT).count()
    };
    let discovery = raw_median - phase_median >= 0.05 && wins(&controls, 0) >= 6 && bias_passes(&controls) >= 6;
    let west_pass = wins(&western, 0) > western.len() / 2 && bias_passes(&western) > western.len() / 2;
    let status = if discovery && west_pass { "frozen_e19" } else { "research_e19" };
    let mut decision = csv::Writer::from_path(args.output.join("decision.csv"))?;
    decision.write_record(["raw_median_wape", "e18_median_wape", "e19_median_wape",
        "region_wins_vs_raw", "region_wins_vs_e18", "regions_abs_bias_le_20pct",
        "western_wins_vs_raw", "western_wins_vs_e18", "western_abs_bias_le_20pct",
        "passes_2020_gate", "passes_western_gate", "status"])?;
    decision.serialize((raw_median, shared_median, phase_median, wins(&controls, 0),
        wins(&controls, 1), bias_passes(&controls), wins(&western, 0), wins(&western, 1),
        bias_passes(&western), discovery, west_pass, status))?;
    decision.flush()?;

    let (factor, n, counts) = west_export.expect("Western training fold");
    export(&args.output, PHASE_RATIO, factor, n, counts, status)?;
    fs::write(args.output.join("COMPLETE"), "E19\n")?;
    eprintln!("E19 {status}: k_day={}, k_night={}", factor[0], factor[1]);
    eprintln!("2020 combined WAPE: raw {:.2}%, E18 {:.2}%, E19 {:.2}%",
              100.0 * raw_median, 100.0 * shared_median, 100.0 * phase_median);
    eprintln!("2020 gate {discovery}: wins vs raw {}/9, bias passes {}/9",
              wins(&controls, 0), bias_passes(&controls));
    eprintln!("Western gate {west_pass}: wins vs raw {}/5, bias passes {}/5",
              wins(&western, 0), bias_passes(&western));
    Ok(())
}

