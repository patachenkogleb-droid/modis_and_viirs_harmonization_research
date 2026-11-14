use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use chrono::{Datelike, NaiveDate};
use clap::Args;
use serde::Deserialize;

use super::e19::{domain_years, load_e18_daily, median};
use super::e20::{add_counts, totals, ByBiome, Months, BIOMES};
use super::models::{periods, train, PHASE_RATIO, POOLED_RATIO};
use super::{Series, REGIONS};

const COMPONENTS: [&str; 3] = ["daylight", "night", "combined"];
const MODELS: [&str; 5] = ["raw", "e18_global", "e19_phase", "e20t_matrix", "e20t_fixed_fallback"];
type Matrix = [[f64; 2]; 4];

#[derive(Args)]
pub struct E20tArgs {
    /// Completed E20 output directory with monthly biome counts.
    #[arg(long)]
    input: PathBuf,
    /// Completed E18 output directory for conservation checks and baselines.
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Deserialize)]
struct Row {
    region: String,
    month_utc: String,
    raster_year: i32,
    biome: String,
    aqua_day: u64,
    aqua_night: u64,
    snpp_day: u64,
    snpp_night: u64,
}

fn load(input: &Path, baseline: &Series) -> Result<Months> {
    ensure!(fs::read_to_string(input.join("COMPLETE"))?.trim() == "E20", "Expected a completed E20 input");
    let mut months = Months::new();
    let mut seen = HashSet::new();
    let mut reader = csv::Reader::from_path(input.join("monthly_biome_counts.csv"))?;
    for record in reader.deserialize::<Row>() {
        let r = record?;
        let region = REGIONS.iter().position(|v| v.name == r.region).context("Unknown region")?;
        let biome = BIOMES.iter().position(|&v| v == r.biome).context("Unknown biome")?;
        let date = NaiveDate::parse_from_str(&r.month_utc, "%Y-%m-%d")?;
        let year = date.year();
        ensure!(date.day() == 1 && domain_years(region).contains(&year), "Unexpected month");
        ensure!(r.raster_year == year.min(2020), "Unexpected raster year");
        let month = date.month0() as usize;
        ensure!(seen.insert((region, year, month, biome)), "Duplicate biome month");
        months.entry((region, year)).or_insert([[[0; 4]; 4]; 12])[month][biome] =
            [r.aqua_day, r.aqua_night, r.snpp_day, r.snpp_night];
    }
    for (r, region) in REGIONS.iter().enumerate() {
        for year in domain_years(r) {
            let rows = months.get(&(r, year)).context("Missing region-year")?;
            for (month, (date, counts)) in periods(baseline, r, 0, year, 2).into_iter().enumerate() {
                ensure!((0..4).all(|biome| seen.contains(&(r, year, month, biome))), "Missing biome month");
                ensure!(totals(&rows[month]) == counts, "Counts differ from E18 for {} {date}", region.name);
            }
        }
    }
    Ok(months)
}

fn training_counts(months: &Months, omit: usize) -> ByBiome {
    let mut counts = [[0; 4]; 4];
    for (&(region, year), rows) in months {
        if region != omit && year < 2020 {
            for row in rows {
                add_counts(&mut counts, row);
            }
        }
    }
    counts
}

fn matrix(counts: ByBiome, fallback: [f64; 2]) -> Matrix {
    let mut result = [fallback; 4];
    for (biome, row) in result[..3].iter_mut().enumerate() {
        for (phase, value) in row.iter_mut().enumerate() {
            if counts[biome][phase] > 0 {
                *value = counts[biome][phase + 2] as f64 / counts[biome][phase] as f64;
            }
        }
    }
    result
}

fn predict(row: ByBiome, matrix: Matrix) -> [f64; 3] {
    let phase: [f64; 2] = std::array::from_fn(|p| row.iter().zip(matrix)
        .map(|(counts, k)| counts[p] as f64 * k[p]).sum());
    [phase[0], phase[1], phase[0] + phase[1]]
}

#[derive(Default, Clone, Copy)]
struct Score {
    observed: u64,
    absolute: f64,
    bias: f64,
}

impl Score {
    fn add(&mut self, observed: u64, predicted: f64) {
        let error = predicted - observed as f64;
        self.observed += observed;
        self.absolute += error.abs();
        self.bias += error;
    }

    fn wape(self) -> Option<f64> {
        (self.observed > 0).then(|| self.absolute / self.observed as f64)
    }

    fn relative_bias(self) -> Option<f64> {
        (self.observed > 0).then(|| self.bias / self.observed as f64)
    }
}

type Scores = [[Score; 3]; 5];

fn write_coefficients(
    writer: &mut csv::Writer<fs::File>, region: &str, counts: ByBiome, k: Matrix,
) -> Result<()> {
    for (b, row) in k.iter().enumerate() {
        for (p, factor) in row.iter().enumerate() {
            let fallback = b == 3 || counts[b][p] == 0;
            writer.serialize((region, BIOMES[b], ["D", "N"][p], factor, counts[b][p],
                              counts[b][p + 2], fallback, "2018-2019"))?;
        }
    }
    Ok(())
}

pub fn run(args: &E20tArgs) -> Result<()> {
    ensure!(!args.output.exists(), "Output directory already exists");
    let baseline = load_e18_daily(&args.baseline)?;
    let months = load(&args.input, &baseline)?;
    fs::create_dir_all(&args.output)?;
    fs::write(args.output.join("input.txt"), format!("{}\n{}\n",
        args.input.canonicalize()?.join("monthly_biome_counts.csv").display(),
        args.baseline.canonicalize()?.join("daily.csv").display()))?;

    let mut coefficients = csv::Writer::from_path(args.output.join("fold_coefficients.csv"))?;
    let coefficient_header = ["excluded_region", "biome", "phase", "k", "aqua_training",
                              "snpp_training", "uses_phase_fallback", "training_years"];
    coefficients.write_record(coefficient_header)?;
    let mut validation = csv::Writer::from_path(args.output.join("validation.csv"))?;
    validation.write_record(["region", "year", "component", "model", "observed", "mae",
                              "wape", "bias_count", "relative_bias"])?;
    let mut monthly = csv::Writer::from_path(args.output.join("monthly_predictions.csv"))?;
    monthly.write_record(["region", "month_utc", "component", "model", "observed", "estimate"])?;
    let mut discovery = Vec::new();
    let mut western = Vec::new();
    let global_fallback = train(&baseline, 0)?.0[PHASE_RATIO];
    let mut frozen = None;

    for (r, region) in REGIONS.iter().enumerate() {
        let (fit, _, support) = train(&baseline, r)?;
        let pooled = fit[POOLED_RATIO];
        let phase = fit[PHASE_RATIO];
        let counts = training_counts(&months, r);
        ensure!(totals(&counts) == support, "Training conservation check failed");
        let k = matrix(counts, phase);
        let fixed = matrix(counts, global_fallback);
        write_coefficients(&mut coefficients, region.name, counts, k)?;
        if r == 0 { frozen = Some((counts, k)) }
        for year in domain_years(r).filter(|&year| year >= 2020) {
            let mut scores = [[Score::default(); 3]; 5];
            for (m, &row) in months[&(r, year)].iter().enumerate() {
                let total = totals(&row);
                let observed = [total[2], total[3], total[2] + total[3]];
                let phase_prediction = |factors: [f64; 2]| {
                    let d = total[0] as f64 * factors[0];
                    let n = total[1] as f64 * factors[1];
                    [d, n, d + n]
                };
                let forecasts = [phase_prediction([1.0; 2]), phase_prediction(pooled),
                    phase_prediction(phase), predict(row, k), predict(row, fixed)];
                let date = NaiveDate::from_ymd_opt(year, m as u32 + 1, 1).unwrap();
                for (model, forecast) in forecasts.into_iter().enumerate() {
                    for (p, value) in forecast.into_iter().enumerate() {
                        scores[model][p].add(observed[p], value);
                        monthly.serialize((region.name, date.to_string(), COMPONENTS[p],
                                           MODELS[model], observed[p], value))?;
                    }
                }
            }
            for (m, model) in MODELS.iter().enumerate() {
                for (p, score) in scores[m].iter().enumerate() {
                    validation.serialize((region.name, year, COMPONENTS[p], model, score.observed,
                        score.absolute / 12.0, score.wape(), score.bias, score.relative_bias()))?;
                }
            }
            if year == 2020 { discovery.push(scores) } else { western.push(scores) }
        }
    }
    coefficients.flush()?;
    validation.flush()?;
    monthly.flush()?;
    ensure!(discovery.len() == 9 && western.len() == 5, "Incomplete evaluation cohorts");

    let mut summary = csv::Writer::from_path(args.output.join("summary.csv"))?;
    summary.write_record(["cohort", "component", "model", "defined_regions_or_years",
        "median_wape", "abs_bias_le_20pct", "wape_lt_15pct", "wins_vs_e19"])?;
    for (cohort, rows) in [("regions_2020", &discovery), ("western_2021_2025", &western)] {
        for (m, model) in MODELS.iter().enumerate() {
            for (p, component) in COMPONENTS.iter().enumerate() {
                let values: Vec<_> = rows.iter().filter_map(|s| s[m][p].wape()).collect();
                let wape = (!values.is_empty()).then(|| median(values.iter().copied()));
                let bias_ok = rows.iter().filter(|s| s[m][p].relative_bias().is_some_and(|b| b.abs() <= 0.20)).count();
                let target = values.iter().filter(|&&w| w < 0.15).count();
                let wins = rows.iter().filter(|s| s[m][p].wape().zip(s[2][p].wape()).is_some_and(|(a, b)| a < b)).count();
                summary.serialize((cohort, component, model, values.len(), wape, bias_ok, target, wins))?;
            }
        }
    }
    summary.flush()?;
    let (counts, k) = frozen.context("Missing western training fold")?;
    let mut exported = csv::Writer::from_path(args.output.join("coefficients_biome_diurnal.csv"))?;
    exported.write_record(coefficient_header)?;
    write_coefficients(&mut exported, "western_us", counts, k)?;
    exported.flush()?;
    let [[fd, fnight], [sd, sn], [cd, cn], [fallback_d, fallback_n]] = k;
    fs::write(args.output.join("constants_biome_diurnal.py"), format!(
        "# Auto-generated by Experiment E20t (3x2 Biome-Diurnal Harmonizer).\n\
         # Frozen coefficient snapshot; validation and applicability are reported separately.\n\
         K_FOREST_DAY = {fd}\nK_FOREST_NIGHT = {fnight}\n\
         K_SAVANNA_DAY = {sd}\nK_SAVANNA_NIGHT = {sn}\n\
         K_CROPLAND_DAY = {cd}\nK_CROPLAND_NIGHT = {cn}\n\
         K_FALLBACK_DAY = {fallback_d}\nK_FALLBACK_NIGHT = {fallback_n}\n"
    ))?;
    let wape = |rows: &[Scores], model: usize| median(rows.iter().map(|s| s[model][2].wape().unwrap()));
    let target_pass = discovery.iter().all(|s| s[3][2].wape().is_some_and(|w| w < 0.15));
    let wins = |rows: &[Scores]| rows.iter().filter(|s| s[3][2].wape() < s[2][2].wape()).count();
    let bias_passes = |rows: &[Scores]| rows.iter().filter(|s| s[3][2].relative_bias().unwrap().abs() <= 0.20).count();
    let mut decision = csv::Writer::from_path(args.output.join("decision.csv"))?;
    decision.write_record(["snapshot_status", "target_wape_lt_15pct_all_regions",
        "e19_median_2020_wape", "e20t_median_2020_wape", "regions_improved_vs_e19",
        "regions_abs_bias_le_20pct", "western_years_improved_vs_e19", "western_abs_bias_le_20pct"])?;
    decision.serialize(("frozen_E20t", target_pass, wape(&discovery, 2), wape(&discovery, 3),
                        wins(&discovery), bias_passes(&discovery), wins(&western), bias_passes(&western)))?;
    decision.flush()?;
    fs::write(args.output.join("COMPLETE"), "E20t\n")?;
    eprintln!("E20t frozen_E20t: forest [{fd}, {fnight}], savanna [{sd}, {sn}], cropland [{cd}, {cn}]");
    eprintln!("2020 combined median WAPE: E19 {:.2}%, E20t {:.2}%; target all regions <15%: {target_pass}",
              100.0 * wape(&discovery, 2), 100.0 * wape(&discovery, 3));
    eprintln!("E20t vs E19: regional wins {}/9, western wins {}/5", wins(&discovery), wins(&western));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{matrix, predict};

    #[test]
    fn zero_source_uses_phase_fallback_and_other_always_uses_it() {
        let counts = [[10, 0, 30, 7], [2, 4, 8, 20], [0, 5, 1, 0], [1, 1, 100, 100]];
        let k = matrix(counts, [5.0, 19.0]);
        assert_eq!(k, [[3.0, 19.0], [4.0, 5.0], [5.0, 0.0], [5.0, 19.0]]);
        assert_eq!(predict([[1; 4]; 4], k), [17.0, 43.0, 60.0]);
    }
}

