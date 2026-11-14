use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::PathBuf;

use anyhow::{ensure, Context, Result};
use arrow::array::{Array, BooleanArray, Float64Array, Int64Array, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use chrono::NaiveDate;
use clap::Args;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Deserialize;

use super::input::{self, Point, Region};
use crate::grid::cell;

const E6_DAY: f64 = 0.732069;
const PAIR_WINDOW_TWICE_MINUTES: u16 = 120;
const PHASES: [&str; 2] = ["D", "N"];

#[derive(Args)]
pub struct VerifyFrpArgs {
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    coefficients: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Default, Clone, Copy)]
struct Errors {
    n: u64,
    absolute: f64,
    signed: f64,
}

impl Errors {
    fn add(&mut self, predicted: f64, target: f64) {
        let error = predicted - target;
        self.n += 1;
        self.absolute += error.abs();
        self.signed += error;
    }

    fn mae(self) -> Option<f64> {
        (self.n > 0).then(|| self.absolute / self.n as f64)
    }

    fn bias(self) -> Option<f64> {
        (self.n > 0).then(|| self.signed / self.n as f64)
    }

    fn improvement(self, raw: Self) -> Option<f64> {
        self.mae().zip(raw.mae()).filter(|(_, r)| *r > 0.0).map(|(v, r)| 100.0 * (1.0 - v / r))
    }
}

#[derive(Default, Clone, Copy)]
struct PairScores {
    aqua: f64,
    snpp: f64,
    errors: [Errors; 5],
}

impl PairScores {
    fn add(&mut self, aqua: f64, snpp: f64, phase: usize) {
        let k = if phase == 0 { E6_DAY } else { 1.0 };
        self.aqua += aqua;
        self.snpp += snpp;
        for (metric, (estimate, reference)) in self.errors.iter_mut().zip([
            (snpp, aqua), (k * snpp, aqua), (snpp / k, aqua), (aqua, snpp), (k * aqua, snpp),
        ]) {
            metric.add(estimate, reference);
        }
    }
}

fn aggregate_passes(detections: &mut [(u16, f64)]) -> Vec<(u16, f64)> {
    detections.sort_by_key(|p| p.0);
    detections.chunk_by(|a, b| b.0 - a.0 <= 10).map(|pass| {
        let twice_median = pass[(pass.len() - 1) / 2].0 + pass[pass.len() / 2].0;
        (twice_median, pass.iter().map(|p| p.1).sum())
    }).collect()
}

fn pair(points: Vec<Point>) -> Result<[PairScores; 2]> {
    type Key = (NaiveDate, usize, i32, i32);
    type Detections = [Vec<(u16, f64)>; 2];
    let mut groups = BTreeMap::<Key, Detections>::new();
    for p in points {
        let (x, y) = cell(p.lon, p.lat, 3000.0)?;
        groups.entry((p.date, p.phase, x, y)).or_default()[p.sensor].push((p.minute, p.frp));
    }
    let mut scores = [PairScores::default(); 2];
    for ((_, phase, _, _), mut detections) in groups {
        if detections.iter().any(Vec::is_empty) { continue }
        let [left, right] = detections.each_mut().map(|p| aggregate_passes(p));
        let mut candidates = Vec::new();
        for (i, &(a, _)) in left.iter().enumerate() {
            for (j, &(v, _)) in right.iter().enumerate() {
                let gap = a.abs_diff(v);
                if gap <= PAIR_WINDOW_TWICE_MINUTES {
                    candidates.push((gap, i, j));
                }
            }
        }
        candidates.sort_unstable();
        let mut used = [vec![false; left.len()], vec![false; right.len()]];
        for (_, i, j) in candidates {
            if !used[0][i] && !used[1][j] {
                used[0][i] = true;
                used[1][j] = true;
                scores[phase].add(left[i].1, right[j].1, phase);
            }
        }
    }
    Ok(scores)
}

#[derive(Deserialize)]
struct Coefficient {
    region: String,
    phase: String,
    k: f64,
    status: String,
}

fn verify_calendar(args: &VerifyFrpArgs) -> Result<()> {
    let mut config = csv::Reader::from_path(&args.coefficients)?;
    let coefficients: Vec<_> = config.deserialize::<Coefficient>().collect::<csv::Result<_>>()?;
    let mut rows = BTreeMap::<(String, String), Vec<(i64, f64, f64, Option<f64>, Option<f64>)>>::new();
    let observations = args.data.join("outputs/e9/reported_observations.parquet");
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(observations)?)?.build()?;
    for batch in reader {
        let batch = batch?;
        macro_rules! col {
            ($name:literal, $ty:ty) => { batch.column_by_name($name)
                .and_then(|v| v.as_any().downcast_ref::<$ty>()).context($name)? };
        }
        let region = cast(batch.column_by_name("region").context("Missing region")?, &DataType::Utf8)?;
        let phase = cast(batch.column_by_name("phase").context("Missing phase")?, &DataType::Utf8)?;
        let region = region.as_any().downcast_ref::<StringArray>().context("Invalid region")?;
        let phase = phase.as_any().downcast_ref::<StringArray>().context("Invalid phase")?;
        let year = col!("year", Int64Array);
        let a = col!("aqua", Float64Array);
        let v = col!("snpp", Float64Array);
        let ad = col!("aqua_D", Float64Array);
        let an = col!("aqua_N", Float64Array);
        let both = col!("both", BooleanArray);
        for i in 0..batch.num_rows() {
            if !both.value(i) { continue }
            ensure!(a.is_valid(i) && v.is_valid(i), "E9 common day lacks FRP");
            rows.entry((region.value(i).into(), phase.value(i).into())).or_default().push(
                (year.value(i), a.value(i), v.value(i),
                 ad.is_valid(i).then(|| ad.value(i)), an.is_valid(i).then(|| an.value(i))));
        }
    }
    let mut writer = csv::Writer::from_path(args.output.join("calendar_validation.csv"))?;
    writer.write_record(["region", "phase", "status", "k_file", "k_refit_2012_2016", "days_2017_2020",
        "raw_mae_mw", "calibrated_mae_mw", "raw_bias_mw", "calibrated_bias_mw", "mae_gain_pct"])?;
    for c in &coefficients {
        ensure!(c.k.is_finite() && c.k > 0.0, "Invalid FRP multiplier");
        let Some(data) = rows.get(&(c.region.clone(), c.phase.clone())) else {
            ensure!(c.region == "_default" && c.phase == "all" && c.k == 1.0, "Missing E9 observations");
            continue;
        };
        let mut training = [0.0; 2];
        let mut raw = Errors::default();
        let mut calibrated = Errors::default();
        let mut split = Errors::default();
        let phase_k = PHASES.map(|p| coefficients.iter().find(|v| v.region == c.region
            && v.phase == p && v.status == "validated_e9").map(|v| v.k));
        for &(year, a, v, ad, an) in data {
            if (2012..=2016).contains(&year) {
                training[0] += a;
                training[1] += v;
            } else if (2017..=2020).contains(&year) {
                raw.add(a, v);
                calibrated.add(a * c.k, v);
                if let ("all", [Some(kd), Some(kn)]) = (c.phase.as_str(), phase_k) {
                    split.add(kd * ad.unwrap_or(0.0) + kn * an.unwrap_or(0.0), v);
                }
            }
        }
        let refit = if c.status == "validated_e9" {
            ensure!(training[0] > 0.0, "No E9 training support");
            let k = training[1] / training[0];
            ensure!((k - c.k).abs() < 1e-10, "E9 multiplier mismatch for {} {}", c.region, c.phase);
            Some(k)
        } else {
            ensure!(c.k == 1.0, "Unexpected non-identity fallback");
            None
        };
        writer.serialize((&c.region, &c.phase, &c.status, c.k, refit, raw.n,
            raw.mae(), calibrated.mae(), raw.bias(), calibrated.bias(), calibrated.improvement(raw)))?;
        if split.n > 0 {
            writer.serialize((&c.region, "all_DN", &c.status, None::<f64>, None::<f64>, raw.n,
                raw.mae(), split.mae(), raw.bias(), split.bias(), split.improvement(raw)))?;
        }
    }
    writer.flush()?;
    Ok(())
}

pub fn run(args: &VerifyFrpArgs) -> Result<()> {
    ensure!(!args.output.exists(), "Output directory already exists");
    fs::create_dir_all(&args.output)?;
    verify_calendar(args)?;
    let mut audit = csv::Writer::from_path(args.output.join("inputs.csv"))?;
    audit.write_record(["region", "year", "path", "bytes", "rows_in_year", "platform",
                         "version", "eligible_before_dedup", "unique_count"])?;
    let mut out = csv::Writer::from_path(args.output.join("matched_validation.csv"))?;
    out.write_record(["target", "first_year", "last_year", "phase", "pairs", "sum_aqua_mw",
        "sum_viirs_mw", "empirical_aqua_over_viirs", "model", "k", "mae_mw", "bias_mw", "mae_gain_pct"])?;
    let west = Region { name: "western_us", bounds: [-124.5, 32.5, -114.0, 42.0], parquet: true };
    let amazon = Region { name: "southern_amazon", bounds: [-65.0, -13.0, -51.0, -5.0], parquet: true };
    let cerrado = Region { name: "cerrado", bounds: [-51.0, -18.0, -45.0, -9.0], parquet: true };
    let australia = Region { name: "australia_west", bounds: [140.0, -39.0, 154.0, -28.0], parquet: false };
    for (name, sources, years) in [
        ("us-west", &[&west][..], 2012..=2025),
        ("amazon-partial", &[&amazon, &cerrado][..], 2012..=2020),
        ("australia", &[&australia][..], 2020..=2020),
    ] {
        let mut spans = BTreeMap::<(i32, i32), [PairScores; 2]>::new();
        for year in years.clone() {
            let mut points = Vec::new();
            for region in sources {
                points.extend(input::load(&args.data, region, year, &mut audit)?);
            }
            let pairs = pair(points)?;
            let mut keys = vec![(*years.start(), *years.end())];
            if name == "us-west" && year >= 2021 {
                keys.push((2021, 2025));
            }
            for key in keys {
                for (dst, src) in spans.entry(key).or_default().iter_mut().zip(pairs) {
                    dst.aqua += src.aqua;
                    dst.snpp += src.snpp;
                    for (a, b) in dst.errors.iter_mut().zip(src.errors) {
                        a.n += b.n;
                        a.absolute += b.absolute;
                        a.signed += b.signed;
                    }
                }
            }
        }
        for ((first, last), phases) in spans {
            for (i, score) in phases.into_iter().enumerate() {
                let k = if i == 0 { E6_DAY } else { 1.0 };
                let empirical = (score.snpp > 0.0).then(|| score.aqua / score.snpp);
                for (j, (model, factor, raw)) in [
                    ("viirs_to_aqua_raw", 1.0, 0), ("viirs_to_aqua_as_requested", k, 0),
                    ("viirs_to_aqua_inverse_e6", 1.0 / k, 0),
                    ("aqua_to_viirs_raw", 1.0, 3), ("aqua_to_viirs_e6", k, 3),
                ].into_iter().enumerate() {
                    let value = score.errors[j];
                    out.serialize((name, first, last, PHASES[i], value.n, score.aqua, score.snpp,
                        empirical, model, factor, value.mae(), value.bias(), value.improvement(score.errors[raw])))?;
                }
                eprintln!("{name} {first}-{last} {}: {} pairs", PHASES[i], score.errors[0].n);
            }
        }
    }
    audit.flush()?;
    out.flush()?;
    fs::write(args.output.join("COMPLETE"), "FRP verification\n")?;
    Ok(())
}

