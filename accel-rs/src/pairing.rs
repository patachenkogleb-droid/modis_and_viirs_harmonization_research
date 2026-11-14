use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Result};
use clap::Args;
use serde::Deserialize;

#[derive(Args)]
pub struct PairArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 60)]
    max_minutes: u32,
}

#[derive(Deserialize)]
struct Pass {
    cell_x: i32,
    cell_y: i32,
    acq_date: String,
    daynight: String,
    sensor: String,
    pass_number: u32,
    time_epoch_ms: i64,
    sum_frp: f64,
    rows: u64,
}

type Key = (i32, i32, String, String);
type SensorPasses = (Vec<Pass>, Vec<Pass>);

pub fn run(args: &PairArgs) -> Result<()> {
    ensure!(args.max_minutes > 0, "Time window must be positive");
    ensure!(!args.output.exists(), "Output file already exists");
    let mut groups: BTreeMap<Key, SensorPasses> = BTreeMap::new();
    let mut reader = csv::Reader::from_path(&args.input)?;
    for record in reader.deserialize::<Pass>() {
        let pass = record?;
        ensure!(pass.daynight == "D" || pass.daynight == "N", "Invalid phase");
        ensure!(pass.sum_frp.is_finite() && pass.sum_frp >= 0.0 && pass.rows > 0, "Invalid pass");
        let group = groups.entry((pass.cell_x, pass.cell_y, pass.acq_date.clone(), pass.daynight.clone()))
            .or_default();
        match pass.sensor.as_str() {
            "aqua" => group.0.push(pass),
            "snpp" => group.1.push(pass),
            _ => bail!("Unexpected sensor in passes CSV"),
        }
    }
    write_matches(&args.output, groups, u64::from(args.max_minutes) * 60_000)
}

fn write_matches(path: &Path, groups: BTreeMap<Key, SensorPasses>, limit_ms: u64) -> Result<()> {
    let temporary = path.with_extension("part");
    let result = (|| {
        let mut writer = csv::Writer::from_path(&temporary)?;
        writer.write_record([
            "cell_x", "cell_y", "acq_date", "daynight", "time_aqua_epoch_ms",
            "time_snpp_epoch_ms", "sum_frp_modis", "sum_frp_viirs", "rows_aqua", "rows_snpp",
            "delta_minutes", "signed_minutes",
        ])?;
        for ((x, y, date, phase), (aqua, snpp)) in groups {
            let mut candidates = Vec::new();
            for left in &aqua {
                for right in &snpp {
                    let delta = left.time_epoch_ms.abs_diff(right.time_epoch_ms);
                    if delta < limit_ms {
                        candidates.push((delta, left, right));
                    }
                }
            }
            candidates.sort_by_key(|(delta, left, right)| {
                (*delta, left.time_epoch_ms, right.time_epoch_ms)
            });
            let mut used_aqua = HashSet::new();
            let mut used_snpp = HashSet::new();
            for (delta, left, right) in candidates {
                if used_aqua.contains(&left.pass_number) || used_snpp.contains(&right.pass_number) {
                    continue;
                }
                used_aqua.insert(left.pass_number);
                used_snpp.insert(right.pass_number);
                writer.serialize((
                    x, y, &date, &phase, left.time_epoch_ms, right.time_epoch_ms,
                    left.sum_frp, right.sum_frp, left.rows, right.rows,
                    delta as f64 / 60_000.0,
                    (right.time_epoch_ms - left.time_epoch_ms) as f64 / 60_000.0,
                ))?;
            }
        }
        writer.flush()?;
        drop(writer);
        std::fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}
