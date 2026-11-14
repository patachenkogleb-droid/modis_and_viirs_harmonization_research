pub(crate) mod e19;
pub(crate) mod e20;
pub(crate) mod e20t;
pub(crate) mod verify_frp;
mod input;
mod models;

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{ensure, Result};
use chrono::{Datelike, NaiveDate};
use clap::Args;

use crate::grid::cell;
use input::{Point, REGIONS};

#[derive(Args)]
pub struct HotspotArgs {
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Check the spatial pipeline on one region-year.
    #[arg(long)]
    smoke: bool,
    /// Include western US 2021–2025 after independent manifest checks.
    #[arg(long)]
    western_holdout: bool,
    /// Export frozen product coefficients after the western holdout passes.
    #[arg(long, requires = "western_holdout", conflicts_with = "smoke")]
    freeze: bool,
}

const SIZES: [i32; 3] = [3000, 6000, 15000];
const SCOPES: [&str; 5] = ["region", "SW", "SE", "NW", "NE"];
type Counts = [u64; 4];
type Series = BTreeMap<(usize, usize, NaiveDate), Counts>;
type Cell = (i32, i32, NaiveDate, usize);

pub fn run(args: &HotspotArgs) -> Result<()> {
    ensure!(!args.output.exists(), "Output directory already exists");
    fs::create_dir_all(&args.output)?;
    let mut inputs = csv::Writer::from_path(args.output.join("inputs.csv"))?;
    inputs.write_record(["region", "year", "path", "bytes", "rows_in_year", "platform",
                         "version", "eligible_before_dedup", "unique_count"])?;
    let mut spatial = csv::Writer::from_path(args.output.join("spatial.csv"))?;
    spatial.write_record(["region", "year", "size_m", "active_cell_days", "viirs_positive_cell_days",
        "viirs_only_cell_days", "viirs_in_viirs_only_cell_days", "aqua", "snpp", "matched_passes",
        "matched_aqua", "matched_snpp", "aqua_occupied_cells", "snpp_occupied_cells"])?;
    let mut daily = csv::Writer::from_path(args.output.join("daily.csv"))?;
    daily.write_record(["region", "scope", "date_utc", "aqua_d", "aqua_n", "snpp_d", "snpp_n"])?;
    let mut coverage = csv::Writer::from_path(args.output.join("coverage_sample.csv"))?;
    coverage.write_record(["region", "date_utc", "aqua", "snpp", "status"])?;
    let mut series = Series::new();
    let mut domains = Vec::new();
    for (index, region) in REGIONS.iter().enumerate() {
        for year in 2018..=2025 {
            if (args.smoke && (index != 5 || year != 2018))
                || (index >= 7 && year < 2020)
                || (year > 2020 && (index != 0 || !args.western_holdout))
            {
                continue;
            }
            let points = input::load(&args.data, region, year, &mut inputs)?;
            eprintln!("{} {year}: {} unique detections", region.name, points.len());
            let mut expected = [0; 4];
            for p in &points {
                expected[2 * p.sensor + p.phase] += 1;
                let east = p.lon >= (region.bounds[0] + region.bounds[2]) / 2.0;
                let north = p.lat >= (region.bounds[1] + region.bounds[3]) / 2.0;
                for scope in [0, 1 + usize::from(east) + 2 * usize::from(north)] {
                    series.entry((index, scope, p.date)).or_default()[2 * p.sensor + p.phase] += 1;
                }
            }
            ensure!(aqua(expected) > 0 && viirs(expected) > 0, "Missing platform");
            let first = NaiveDate::from_ymd_opt(year, 1, 1).unwrap();
            let mut quadrant_totals = [0; 4];
            let mut region_totals = [0; 4];
            for date in first.iter_days().take_while(|d| d.year() == year) {
                for (scope, label) in SCOPES.iter().enumerate() {
                    let c = *series.entry((index, scope, date)).or_default();
                    daily.serialize((region.name, label, date.to_string(), c[0], c[1], c[2], c[3]))?;
                    let target = if scope == 0 { &mut region_totals } else { &mut quadrant_totals };
                    for (dst, value) in target.iter_mut().zip(c) {
                        *dst += value;
                    }
                    let sample_day = 1 + ((index as u32 * 17 + date.month() * 11
                        + year as u32 * 7) % 28);
                    if scope == 0 && year == 2020 && date.day() == sample_day {
                        coverage.serialize((region.name, date.to_string(), aqua(c), viirs(c), "unknown"))?;
                    }
                }
            }
            ensure!(region_totals == expected && quadrant_totals == expected, "Daily sum mismatch");
            domains.push((index, year));
            for size in SIZES {
                let dest = args.output.join(format!("cell_{}_{}_{}.csv", region.name, year, size));
                spatial_counts(&points, expected, region.name, year, size, dest, &mut spatial)?;
            }
        }
    }
    inputs.flush()?;
    spatial.flush()?;
    daily.flush()?;
    coverage.flush()?;
    if !args.smoke {
        models::evaluate(&series, &domains, &args.output, args.freeze)?;
    }
    fs::write(args.output.join("COMPLETE"), "E18\n")?;
    Ok(())
}

fn aqua(c: Counts) -> u64 { c[0] + c[1] }
fn viirs(c: Counts) -> u64 { c[2] + c[3] }

fn spatial_counts(
    points: &[Point], expected: Counts, region: &str, year: i32, size: i32,
    dest: PathBuf, spatial: &mut csv::Writer<fs::File>,
) -> Result<()> {
    let mut cells: BTreeMap<Cell, [Vec<u16>; 2]> = BTreeMap::new();
    for p in points {
        let (x, y) = cell(p.lon, p.lat, 3000.0)?;
        let multiple = size / 3000;
        cells.entry((x.div_euclid(multiple), y.div_euclid(multiple), p.date, p.phase))
            .or_default()[p.sensor].push(p.minute);
    }
    let mut writer = csv::Writer::from_path(dest)?;
    writer.write_record(["cell_x", "cell_y", "date_utc", "phase", "aqua", "snpp",
                         "matched_passes", "matched_aqua", "matched_snpp"])?;
    let mut totals = [0; 4];
    let mut matched = [0; 3];
    let mut cell_days = BTreeMap::<_, [u64; 2]>::new();
    let mut occupied = BTreeMap::<_, [bool; 2]>::new();
    for ((x, y, date, phase), mut times) in cells {
        let counts = times.each_ref().map(|v| v.len() as u64);
        let pairs = match_counts(&mut times);
        writer.serialize((x, y, date.to_string(), ["D", "N"][phase], counts[0], counts[1],
                          pairs[0], pairs[1], pairs[2]))?;
        let sum = cell_days.entry((x, y, date)).or_default();
        let present = occupied.entry((x, y)).or_default();
        for sensor in 0..2 {
            totals[2 * sensor + phase] += counts[sensor];
            sum[sensor] += counts[sensor];
            present[sensor] |= counts[sensor] > 0;
        }
        for (sum, count) in matched.iter_mut().zip(pairs) {
            *sum += count;
        }
    }
    writer.flush()?;
    ensure!(totals == expected, "Spatial sum mismatch");
    ensure!(matched[1] <= aqua(expected) && matched[2] <= viirs(expected), "Invalid pairing sums");
    let viirs_days = cell_days.values().filter(|c| c[1] > 0).count();
    let viirs_only = cell_days.values().filter(|c| c[0] == 0 && c[1] > 0);
    spatial.serialize((region, year, size, cell_days.len(), viirs_days, viirs_only.clone().count(),
        viirs_only.map(|c| c[1]).sum::<u64>(), aqua(totals), viirs(totals),
        matched[0], matched[1], matched[2],
        occupied.values().filter(|c| c[0]).count(), occupied.values().filter(|c| c[1]).count()))?;
    Ok(())
}

fn match_counts(times: &mut [Vec<u16>; 2]) -> [u64; 3] {
    if times.iter().any(Vec::is_empty) {
        return [0; 3];
    }
    let passes = times.each_mut().map(|t| {
        t.sort_unstable();
        t.chunk_by(|a, b| b - a <= 10).map(|p| {
            // Twice the median, preserving half-minute timestamps exactly.
            (p[(p.len() - 1) / 2] + p[p.len() / 2], p.len() as u64)
        }).collect::<Vec<_>>()
    });
    let [left, right] = passes;
    let mut candidates = Vec::new();
    for (i, (a, _)) in left.iter().enumerate() {
        for (j, (v, _)) in right.iter().enumerate() {
            let gap = a.abs_diff(*v);
            if gap < 40 {
                candidates.push((gap, i, j));
            }
        }
    }
    candidates.sort_unstable();
    let mut used_left = vec![false; left.len()];
    let mut used_right = vec![false; right.len()];
    let mut result = [0; 3];
    for (_, i, j) in candidates {
        if !used_left[i] && !used_right[j] {
            used_left[i] = true;
            used_right[j] = true;
            result[0] += 1;
            result[1] += left[i].1;
            result[2] += right[j].1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::match_counts;

    #[test]
    fn pairing_is_strict_and_one_to_one() {
        assert_eq!(match_counts(&mut [vec![0], vec![20]]), [0, 0, 0]);
        assert_eq!(match_counts(&mut [vec![10, 0, 34], vec![20]]), [1, 1, 1]);
        assert_eq!(match_counts(&mut [vec![0, 1], vec![20]]), [1, 2, 1]);
    }
}

