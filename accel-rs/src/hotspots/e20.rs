use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use chrono::{Datelike, NaiveDate};
use clap::Args;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;
use tiff::ColorType;

use super::e19::{domain_years, load_e18_daily, median};
use super::input::{self, Region};
use super::models::{periods, train, PHASE_RATIO, POOLED_RATIO};
use super::{aqua, viirs, Counts, REGIONS};

pub(super) const BIOMES: [&str; 4] = ["forest", "savanna_shrub", "cropland", "other_unknown"];
const MODELS: [&str; 4] = ["e18_global", "e19_phase", "e20_biome", "e20_fixed_fallback"];
pub(super) type ByBiome = [Counts; 4];
pub(super) type Months = BTreeMap<(usize, i32), [ByBiome; 12]>;

#[derive(Args)]
pub struct E20Args {
    /// Original FIRMS and land-cover data directory.
    #[arg(long)]
    data: PathBuf,
    /// Completed E18 run with the western holdout, used to verify count conservation.
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

struct Cover {
    width: usize,
    height: usize,
    west: f64,
    north: f64,
    dx: f64,
    dy: f64,
    pixels: Vec<u8>,
}

impl Cover {
    fn open(path: &Path, bounds: [f64; 4]) -> Result<Self> {
        let mut reader = Decoder::new(BufReader::new(File::open(path)?))?;
        let (width, height) = reader.dimensions()?;
        ensure!(reader.colortype()? == ColorType::Gray(8), "Expected an 8-bit, single-band GeoTIFF");
        let scale = reader.get_tag_f64_vec(Tag::ModelPixelScaleTag)?;
        let tie = reader.get_tag_f64_vec(Tag::ModelTiepointTag)?;
        let keys = reader.get_tag_u16_vec(Tag::GeoKeyDirectoryTag)?;
        ensure!(keys.len() >= 4 && keys.len() == 4 + 4 * usize::from(keys[3]), "Malformed GeoKey directory");
        let has_key = |key, value| keys[4..].chunks_exact(4).any(|e| e == [key, 0, 1, value]);
        ensure!(has_key(2048, 4326) && has_key(1025, 1), "Expected EPSG:4326 and PixelIsArea");
        ensure!(scale.len() == 3 && tie.len() == 6 && scale.iter().chain(&tie).all(|v| v.is_finite())
            && scale[0] > 0.0 && scale[1] > 0.0, "Invalid north-up affine transform");
        let west = tie[3] - tie[0] * scale[0];
        let north = tie[4] + tie[1] * scale[1];
        let [w, s, e, n] = bounds;
        ensure!(west <= w && north >= n && west + f64::from(width) * scale[0] > e
            && north - f64::from(height) * scale[1] < s, "Land-cover clip does not cover region");
        let DecodingResult::U8(pixels) = reader.read_image()? else {
            anyhow::bail!("Expected uint8 raster data")
        };
        ensure!(pixels.len() == width as usize * height as usize, "Incomplete raster band");
        Ok(Self { width: width as usize, height: height as usize, west, north,
                  dx: scale[0], dy: scale[1], pixels })
    }

    fn sample(&self, lon: f64, lat: f64) -> Result<u8> {
        let col = ((lon - self.west) / self.dx).floor();
        let row = ((self.north - lat) / self.dy).floor();
        ensure!(col >= 0.0 && col < self.width as f64 && row >= 0.0 && row < self.height as f64,
            "Detection outside validated raster extent: {lon}, {lat}");
        Ok(self.pixels[row as usize * self.width + col as usize])
    }
}

fn biome(code: u8) -> usize {
    match code {
        1..=5 => 0,
        6..=10 => 1,
        12 | 14 => 2,
        _ => 3,
    }
}

fn cover_path(data: &Path, region: &Region, index: usize, year: i32) -> PathBuf {
    let dir = match index {
        0 => data.join("landcover"),
        1..=3 => data.join("landcover").join(region.name),
        _ => data.join("outputs/e14/landcover").join(region.name),
    };
    dir.join(format!("LC_Type1_{year}.tif"))
}

pub(super) fn totals(month: &ByBiome) -> Counts {
    std::array::from_fn(|phase| month.iter().map(|c| c[phase]).sum())
}

pub(super) fn add_counts(target: &mut ByBiome, values: &ByBiome) {
    for (dst, src) in target.iter_mut().zip(values) {
        for (d, s) in dst.iter_mut().zip(src) {
            *d += s;
        }
    }
}

fn fit(months: &Months, omit: usize) -> Result<([f64; 3], ByBiome)> {
    let mut counts = [[0; 4]; 4];
    for (&(region, year), rows) in months {
        if region != omit && year < 2020 {
            for row in rows {
                add_counts(&mut counts, row);
            }
        }
    }
    let mut factors = [0.0; 3];
    for (i, factor) in factors.iter_mut().enumerate() {
        ensure!(aqua(counts[i]) > 0 && viirs(counts[i]) > 0, "Insufficient support for {}", BIOMES[i]);
        *factor = viirs(counts[i]) as f64 / aqua(counts[i]) as f64;
    }
    Ok((factors, counts))
}

#[derive(Default, Clone, Copy)]
struct Score {
    observed: u64,
    abs_error: f64,
    bias: f64,
}

impl Score {
    fn add(&mut self, observed: u64, predicted: f64) {
        let error = predicted - observed as f64;
        self.observed += observed;
        self.abs_error += error.abs();
        self.bias += error;
    }

    fn wape(self) -> f64 {
        self.abs_error / self.observed as f64
    }

    fn relative_bias(self) -> f64 {
        self.bias / self.observed as f64
    }
}

pub fn run(args: &E20Args) -> Result<()> {
    ensure!(!args.output.exists(), "Output directory already exists");
    let baseline = load_e18_daily(&args.baseline)?;
    fs::create_dir_all(&args.output)?;
    let mut inventory = csv::Writer::from_path(args.output.join("inputs.csv"))?;
    inventory.write_record(["region", "year", "path", "bytes", "rows_in_year", "platform",
                         "version", "eligible_before_dedup", "unique_count"])?;
    let mut rasters = csv::Writer::from_path(args.output.join("rasters.csv"))?;
    rasters.write_record(["region", "detection_year", "raster_year", "path", "width", "height"])?;
    let mut monthly = csv::Writer::from_path(args.output.join("monthly_biome_counts.csv"))?;
    monthly.write_record(["region", "month_utc", "raster_year", "biome", "aqua_day", "aqua_night", "snpp_day", "snpp_night"])?;
    let mut codes = csv::Writer::from_path(args.output.join("igbp_counts.csv"))?;
    codes.write_record(["region", "detection_year", "igbp", "biome", "aqua", "snpp"])?;

    let mut months = Months::new();
    for (r, region) in REGIONS.iter().enumerate() {
        let mut cached = None;
        for year in domain_years(r) {
            let cover_year = year.min(2020);
            let path = cover_path(&args.data, region, r, cover_year);
            if cached.as_ref().is_none_or(|(y, _)| *y != cover_year) {
                cached = Some((cover_year, Cover::open(&path, region.bounds)
                    .with_context(|| format!("Reading {}", path.display()))?));
            }
            let cover = &cached.as_ref().unwrap().1;
            rasters.serialize((region.name, year, cover_year, path.display().to_string(), cover.width, cover.height))?;
            let points = input::load(&args.data, region, year, &mut inventory)?;
            let mut rows = [[[0; 4]; 4]; 12];
            let mut class_counts = [[0_u64; 2]; 256];
            for p in &points {
                let code = cover.sample(p.lon, p.lat)?;
                class_counts[usize::from(code)][p.sensor] += 1;
                rows[p.date.month0() as usize][biome(code)][2 * p.sensor + p.phase] += 1;
            }
            for (code, c) in class_counts.into_iter().enumerate().filter(|(_, c)| c != &[0, 0]) {
                codes.serialize((region.name, year, code, BIOMES[biome(code as u8)], c[0], c[1]))?;
            }
            let expected = periods(&baseline, r, 0, year, 2);
            ensure!(expected.len() == 12, "Incomplete E18 baseline");
            for ((date, expected), row) in expected.into_iter().zip(&rows) {
                ensure!(totals(row) == expected, "Counts differ from E18 at {} {date}", region.name);
                for (b, c) in row.iter().enumerate() {
                    monthly.serialize((region.name, date.to_string(), cover_year, BIOMES[b],
                                       c[0], c[1], c[2], c[3]))?;
                }
            }
            months.insert((r, year), rows);
            eprintln!("{} {year}: classified {} detections with {cover_year} LC_Type1", region.name, points.len());
        }
    }
    inventory.flush()?;
    rasters.flush()?;
    monthly.flush()?;
    codes.flush()?;
    evaluate(&baseline, &months, &args.output)?;
    fs::write(args.output.join("COMPLETE"), "E20\n")?;
    Ok(())
}

fn evaluate(baseline: &super::Series, months: &Months, out: &Path) -> Result<()> {
    let mut factors_out = csv::Writer::from_path(out.join("fold_coefficients.csv"))?;
    factors_out.write_record(["excluded_region", "biome", "k", "aqua_training", "snpp_training", "training_regions"])?;
    let mut validation = csv::Writer::from_path(out.join("validation.csv"))?;
    validation.write_record(["region", "year", "raster_year", "model", "observed", "wape", "relative_bias", "mae"])?;
    let mut predictions = csv::Writer::from_path(out.join("monthly_predictions.csv"))?;
    predictions.write_record(["region", "month_utc", "model", "observed", "estimate"])?;

    let (western_base, _, _) = train(baseline, 0)?;
    let fixed_fallback = western_base[POOLED_RATIO][0];
    let mut control = Vec::new();
    let mut western = Vec::new();
    for (r, region) in REGIONS.iter().enumerate() {
        let (fit_baseline, training_regions, support) = train(baseline, r)?;
        let pooled = fit_baseline[POOLED_RATIO][0];
        let phase = fit_baseline[PHASE_RATIO];
        let (k, counts) = fit(months, r)?;
        ensure!(totals(&counts) == support, "Biome training counts differ from E18");
        for (b, factor) in k.into_iter().chain([pooled]).enumerate() {
            factors_out.serialize((region.name, BIOMES[b], factor, aqua(counts[b]),
                                   viirs(counts[b]), training_regions))?;
        }

        for year in domain_years(r).filter(|&year| year >= 2020) {
            let mut scores = [Score::default(); 4];
            for (month, row) in months[&(r, year)].iter().enumerate() {
                let c = totals(row);
                let targeted: f64 = row[..3].iter().zip(k).map(|(&counts, k)| aqua(counts) as f64 * k).sum();
                let other = aqua(row[3]) as f64;
                let values = [c[0] as f64 * pooled + c[1] as f64 * pooled,
                    c[0] as f64 * phase[0] + c[1] as f64 * phase[1],
                    targeted + other * pooled, targeted + other * fixed_fallback];
                let date = NaiveDate::from_ymd_opt(year, month as u32 + 1, 1).unwrap();
                for (m, estimate) in values.into_iter().enumerate() {
                    scores[m].add(viirs(c), estimate);
                    predictions.serialize((region.name, date.to_string(), MODELS[m], viirs(c), estimate))?;
                }
            }
            ensure!(scores[0].observed > 0, "Missing regional S-NPP observations");
            for (m, score) in scores.iter().enumerate() {
                validation.serialize((region.name, year, year.min(2020), MODELS[m], score.observed,
                    score.wape(), score.relative_bias(), score.abs_error / 12.0))?;
            }
            if year == 2020 { control.push(scores) } else { western.push(scores) }
        }
    }
    factors_out.flush()?;
    validation.flush()?;
    predictions.flush()?;

    let (k, _) = fit(months, 0)?;
    let wape = |rows: &[[Score; 4]], m: usize| median(rows.iter().map(|scores| scores[m].wape()));
    let better = |rows: &[[Score; 4]], reference: usize| {
        rows.iter().filter(|scores| scores[2].wape() < scores[reference].wape()).count()
    };
    let east_west_ok = control[..2].iter().all(|scores| scores[2].relative_bias().abs() < 0.20);
    let passed = wape(&control, 2) < wape(&control, 0) && better(&control, 0) >= 6
        && east_west_ok && better(&western, 0) >= 3;
    let status = if passed { "passes_e20_gate" } else { "research_e20" };
    let mut decision = csv::Writer::from_path(out.join("decision.csv"))?;
    decision.write_record(["e18_2020_wape", "e19_2020_wape", "e20_2020_wape", "e20_fixed_2020_wape",
        "region_wins_vs_e18", "region_wins_vs_e19", "west_year_wins_vs_e18",
        "west_year_wins_vs_e19", "both_us_regions_abs_bias_lt_20pct", "status"])?;
    decision.serialize((wape(&control, 0), wape(&control, 1), wape(&control, 2), wape(&control, 3),
        better(&control, 0), better(&control, 1), better(&western, 0), better(&western, 1), east_west_ok, status))?;
    decision.flush()?;

    let [forest, savanna, cropland] = k;
    fs::write(out.join("constants_biome.py"), format!(
        "# Auto-generated by fire-prep hotspots-e20 ({status}).\n\
         K_HOTSPOTS_FOREST = {forest}\nK_HOTSPOTS_SAVANNA = {savanna}\n\
         K_HOTSPOTS_CROPLAND = {cropland}\nK_HOTSPOTS_GLOBAL_FALLBACK = {fixed_fallback}\n"
    ))?;
    eprintln!("E20 {status}: forest={forest}, savanna={savanna}, cropland={cropland}, fallback={fixed_fallback}");
    eprintln!("2020 monthly WAPE: E18 {:.2}%, E19 {:.2}%, E20 {:.2}%",
              100.0 * wape(&control, 0), 100.0 * wape(&control, 1), 100.0 * wape(&control, 2));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{biome, Cover};

    #[test]
    fn igbp_mapping_includes_water_and_missing_in_fallback() {
        let expected = [3, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 3, 2, 3, 2, 3, 3, 3];
        for (code, group) in expected.into_iter().enumerate() {
            assert_eq!(biome(code as u8), group);
        }
        assert_eq!(biome(255), 3);
    }

    #[test]
    fn point_sampling_uses_pixel_areas_and_rejects_outside() {
        let cover = Cover { width: 2, height: 2, west: -2.0, north: 2.0,
                            dx: 1.0, dy: 1.0, pixels: vec![1, 12, 17, 255] };
        assert_eq!(cover.sample(-1.5, 1.5).unwrap(), 1);
        assert_eq!(cover.sample(-1.0, 1.0).unwrap(), 255);
        assert_eq!(cover.sample(-1.5, 0.5).unwrap(), 17);
        assert!(cover.sample(0.0, 1.0).is_err());
        assert!(cover.sample(-2.5, 1.0).is_err());
    }
}

