use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use arrow::array::{Array, Date32Array, Float64Array, Int8Array, StringArray, TimestampNanosecondArray};
use chrono::{Datelike, NaiveDate};
use csv::{ByteRecord, Writer};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::filters::Fields;

pub(super) struct Region {
    pub name: &'static str,
    pub bounds: [f64; 4],
    pub(super) parquet: bool,
}

pub(super) const REGIONS: [Region; 9] = [
    Region { name: "western_us", bounds: [-125., 32., -102., 49.], parquet: true },
    Region { name: "eastern_us", bounds: [-88., 32., -76., 43.], parquet: true },
    Region { name: "southern_amazon", bounds: [-65., -13., -51., -3.], parquet: true },
    Region { name: "cerrado", bounds: [-51., -22., -43., -9.], parquet: true },
    Region { name: "canada_boreal", bounds: [-123., 51., -101., 62.], parquet: false },
    Region { name: "romania_carpathians", bounds: [20., 43., 30., 49.], parquet: false },
    Region { name: "indonesia_sumatra", bounds: [94., -6., 107., 6.], parquet: false },
    Region { name: "zambia_savanna", bounds: [21., -19., 34., -8.], parquet: false },
    Region { name: "australia_west", bounds: [113., -36., 127., -24.], parquet: false },
];

pub(super) struct Point {
    pub date: NaiveDate,
    pub minute: u16,
    pub lat: f64,
    pub lon: f64,
    pub sensor: usize,
    pub phase: usize,
    pub frp: f64,
}

type Identity = (NaiveDate, u16, usize, usize, [u64; 5], String, String);

#[derive(Default)]
struct Collector {
    points: Vec<Point>,
    seen: HashSet<Identity>,
    counts: BTreeMap<(usize, String), [usize; 2]>,
}

impl Collector {
    fn insert(&mut self, p: Point, geometry: [f64; 3], confidence: &str, version: &str) {
        let values = [p.lat, p.lon, geometry[0], geometry[1], geometry[2]];
        let numbers = values.map(|v| if v == 0.0 { 0 } else { v.to_bits() });
        let key = (p.date, p.minute, p.sensor, p.phase, numbers, confidence.into(), version.into());
        let counts = self.counts.entry((p.sensor, version.into())).or_default();
        counts[0] += 1;
        if self.seen.insert(key) {
            counts[1] += 1;
            self.points.push(p);
        }
    }
}

pub(super) fn load(data: &Path, region: &Region, year: i32, audit: &mut Writer<File>) -> Result<Vec<Point>> {
    let mut result = Collector::default();
    let sources: Vec<(PathBuf, Option<usize>)> = if region.parquet {
        let relative = if region.name == "western_us" { "firms.parquet".into() }
            else { format!("regions/{}_2012-01-20_2020-12-31.parquet", region.name) };
        vec![(data.join("parquet").join(relative), None)]
    } else {
        ["modis", "viirs-snpp"].iter().enumerate().map(|(sensor, product)| {
            let name = format!("{}_{year}_{product}.csv", region.name);
            (data.join("outputs/e14/raw").join(name), Some(sensor))
        }).collect()
    };
    for (path, expected_sensor) in sources {
        let rows = match expected_sensor {
            Some(sensor) => read_csv(&path, region, year, sensor, &mut result),
            None => read_parquet(&path, region, year, &mut result),
        }.with_context(|| format!("{} {year}", path.display()))?;
        ensure!(!result.counts.is_empty(), "No eligible records in {}", path.display());
        let bytes = path.metadata()?.len();
        for ((sensor, version), [eligible, unique]) in std::mem::take(&mut result.counts) {
            audit.serialize((region.name, year, path.display().to_string(), bytes, rows,
                             ["aqua", "snpp"][sensor], version, eligible, unique))?;
        }
    }
    Ok(result.points)
}

fn platform(value: &str) -> Result<Option<usize>> {
    match value.trim().to_ascii_uppercase().as_str() {
        "A" | "AQUA" => Ok(Some(0)),
        "N" | "NPP" | "SNPP" | "S-NPP" => Ok(Some(1)),
        "T" | "TERRA" => Ok(None),
        _ => bail!("Unexpected platform {value}"),
    }
}

fn phase(value: &str) -> Result<usize> {
    match value.trim().to_ascii_uppercase().as_str() {
        "D" => Ok(0),
        "N" => Ok(1),
        _ => bail!("Invalid phase {value}"),
    }
}

fn minute(value: &str) -> Result<u16> {
    let s = value.trim();
    ensure!(!s.is_empty() && s.len() <= 4 && s.bytes().all(|b| b.is_ascii_digit()), "Invalid HHMM");
    let value: u16 = s.parse()?;
    ensure!(value / 100 < 24 && value % 100 < 60, "Invalid UTC time");
    Ok(value / 100 * 60 + value % 100)
}

fn version_check(version: &str, sensor: usize) -> Result<()> {
    let allowed = if sensor == 0 { &["6.03", "61.03"][..] } else { &["2"][..] };
    ensure!(allowed.contains(&version), "Unexpected product version {version} for platform {sensor}");
    Ok(())
}

fn inside(region: &Region, lon: f64, lat: f64) -> bool {
    let [w, s, e, n] = region.bounds;
    (w..=e).contains(&lon) && (s..=n).contains(&lat)
        && !(region.name == "southern_amazon" && lon == e)
}

fn read_csv(path: &Path, region: &Region, year: i32, expected_sensor: usize,
            result: &mut Collector) -> Result<usize> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.byte_headers()?.clone();
    let fields = Fields::new(&headers)?;
    let column = |name: &str| headers.iter().position(|s| s == name.as_bytes())
        .with_context(|| format!("Missing CSV column {name}"));
    let [sat, instrument, date, time, dn, ver, scan, track, conf] =
        ["satellite", "instrument", "acq_date", "acq_time", "daynight", "version", "scan", "track", "confidence"]
            .map(column).into_iter().collect::<Result<Vec<_>>>()?.try_into().unwrap();
    let mut rows = 0;
    let mut record = ByteRecord::new();
    while reader.read_byte_record(&mut record)? {
        rows += 1;
        let value = |i: usize| -> Result<&str> { Ok(std::str::from_utf8(&record[i])?.trim()) };
        let Some(sensor) = platform(value(sat)?)? else { continue };
        ensure!(sensor == expected_sensor, "Platform differs from archive product");
        ensure!(value(instrument)? == ["MODIS", "VIIRS"][sensor], "Instrument differs from archive product");
        let date = NaiveDate::parse_from_str(value(date)?, "%Y-%m-%d")?;
        ensure!(date.year() == year, "Record lies outside archive year");
        let version = value(ver)?;
        version_check(version, sensor)?;
        let Some(fire) = fields.parse(&record, None, &region.bounds)? else { continue };
        if inside(region, fire.lon, fire.lat) {
            let p = Point { date, minute: minute(value(time)?)?, lat: fire.lat, lon: fire.lon,
                            sensor, phase: phase(value(dn)?)?, frp: fire.frp };
            result.insert(p, [fire.frp, value(scan)?.parse()?, value(track)?.parse()?], value(conf)?, version);
        }
    }
    Ok(rows)
}

fn read_parquet(path: &Path, region: &Region, year: i32, result: &mut Collector) -> Result<usize> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?.build()?;
    let mut rows = 0;
    for batch in reader {
        let b = batch?;
        macro_rules! column {
            ($name:literal, $ty:ty) => {
                b.column_by_name($name).and_then(|a| a.as_any().downcast_ref::<$ty>())
                    .with_context(|| format!("Invalid Parquet column {}", $name))?
            };
        }
        let dates = column!("acq_date", Date32Array);
        let sensor = column!("sensor", StringArray);
        let versions = column!("source_version", StringArray);
        let time = column!("acq_time", StringArray);
        let timestamp = column!("timestamp", TimestampNanosecondArray);
        let dn = column!("daynight", StringArray);
        let kind = column!("type", Int8Array);
        let lat = column!("latitude", Float64Array);
        let lon = column!("longitude", Float64Array);
        let frp = column!("frp", Float64Array);
        let scan = column!("scan", Float64Array);
        let track = column!("track", Float64Array);
        let conf = column!("confidence", StringArray);
        for i in 0..b.num_rows() {
            ensure!(dates.is_valid(i), "Missing date");
            let date = dates.value_as_date(i).context("Invalid Parquet date")?;
            if date.year() != year { continue }
            rows += 1;
            ensure!(b.columns().iter().all(|a| a.is_valid(i)), "Null Parquet field");
            let Some(sensor) = platform(sensor.value(i))? else { continue };
            version_check(versions.value(i), sensor)?;
            let typ = kind.value(i);
            ensure!((0..=3).contains(&typ), "Invalid hotspot type");
            let m = minute(time.value(i))?;
            let epoch_minutes = i64::from(dates.value(i)) * 1440 + i64::from(m);
            ensure!(timestamp.value(i) == epoch_minutes * 60_000_000_000, "UTC timestamp mismatch");
            let phase = phase(dn.value(i))?;
            let (lon, lat) = (lon.value(i), lat.value(i));
            let geometry = [frp.value(i), scan.value(i), track.value(i)];
            ensure!(lon.is_finite() && lat.is_finite() && geometry.iter().all(|v| v.is_finite())
                    && geometry[0] >= 0. && geometry[1] > 0. && geometry[2] > 0., "Invalid Parquet point");
            if typ == 0 && inside(region, lon, lat) {
                result.insert(Point { date, minute: m, lon, lat, sensor, phase, frp: geometry[0] },
                              geometry, conf.value(i), versions.value(i));
            }
        }
    }
    Ok(rows)
}

