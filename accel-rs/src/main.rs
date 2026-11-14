mod filters;
mod grid;
mod hotspots;
mod pairing;

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use arrow::array::{Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use clap::{Parser, Subcommand};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use filters::{Fields, Fire};

const CHUNK: usize = 100_000;

#[derive(Parser)]
#[command(about = "Stream FIRMS CSV into gridded Parquet or pair aggregated passes", subcommand_negates_reqs = true)]
struct Args {
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 3000)]
    grid_size: u32,
    #[arg(long)]
    max_aqua_scan: Option<f64>,
    #[arg(long, value_delimiter = ',', allow_hyphen_values = true,
          num_args = 4, default_value = "-125,32,-102,49")]
    bbox: Vec<f64>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    Pair(pairing::PairArgs),
    Hotspots(hotspots::HotspotArgs),
    HotspotsE19(hotspots::e19::E19Args),
    HotspotsE20(hotspots::e20::E20Args),
    HotspotsE20t(hotspots::e20t::E20tArgs),
    VerifyFrp(hotspots::verify_frp::VerifyFrpArgs),
}

fn write_batch(writer: &mut ArrowWriter<File>, schema: &Arc<Schema>, fires: &[Fire],
               size: f64) -> Result<()> {
    let cells = grid::assign(fires, size)?;
    let batch = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(Float64Array::from_iter_values(fires.iter().map(|fire| fire.lat))),
        Arc::new(Float64Array::from_iter_values(fires.iter().map(|fire| fire.lon))),
        Arc::new(Float64Array::from_iter_values(fires.iter().map(|fire| fire.frp))),
        Arc::new(Int32Array::from_iter_values(cells.iter().map(|cell| cell.0))),
        Arc::new(Int32Array::from_iter_values(cells.iter().map(|cell| cell.1))),
    ])?;
    writer.write(&batch)?;
    Ok(())
}

fn convert(input: &Path, output: &Path, args: &Args) -> Result<usize> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("latitude", DataType::Float64, false),
        Field::new("longitude", DataType::Float64, false),
        Field::new("frp", DataType::Float64, false),
        Field::new("cell_x", DataType::Int32, false),
        Field::new("cell_y", DataType::Int32, false),
    ]));
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .set_max_row_group_row_count(Some(CHUNK)).build();
    let temporary = output.with_extension("part");
    let result = (|| {
        let mut reader = csv::Reader::from_path(input)?;
        let fields = Fields::new(reader.byte_headers()?)?;
        let mut writer = ArrowWriter::try_new(File::create(&temporary)?, schema.clone(), Some(properties))?;
        let mut record = csv::ByteRecord::new();
        let mut fires = Vec::with_capacity(CHUNK);
        let mut rows = 0;
        while reader.read_byte_record(&mut record)? {
            let fire = fields.parse(&record, args.max_aqua_scan, &args.bbox)
                .with_context(|| format!("CSV record {:?}", record.position()))?;
            if let Some(fire) = fire {
                fires.push(fire);
            }
            if fires.len() == CHUNK {
                write_batch(&mut writer, &schema, &fires, f64::from(args.grid_size))?;
                rows += fires.len();
                fires.clear();
            }
        }
        if !fires.is_empty() {
            write_batch(&mut writer, &schema, &fires, f64::from(args.grid_size))?;
            rows += fires.len();
        }
        writer.close()?;
        fs::rename(&temporary, output)?;
        Ok(rows)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn main() -> Result<()> {
    let args = Args::parse();
    match &args.command {
        Some(Command::Pair(args)) => return pairing::run(args),
        Some(Command::Hotspots(args)) => return hotspots::run(args),
        Some(Command::HotspotsE19(args)) => return hotspots::e19::run(args),
        Some(Command::HotspotsE20(args)) => return hotspots::e20::run(args),
        Some(Command::HotspotsE20t(args)) => return hotspots::e20t::run(args),
        Some(Command::VerifyFrp(args)) => return hotspots::verify_frp::run(args),
        None => (),
    }
    let input = args.input.as_ref().context("--input is required")?;
    let destination = args.output.as_ref().context("--output is required")?;
    ensure!(args.grid_size > 0, "Grid size must be positive");
    ensure!(args.max_aqua_scan.is_none_or(|x| x.is_finite() && x > 0.0), "Invalid scan cutoff");
    ensure!(args.bbox.iter().all(|x| x.is_finite()) &&
            args.bbox[0] <= args.bbox[2] && args.bbox[1] <= args.bbox[3], "Invalid bounding box");
    let mut inputs = if input.is_dir() {
        let mut files = Vec::new();
        for entry in fs::read_dir(input)? {
            let path = entry?.path();
            if path.is_file() && path.extension().and_then(|x| x.to_str())
                .is_some_and(|x| x.eq_ignore_ascii_case("csv")) {
                files.push(path);
            }
        }
        files
    } else {
        vec![input.clone()]
    };
    inputs.sort();
    ensure!(!inputs.is_empty(), "No CSV inputs");
    fs::create_dir_all(destination)?;
    ensure!(fs::read_dir(destination)?.next().is_none(), "Output directory must be empty");
    for (index, input) in inputs.iter().enumerate() {
        let output = destination.join(format!("part-{index:05}.parquet"));
        let rows = convert(input, &output, &args).with_context(|| format!("Input {}", input.display()))?;
        println!("{}: {rows} rows", output.display());
    }
    Ok(())
}
