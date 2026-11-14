from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
import argparse
import json
import re
from threading import Lock
from time import monotonic, sleep
from zipfile import ZipFile

from dotenv import dotenv_values
import numpy as np
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.benchmark import measure, write_json

BBOX = (-125, 32, -102, 49)
SOURCES = ("MODIS_SP", "VIIRS_SNPP_SP")
RAW_COLUMNS = ("latitude", "longitude", "frp", "scan", "track", "acq_date",
               "acq_time", "confidence", "type", "daynight", "satellite")
SCHEMA = pa.schema([
    ("latitude", pa.float64()), ("longitude", pa.float64()), ("frp", pa.float64()),
    ("scan", pa.float64()), ("track", pa.float64()), ("acq_date", pa.date32()),
    ("acq_time", pa.string()), ("timestamp", pa.timestamp("ns")), ("confidence", pa.string()),
    ("type", pa.int8()), ("daynight", pa.string()), ("sensor", pa.string()),
    ("source_version", pa.string()),
])
SENSOR = {"A": "aqua", "AQUA": "aqua", "N": "snpp", "NPP": "snpp",
          "SNPP": "snpp", "S-NPP": "snpp", "T": "terra", "TERRA": "terra"}


class QuotaGate:
    def __init__(self):
        self._lock = Lock()
        self._resume = 0.0
        self._blocked = False
        self._probe = Lock()

    def enter(self):
        while True:
            with self._lock:
                delay = self._resume - monotonic()
                blocked = self._blocked
            if delay > 0:
                sleep(delay)
            elif not blocked:
                return False
            else:
                self._probe.acquire()
                with self._lock:
                    if self._blocked and self._resume <= monotonic():
                        return True
                self._probe.release()

    def leave(self, probing, limited):
        with self._lock:
            if limited:
                if not self._blocked:
                    print("NASA FIRMS quota reached; waiting before probing again", flush=True)
                self._blocked = True
                self._resume = max(self._resume, monotonic() + 65)
            elif probing:
                self._blocked = False
                print("NASA FIRMS quota available; resuming cached downloads", flush=True)
        if probing:
            self._probe.release()


def resolve_map_key():
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        env_path = Path(os.environ.get("FIRMS_ENV_FILE", ".env"))
        if env_path.is_file():
            content = env_path.read_text(encoding="utf-8-sig").strip()
            candidates = re.findall(r"(?i)(?<![a-z0-9])[a-f0-9]{32}(?![a-z0-9])", content)
            if len(candidates) == 1:
                key = candidates[0]
            elif re.fullmatch(r"[a-zA-Z0-9_-]+", content):
                key = content
            else:
                values = dotenv_values(env_path, encoding="utf-8-sig")
                key = values.get("FIRMS_MAP_KEY") or values.get("MAP_KEY")
    if not key:
        raise ValueError("Set FIRMS_MAP_KEY in .env or provide archive CSV/ZIP files")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", key):
        raise ValueError("Unexpected FIRMS_MAP_KEY format")
    return key


def fetch_window(session, gate, url, source, current, target):
    deadline = monotonic() + 1_200
    failures = 0
    while True:
        probing = gate.enter()
        limited = False
        transient = None
        try:
            with measure("download") as record:
                try:
                    with session.get(url, timeout=(15, 90), allow_redirects=False) as response:
                        payload = response.content
                        status = response.status_code
                        limited = status == 429 or (status == 400 and
                            b"exceeding allowed transaction limit" in payload.lower())
                        if status in (408, 500, 502, 503, 504):
                            transient = f"HTTP {status}"
                except requests.RequestException as error:
                    transient = type(error).__name__
                if not limited and transient is None:
                    if status != 200:
                        raise RuntimeError(f"FIRMS {source} {current}: HTTP {status}")
                    header = payload.splitlines()[0].decode("utf-8-sig") if payload else ""
                    if not set(RAW_COLUMNS).issubset(header.lower().split(",")):
                        raise RuntimeError(f"FIRMS returned an unexpected CSV schema for {source} {current}")
                    temporary = target.with_suffix(".part")
                    temporary.write_bytes(payload)
                    temporary.replace(target)
                    return record
        finally:
            gate.leave(probing, limited or (probing and transient is not None))
        if transient is not None:
            failures += 1
            session.close()
            if failures >= 6:
                raise RuntimeError(f"FIRMS {source} {current}: {transient} after {failures} attempts")
            print(f"Retrying FIRMS {source} {current} after {transient}", flush=True)
            sleep(min(5 * failures, 20))
        if monotonic() >= deadline:
            raise RuntimeError(f"FIRMS did not recover for {source} {current} within 20 minutes")

def download(start: date, end: date, raw: Path, bbox=BBOX, manifest=None, gate=None):
    if end < start:
        raise ValueError("End date precedes start date")
    key = resolve_map_key()
    gate = gate or QuotaGate()
    raw.mkdir(parents=True, exist_ok=True)
    metrics = []
    prior = {}
    if manifest is not None and manifest.is_file():
        prior = {(row["source"], row["start"], row["days"]): row
                 for row in json.loads(manifest.read_text(encoding="utf-8"))}
    area = ",".join(map(str, bbox))
    extent = "_".join(map(str, bbox))
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=1, backoff_factor=1, respect_retry_after_header=False, raise_on_status=False)))
        for source in SOURCES:
            current = start
            while current <= end:
                days = min(5, (end - current).days + 1)
                target = raw / f"{source}_{extent}_{current}_{days}d.csv"
                cache_key = (source, str(current), days)
                if target.exists():
                    record = prior.get(cache_key)
                    if record is None:
                        with measure("download") as record:
                            pass
                        record["cached"] = True
                else:
                    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/{days}/{current}"
                    record = fetch_window(session, gate, url, source, current, target)
                    record["cached"] = False
                stat = target.stat()
                record.update(source=source, start=str(current), days=days,
                              bytes=stat.st_size, downloaded_at_unix=stat.st_mtime)
                metrics.append(record)
                if manifest is not None:
                    prior[cache_key] = record
                    write_json(manifest, list(prior.values()))
                if len(metrics) % 40 == 0:
                    print(f"FIRMS {start.year}: {len(metrics)} windows processed ({source}, {current})",
                          flush=True)
                current += timedelta(days=days)
    if manifest is not None:
        write_json(manifest, metrics)
    return metrics

def csv_frames(path: Path, chunksize=200_000):
    options = dict(chunksize=chunksize, dtype={"acq_time": "string", "confidence": "string",
                                              "satellite": "string", "version": "string"}, encoding="utf-8-sig")
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            names = [name for name in archive.namelist()
                     if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")]
            if not names:
                raise ValueError(f"No CSV members in {path}")
            for name in names:
                with archive.open(name) as source:
                    yield from pd.read_csv(source, **options)
    else:
        yield from pd.read_csv(path, **options)


def normalize(frame, bbox=BBOX, audit=None):
    missing = set(RAW_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required FIRMS columns: {sorted(missing)}")
    version = frame["version"] if "version" in frame else pd.NA
    frame = frame.loc[:, list(RAW_COLUMNS)].copy()
    frame["source_version"] = version
    frame["sensor"] = frame.pop("satellite").astype("string").str.strip().str.upper().map(SENSOR)
    if frame["sensor"].isna().any():
        raise ValueError("Unexpected satellite value. Supply only MODIS Aqua/Terra and S-NPP")
    west, south, east, north = bbox
    for column in ("latitude", "longitude", "frp", "scan", "track", "type"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[["latitude", "longitude", "type"]]).all().all():
        raise ValueError("Input has invalid coordinates or hotspot type")
    frame = frame.loc[frame.sensor.ne("terra") & frame.longitude.between(west, east)
                      & frame.latitude.between(south, north)].copy()
    bad_frp = ~np.isfinite(frame.frp) | frame.frp.lt(0)
    bad_geometry = (~np.isfinite(frame[["scan", "track"]]).all(axis=1)
                    | frame.scan.le(0) | frame.track.le(0))
    if audit is not None:
        for sensor in ("aqua", "snpp"):
            platform = frame.sensor.eq(sensor)
            audit[f"{sensor}_invalid_frp"] += int((platform & bad_frp).sum())
            audit[f"{sensor}_invalid_footprint"] += int((platform & bad_geometry).sum())
    frame = frame.loc[~(bad_frp | bad_geometry)].copy()
    frame["acq_date"] = pd.to_datetime(frame.acq_date, format="%Y-%m-%d", errors="raise")
    frame["acq_time"] = frame.acq_time.astype("string").str.strip().str.zfill(4)
    valid_time = frame.acq_time.str.fullmatch(r"(?:[01][0-9]|2[0-3])[0-5][0-9]").fillna(False)
    if not valid_time.all() or frame.acq_date.isna().any():
        raise ValueError("Invalid FIRMS UTC timestamp")
    frame["timestamp"] = pd.to_datetime(frame.acq_date.dt.strftime("%Y-%m-%d") + frame.acq_time,
                                        format="%Y-%m-%d%H%M")
    frame["daynight"] = frame.daynight.astype("string").str.strip().str.upper()
    if not frame.daynight.isin(["D", "N"]).all() or not frame["type"].isin([0, 1, 2, 3]).all():
        raise ValueError("Invalid day/night or hotspot type")
    return frame.loc[:, SCHEMA.names]


def convert(raw: Path, output: Path, *, year=None, bbox=BBOX):
    paths = sorted(p for p in raw.iterdir() if p.suffix.lower() in {".csv", ".zip", ".gz"}) if raw.is_dir() else []
    if year is not None:
        paths = [path for path in paths if re.search(rf"_{year}-\d{{2}}-\d{{2}}_\d+d\.csv$", path.name)]
    if not paths:
        raise FileNotFoundError(f"No archive CSV/ZIP inputs in {raw}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".part")
    rows = 0
    input_rows = 0
    audit = Counter()
    versions = Counter()
    try:
        with measure("csv_to_parquet") as record, pq.ParquetWriter(temporary, SCHEMA, compression="zstd") as writer:
            for path in paths:
                for frame in csv_frames(path):
                    input_rows += len(frame)
                    normalized = normalize(frame, bbox=bbox, audit=audit)
                    versions.update({f"{sensor} {version}": int(count) for (sensor, version), count
                                     in normalized.value_counts(["sensor", "source_version"]).items()})
                    rows += len(normalized)
                    writer.write_table(pa.Table.from_pandas(normalized, schema=SCHEMA, preserve_index=False))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    record.update(input_rows=input_rows, output_rows=rows, input_bytes=sum(p.stat().st_size for p in paths),
                  parquet_bytes=output.stat().st_size, input_files=len(paths),
                  input_file_names=[path.name for path in paths], rejected=dict(audit),
                  source_versions=dict(versions), bbox=list(bbox))
    return record


def download_years(years, data: Path, workers: int):
    if workers < 1 or workers > 5:
        raise ValueError("Use between 1 and 5 download workers")
    gate = QuotaGate()

    def fetch(year):
        start = date(year, 1, 20 if year == 2012 else 1)
        end = date(year, 12, 31)
        manifest = data / "outputs" / f"download_{start}_{end}.json"
        records = download(start, end, data / "raw", manifest=manifest, gate=gate)
        print(f"FIRMS {year} complete: {len(records)} windows", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, year) for year in dict.fromkeys(years)]
        for future in as_completed(futures):
            future.result()


def parse_bbox(value):
    try:
        west, south, east, north = map(float, value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use west,south,east,north in decimal degrees") from error
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise argparse.ArgumentTypeError("Expected ordered longitude and latitude bounds")
    return west, south, east, north


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("download")
    get.add_argument("--start", type=date.fromisoformat, required=True)
    get.add_argument("--end", type=date.fromisoformat, required=True)
    years = commands.add_parser("download-years")
    years.add_argument("years", nargs="+", type=int)
    years.add_argument("--workers", type=int, default=3)
    commands.add_parser("convert")
    archive = commands.add_parser("convert-archive")
    archive.add_argument("--raw", type=Path, required=True, help="Directory for one region's archives")
    archive.add_argument("--output", type=Path, required=True, help="Separate regional .parquet file")
    archive.add_argument("--bbox", type=parse_bbox, required=True)
    args = parser.parse_args()
    if args.command == "download":
        manifest = args.data / "outputs" / f"download_{args.start}_{args.end}.json"
        download(args.start, args.end, args.data / "raw", manifest=manifest)
    elif args.command == "download-years":
        download_years(args.years, args.data, args.workers)
    elif args.command == "convert-archive":
        if args.output.suffix.lower() != ".parquet":
            parser.error("--output must end in .parquet")
        if args.output.resolve() == (args.data / "parquet" / "firms.parquet").resolve():
            parser.error("Use a separate Parquet file for archive imports")
        stats = convert(args.raw, args.output, bbox=args.bbox)
        write_json(args.output.with_suffix(".ingest.json"), stats)
        print(f"Imported {stats['output_rows']} rows to {args.output}")
    else:
        stats = convert(args.data / "raw", args.data / "parquet" / "firms.parquet")
        write_json(args.data / "outputs" / "ingest.json", stats)
    print(f"{args.command} complete")


if __name__ == "__main__":
    main()
