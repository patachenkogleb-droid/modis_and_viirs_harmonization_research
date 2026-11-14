"""E9: reproducible calibration of regional UTC daily FIRMS sums."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import duckdb
import pandas as pd

from experiments.regional_daily_inputs import REGIONS, YEARS, check_archive, prepare_year, season, verify_boundary
from src.landcover import landcover_mapper
from src.pairing import match_sensors
from experiments import regional_daily_inputs, e8_unpaired_detections


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def identity(path):
    return (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) if path.is_file() else None


def prepare(data, landcover, output):
    code = sha256(b"".join(Path(module.__file__).read_bytes() for module in
                           (regional_daily_inputs, match_sensors, landcover_mapper,
                            e8_unpaired_detections))).hexdigest()
    archive, checks, audit, evidence = [], [], [], []
    with duckdb.connect(config={"threads": "4"}) as db:
        archive.extend(check_archive(db, data, region) for region in REGIONS)
        boundary = verify_boundary(db, data)
        for region, (rel, _) in REGIONS.items():
            for year in YEARS:
                base = output / "cache" / region / str(year)
                base.mkdir(parents=True, exist_ok=True)
                raster = Path(f"LC_Type1_{year}.tif")
                if region != "western_us":
                    raster = Path(region) / raster
                signature = json.dumps([code, identity(data / rel), identity(landcover / raster),
                                        identity(data / "landcover" / raster)])
                files = [base / f"{name}.parquet" for name in ("pairs", "reported", "evidence", "audit")]
                manifest = base / "checks.json"
                if manifest.is_file() and all(path.is_file() for path in files):
                    cached = json.loads(manifest.read_text(encoding="utf-8"))
                    if cached["signature"] == signature:
                        checks.append(cached["checks"])
                        evidence.append(pd.read_parquet(files[2]))
                        audit.append(pd.read_parquet(files[3]))
                        continue
                print(f"Preparing {region} {year}", flush=True)
                *frames, check = prepare_year(db, data, landcover, region, year)
                prefix = dict(region=region, year=year, period=season(year))
                if manifest.is_file():
                    manifest.unlink()
                for path, frame in zip(files, frames):
                    frame.assign(**prefix).to_parquet(path, index=False)
                check = {**prefix, **check}
                write_json(manifest, dict(signature=signature, checks=check))
                checks.append(check)
                evidence.append(pd.read_parquet(files[2]))
                audit.append(pd.read_parquet(files[3]))
    pd.DataFrame(checks).to_csv(output / "grouping_audit.csv", index=False)
    pd.concat(audit, ignore_index=True).to_csv(output / "archive_audit.csv", index=False)
    tab = pd.concat(evidence, ignore_index=True)
    tab = tab.groupby(["region", "period", "phase", "sensor", "evidence"], as_index=False)[
        ["groups", "frp_mw"]].sum()
    total = tab.groupby(["region", "period", "phase", "sensor"])[["groups", "frp_mw"]].transform("sum")
    tab[["group_fraction", "frp_fraction"]] = tab[["groups", "frp_mw"]] / total.to_numpy()
    tab.to_csv(output / "unpaired_evidence.csv", index=False)
    write_json(output / "input_checks.json", dict(archives=archive, boundary=boundary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--landcover-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output or args.data / "outputs" / "e9"
    output.mkdir(parents=True, exist_ok=True)
    prepare(args.data, args.landcover_root or args.data / "landcover", output)
    if not args.prepare_only:
        from experiments.regional_daily_models import analyze
        analyze(output, args.data / "outputs")


if __name__ == "__main__":
    main()
