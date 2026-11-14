import argparse
import importlib
import json
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.filters.clean_fires import filter_vegetation
from src.grid.spatial_grid import add_grid
from src.ingest.firms_client import BBOX, SOURCES
from src.pairing.match_sensors import match_passes
from src.utils.benchmark import measure, write_json

EXPERIMENTS = (
    "e0_download_benchmark", "e1_time_offset_orbit", "e2_grid_size_stability",
    "e3_frp_nonlinearity_bins", "e4_landcover_variance", "e5_scan_edge_daynight",
    "e6_backtest_validation",
    "e7_daily_calibration",
    "e8_unpaired_detections",
)
TRAIN = (2012, 2016)
HOLDOUT = (2017, 2020)


class MissingInputs(Exception):
    pass


class Context:
    def __init__(self, data, grid_size, rust):
        self.data = data
        self.output = data / "outputs"
        self.parquet = data / "parquet" / "firms.parquet"
        self.grid_size = grid_size
        self.rust = rust
        self.metrics = []
        self.output.mkdir(parents=True, exist_ok=True)

    def load(self, first, last, daynight=None):
        if not self.parquet.is_file():
            raise MissingInputs(f"Run FIRMS conversion to create {self.parquet}")
        with measure("parquet_query") as record, duckdb.connect() as connection:
            frame = connection.execute(
                """SELECT * FROM read_parquet(?) WHERE acq_date BETWEEN make_date(?, 1, 1)
                   AND make_date(?, 12, 31) AND (CAST(? AS VARCHAR) IS NULL OR daynight = ?)
                   ORDER BY acq_date, timestamp""",
                [str(self.parquet), first, last, daynight, daynight],
            ).fetchdf()
        frame = filter_vegetation(frame)
        if set(frame.sensor.unique()) != {"aqua", "snpp"}:
            raise MissingInputs(f"Both sensors are required in {first}–{last}")
        record["rows"] = len(frame)
        self.metrics.append(record)
        return frame

    def archive_complete(self, first, last):
        ingest = self.output / "ingest.json"
        if not ingest.exists():
            return False
        names = set(json.loads(ingest.read_text(encoding="utf-8")).get("input_file_names", []))
        extent = "_".join(map(str, BBOX))
        for year in range(first, last + 1):
            current = date(year, 1, 20 if year == 2012 else 1)
            end = date(year, 12, 31)
            while current <= end:
                days = min(5, (end - current).days + 1)
                if not all(f"{source}_{extent}_{current}_{days}d.csv" in names for source in SOURCES):
                    return False
                current += timedelta(days=days)
        return True

    def pair(self, frame, size=None, minutes=60):
        with measure("gridding") as record:
            gridded = add_grid(frame, size or self.grid_size)
        record.update(rows=len(frame), grid_size_m=size or self.grid_size)
        self.metrics.append(record)
        with measure("pairing") as record:
            pairs, coverage = match_passes(gridded, max_minutes=minutes)
        record.update(rows=len(frame), pairs=len(pairs))
        self.metrics.append(record)
        return pairs, coverage

    def write_table(self, filename, records):
        pd.DataFrame.from_records(records).to_csv(self.output / filename, index=False)

    def plot(self, filename, x, y, ylabel, xlabel, *, line=False, intervals=None,
             x_rotation=0, reference=None):
        figure, axis = plt.subplots(figsize=(8, 4))
        if line:
            axis.plot(x, y, marker="o")
        else:
            axis.bar(x, y)
        if intervals:
            for position, (low, high) in enumerate(intervals):
                if low is not None:
                    axis.vlines(position, low, high, color="#263238", linewidth=2)
                    axis.hlines((low, high), position - .09, position + .09, color="#263238")
        if reference is not None:
            axis.axhline(reference, color="#c04a27", linestyle="--", label=f"{reference} min")
            axis.legend()
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.tick_params(axis="x", labelrotation=x_rotation)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(self.output / filename, dpi=160)
        plt.close(figure)


def run(indices=range(7)):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--grid-size", type=int, choices=(3000, 6000, 15000), default=3000)
    parser.add_argument("--rust", type=Path)
    args = parser.parse_args()
    for index in indices:
        context = Context(args.data, args.grid_size, args.rust)
        name = EXPERIMENTS[index]
        module = importlib.import_module(f"experiments.{name}")
        try:
            payload = module.experiment(context)
        except MissingInputs as error:
            payload = {"status": "NOT_RUN", "reason": str(error)}
        payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        payload["profiling"] = context.metrics
        payload["reference_grid_size_m"] = context.grid_size
        write_json(context.output / f"e{index}.json", payload)
        print(f"E{index}: {payload['status']} — {payload.get('reason', 'see data/outputs')}", flush=True)


if __name__ == "__main__":
    from experiments.run import run as run_experiments

    run_experiments()
