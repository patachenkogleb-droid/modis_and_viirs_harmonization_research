"""Build regional day and cover sums for Aqua/S-NPP landscape transfer."""
import argparse
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from experiments.regional_daily_inputs import BORDER_OWNER, BORDER_X, GRID, REGIONS, read_year
from src.grid.spatial_grid import add_grid
from src.ingest.firms_client import csv_frames, normalize
from src.landcover.landcover_mapper import IGBP_GROUPS, attach_landcover
from src.pairing.match_sensors import aggregate_passes, match_aggregated_passes

NEW = {
    "canada_boreal": [
        -123,
        51,
        -101,
        62
    ],
    "romania_carpathians": [
        20,
        43,
        30,
        49
    ],
    "indonesia_sumatra": [
        94,
        -6,
        107,
        6
    ],
    "zambia_savanna": [
        21,
        -19,
        34,
        -8
    ],
    "australia_west": [
        113,
        -36,
        127,
        -24
    ]
}
YEARS = (2018, 2019, 2020)
ALL_REGIONS = (*REGIONS, *NEW)
KEYS = ["acq_date", "daynight", "cover"]


def read_new(out, region, year):
    chunks = []
    versions = Counter()
    for product, sensor in (("modis", "aqua"), ("viirs-snpp", "snpp")):
        path = out / "raw" / f"{region}_{year}_{product}.csv"
        for frame in csv_frames(path):
            clean = normalize(frame, NEW[region])
            if not (clean.sensor.eq(sensor).all() and clean.acq_date.dt.year.eq(year).all()
                    and clean.source_version.notna().all()
                    and not clean.source_version.str.contains("NRT|RT", case=False).any()):
                raise ValueError(f"Sensor, version or year mismatch in {path.name}")
            versions.update(clean.source_version)
            chunks.append(clean)
    points = pd.concat(chunks, ignore_index=True).drop_duplicates()
    print(region, year, len(points), dict(versions), flush=True)
    return points.loc[points.type.eq(0)]


def prepare(data, out, region, year):
    if region in NEW:
        points = read_new(out, region, year)
        rasters = out / "landcover" / region
    else:
        rel, bbox = REGIONS[region]
        with duckdb.connect(config={"threads": "3"}) as db:
            points, _ = read_year(db, data / rel, bbox, year)
        rasters = data / "landcover"
        if region != "western_us":
            rasters = rasters / region
    raster = rasters / f"LC_Type1_{year}.tif"
    if not raster.is_file():
        raise FileNotFoundError(raster)
    if set(points.sensor) != {"aqua", "snpp"}:
        raise ValueError(f"Both sensors must be present in {region}, {year}")
    points = add_grid(points, GRID)
    if region in ("southern_amazon", "cerrado") and region != BORDER_OWNER:
        points = points.loc[points.cell_x.ne(BORDER_X)]
    passes = aggregate_passes(points)
    for sensor in ("aqua", "snpp"):
        np.testing.assert_allclose(passes.loc[passes.sensor.eq(sensor), "sum_frp"].sum(),
                                   points.loc[points.sensor.eq(sensor), "frp"].sum(), atol=1e-7)
    passes = attach_landcover(passes, rasters, GRID, audit=True)
    passes["cover"] = passes.grouped_landcover.where(
        passes.grouped_share.ge(.60) & passes.grouped_landcover.isin(IGBP_GROUPS.values()), "unknown")
    classified = passes[["cell_x", "cell_y", "cover"]].drop_duplicates()
    pairs, _ = match_aggregated_passes(passes)
    pairs = pairs.loc[pairs.sum_frp_modis.gt(0) & pairs.sum_frp_viirs.gt(0)]
    pairs = pairs.merge(classified, on=["cell_x", "cell_y"], validate="many_to_one")
    matched = pairs.groupby(KEYS, as_index=False).agg(
        aqua=("sum_frp_modis", "sum"), snpp=("sum_frp_viirs", "sum"), pairs=("sum_frp_modis", "size"))
    total = passes.groupby([*KEYS, "sensor"]).sum_frp.sum().unstack("sensor", fill_value=0).reset_index()
    # Fill a missing cover contribution only on dates when both platforms reported
    # in the region and phase. Entire missing sensor-days stay out of the comparison.
    presence = passes.groupby(["acq_date", "daynight"]).sensor.nunique().rename("present")
    total = total.join(presence, on=["acq_date", "daynight"]).query("present == 2").drop(columns="present")
    return pd.concat([matched.assign(scope="matched"), total.assign(scope="reported")],
                     ignore_index=True).assign(region=region, year=year)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regions", nargs="+", choices=ALL_REGIONS, required=True)
    parser.add_argument("--years", nargs="+", type=int, choices=YEARS, default=list(YEARS))
    args = parser.parse_args()
    dest = args.output / "daily_inputs"
    dest.mkdir(parents=True, exist_ok=True)
    for region in args.regions:
        for year in args.years:
            target = dest / f"{region}_{year}.parquet"
            if not target.exists():
                print(f"Preparing {region} {year}", flush=True)
                data = prepare(args.data, args.output, region, year)
                temp = target.with_suffix(".part")
                data.to_parquet(temp, index=False)
                temp.replace(target)


if __name__ == "__main__":
    main()
