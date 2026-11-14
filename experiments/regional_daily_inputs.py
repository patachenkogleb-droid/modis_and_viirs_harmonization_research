"""Prepare E9 one region/year at a time; retain only groups, pairs and daily summaries."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from experiments.e8_unpaired_detections import classify_unpaired, mark_matches
from experiments.run import Context
from src.calibration.harmonizer import frp_bin
from src.grid.spatial_grid import add_grid, grid_indices
from src.landcover.download_mosaic import validate_clip
from src.landcover.landcover_mapper import IGBP_GROUPS, attach_landcover
from src.pairing.match_sensors import PASS_KEYS, aggregate_passes, match_aggregated_passes

START = "2012-01-20"
END = "2020-12-31"
GRID = 3000
REGIONS = {
    "western_us": ("parquet/firms.parquet", (-125, 32, -102, 49)),
    **{region: (f"parquet/regions/{region}_{START}_{END}.parquet", bbox)
       for region, bbox in {
           "eastern_us": (-88, 32, -76, 43),
           "southern_amazon": (-65, -13, -51, -3),
           "cerrado": (-51, -22, -43, -9),
       }.items()},
}
YEARS = range(2012, 2021)
PHASES = ("D", "N", "all")
SENSORS = ("aqua", "snpp")
DETECTION = ["sensor", "timestamp", "latitude", "longitude"]
INVERSE = Transformer.from_crs(6933, 4326, always_xy=True)
BORDER_X = int(grid_indices(np.array([-51.]), np.array([0.]), GRID)[0][0])
BORDER_LON = float(INVERSE.transform((BORDER_X + .5) * GRID, 0)[0])
BORDER_OWNER = "southern_amazon" if BORDER_LON < -51 else "cerrado"
LAT_EDGES = [0, 15, 30, 40, 90]
LAT_LABELS = ["0-<15", "15-<30", "30-<40", "40-90"]
FRP_COLS = ["aqua", "snpp"]


def season(year):
    return "fit" if year <= 2014 else "validation" if year <= 2016 else "holdout"


def check_archive(db, data, region):
    rel, bbox = REGIONS[region]
    path = data / rel
    lo, hi, rows = db.execute("SELECT min(acq_date), max(acq_date), count(*) FROM read_parquet(?)",
                              [str(path)]).fetchone()
    if str(lo) != START or str(hi) < END:
        raise ValueError(f"{region}: archive dates {lo} to {hi}")
    if region == "western_us":
        if not Context(data, GRID, None).archive_complete(2012, 2020):
            raise ValueError("Missing western Area API manifest slices")
    else:
        info = json.loads(path.with_suffix(".ingest.json").read_text(encoding="utf-8"))
        if str(hi) != END or info["bbox"] != list(bbox) or info["output_rows"] != rows:
            raise ValueError(f"{region}: archive disagrees with its ingest manifest")
        versions = db.execute(
            """SELECT sensor || ' ' || source_version AS version, count(*)
               FROM read_parquet(?) GROUP BY 1""", [str(path)]).fetchall()
        if dict(versions) != info["source_versions"]:
            raise ValueError(f"{region}: source versions disagree with the ingest manifest")
        folder = data / "archive_download" / f"{region}_{START}_{END}"
        if len(set(info["input_file_names"])) != 2 or not all(
                (folder / name).is_file() for name in info["input_file_names"]):
            raise ValueError(f"{region}: missing original archive ZIPs")
    return dict(region=region, parquet=rel, bbox=list(bbox), archive_rows=rows,
                archive_first=str(lo), archive_last=str(hi))


def read_year(db, path, bbox, year):
    first = START if year == 2012 else f"{year}-01-01"
    points = db.execute(
        "SELECT * FROM read_parquet(?) WHERE acq_date BETWEEN ?::DATE AND ?::DATE",
        [str(path), first, f"{year}-12-31"]).fetchdf()
    west, south, east, north = bbox
    if (points.isna().any().any() or not points.daynight.isin(["D", "N"]).all()
            or set(points.sensor) != set(SENSORS)
            or not (points.timestamp.dt.normalize() == points.acq_date).all()
            or not (points.timestamp.dt.strftime("%H%M") == points.acq_time).all()
            or not points.longitude.between(west, east).all()
            or not points.latitude.between(south, north).all()
            or not np.isfinite(points[["frp", "scan", "track"]]).all().all()
            or (points.frp < 0).any() or (points[["scan", "track"]] <= 0).any().any()
            or points.source_version.str.strip().eq("").any()):
        raise ValueError(f"Invalid archive fields: {path}, {year}")
    points["duplicate_exact"] = points.duplicated()
    points["duplicate_key"] = points.duplicated(DETECTION)
    audit = points.assign(vegetation=points.type.eq(0)).groupby(
        ["sensor", "source_version", "daynight"], as_index=False).agg(
        rows=("frp", "size"), vegetation_rows=("vegetation", "sum"),
        duplicate_exact=("duplicate_exact", "sum"), repeated_key=("duplicate_key", "sum"),
        first=("acq_date", "min"), last=("acq_date", "max"),
        lon_min=("longitude", "min"), lon_max=("longitude", "max"),
        lat_min=("latitude", "min"), lat_max=("latitude", "max"))
    # Exact repeats are duplicates. Same location and minute with different payloads
    # can be different source records; report them and preserve them for pilot parity.
    return points.loc[points.type.eq(0) & ~points.duplicate_exact].copy(), audit


def daily_summary(marked):
    parts = []
    for phase in PHASES:
        data = marked if phase == "all" else marked.loc[marked.daynight.eq(phase)]
        by_sensor = {}
        for sensor in SENSORS:
            sub = data.loc[data.sensor.eq(sensor)]
            total = sub.groupby("acq_date").sum_frp.agg(["sum", "size"])
            paired = sub.loc[sub.matched].groupby("acq_date").sum_frp.sum()
            total[sensor + "_matched"] = paired.reindex(total.index, fill_value=0)
            total[sensor + "_unpaired"] = total["sum"] - total[sensor + "_matched"]
            for dn in ("D", "N"):
                channel = sub.loc[sub.daynight.eq(dn)].groupby("acq_date").sum_frp.sum()
                total[sensor + "_" + dn] = channel.reindex(total.index)
            by_sensor[sensor] = total.rename(columns={"sum": sensor, "size": sensor + "_groups"})
        daily = by_sensor["aqua"].join(by_sensor["snpp"], how="outer")
        for sensor in SENSORS:
            daily["has_" + sensor] = daily[sensor].notna()
            daily["has_" + sensor + "_D"] = daily[sensor + "_D"].notna()
            daily["has_" + sensor + "_N"] = daily[sensor + "_N"].notna()
        daily["both"] = daily.has_aqua & daily.has_snpp
        check = daily.loc[daily.both]
        daily["difference_mw"] = daily.aqua - daily.snpp
        daily["matched_difference_mw"] = daily.aqua_matched - daily.snpp_matched
        np.testing.assert_allclose(
            check.aqua - check.snpp,
            check.aqua_matched - check.snpp_matched + check.aqua_unpaired - check.snpp_unpaired,
            atol=1e-7)
        parts.append(daily.reset_index().assign(phase=phase))
    return pd.concat(parts, ignore_index=True)


def evidence_summary(marked):
    classes = classify_unpaired(marked)
    output = []
    for phase in PHASES:
        part = classes if phase == "all" else classes.loc[classes.daynight.eq(phase)]
        output.append(part.groupby(["sensor", "evidence"], as_index=False).sum_frp.agg(
            groups="size", frp_mw="sum").assign(phase=phase))
    return pd.concat(output, ignore_index=True)


def prepare_year(db, data, raster_root, region, year):
    rel, bbox = REGIONS[region]
    points, audit = read_year(db, data / rel, bbox, year)
    points = add_grid(points, GRID)
    border_points = points.iloc[:0]
    if region in ("southern_amazon", "cerrado"):
        border_points = points.loc[points.cell_x.eq(BORDER_X)]
        if region != BORDER_OWNER:
            points = points.loc[points.cell_x.ne(BORDER_X)]
    passes = aggregate_passes(points, with_scan=True)
    assert not passes.duplicated(["sensor", *PASS_KEYS, "time"]).any()
    left = points.groupby("sensor").frp.agg(["size", "sum"]).sort_index()
    right = passes.groupby("sensor").agg(size=("rows", "sum"), sum=("sum_frp", "sum")).sort_index()
    pd.testing.assert_frame_equal(left, right, check_exact=False, atol=1e-7, rtol=1e-12)
    pairs, coverage = match_aggregated_passes(passes)
    assert pairs.delta_minutes.lt(60).all()
    marked = mark_matches(passes, pairs)
    assert marked.matched.sum() == 2 * len(pairs)
    daily = daily_summary(marked)
    valid = pairs.sum_frp_modis.gt(0) & pairs.sum_frp_viirs.gt(0)
    aqua = passes.loc[passes.sensor.eq("aqua"), [*PASS_KEYS, "time", "scan_aqua"]]
    pairs = pairs.assign(eligible=valid).merge(aqua, left_on=[*PASS_KEYS, "time_aqua"],
                                   right_on=[*PASS_KEYS, "time"], validate="one_to_one").drop(columns="time")
    year_rasters = raster_root if region == "western_us" else raster_root / region
    raster = year_rasters / f"LC_Type1_{year}.tif"
    if not raster.exists() and raster_root != data / "landcover":
        year_rasters = data / "landcover" if region == "western_us" else data / "landcover" / region
        raster = year_rasters / raster.name
    if raster.exists():
        with rasterio.open(raster) as dataset:
            validate_clip(dataset, bbox, year, require_tags=region != "western_us")
    pairs = attach_landcover(pairs, year_rasters, GRID, audit=True)
    _, latitude = INVERSE.transform((pairs.cell_x.to_numpy() + .5) * GRID,
                                    (pairs.cell_y.to_numpy() + .5) * GRID)
    strong = pairs.grouped_share.ge(.60) & pairs.grouped_landcover.isin(IGBP_GROUPS.values())
    pairs = pairs.assign(
        latitude=latitude, latitude_band=pd.cut(abs(latitude), LAT_EDGES, labels=LAT_LABELS,
                                                right=False),
        frp_band=frp_bin(pairs.sum_frp_modis),
        strong_cover=pairs.grouped_landcover.where(strong))
    cells = pairs.drop_duplicates(["cell_x", "cell_y"])
    checks = dict(
        **coverage, eligible_pairs=int(pairs.eligible.sum()), raster_available=raster.is_file(),
        cell_years=len(cells), strong_cell_years=int(cells.strong_cover.notna().sum()),
        strong_pairs=int((strong & pairs.eligible).sum()), weak_valid_cells=int(
            (cells.strong_cover.isna() & cells.valid_share.ge(.5)).sum()),
        missing_cells=int(cells.valid_share.lt(.5).sum()), border_rows=len(border_points),
        border_frp_mw=float(border_points.frp.sum()), border_cells=int(border_points.cell_y.nunique()),
        excluded_border_rows=len(border_points) if region != BORDER_OWNER else 0,
        excluded_border_frp_mw=float(border_points.frp.sum()) if region != BORDER_OWNER else 0.0,
        vegetation_frp_mw=float(points.frp.sum()))
    return pairs, daily, evidence_summary(marked), audit, checks


def verify_boundary(db, data):
    amazon = str(data / REGIONS["southern_amazon"][0])
    cerrado = str(data / REGIONS["cerrado"][0])
    keys = ", ".join(DETECTION)
    count = db.execute(f"""WITH a AS (SELECT DISTINCT {keys} FROM read_parquet(?) WHERE longitude = -51),
                                c AS (SELECT DISTINCT {keys} FROM read_parquet(?) WHERE longitude = -51)
                            SELECT count(*) FROM (SELECT * FROM a INTERSECT SELECT * FROM c)""",
                       [amazon, cerrado]).fetchone()[0]
    return dict(shared_meridian_detection_keys=count, shared_grid_column=BORDER_X,
                grid_column_center_lon=BORDER_LON, owner=BORDER_OWNER)
