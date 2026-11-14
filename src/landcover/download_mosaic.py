import argparse
from math import ceil, floor, isfinite
from pathlib import Path
from uuid import uuid4

import rasterio
from rasterio.windows import Window, from_bounds

from src.ingest.firms_client import BBOX

BUFFER_DEGREES = 0.2
SOURCE_RECORD = 8338928


def parse_bbox(value):
    try:
        west, south, east, north = map(float, value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Specify W,S,E,N") from error
    if not all(map(isfinite, (west, south, east, north))) or not (
        -180 <= west < east <= 180 and -90 <= south < north <= 90
    ):
        raise argparse.ArgumentTypeError("Invalid W,S,E,N")
    return west, south, east, north


def validate_clip(dataset, bbox, year, *, require_tags=False):
    west, south, east, north = bbox
    bounds = dataset.bounds
    if (dataset.crs != rasterio.crs.CRS.from_epsg(4326) or dataset.count != 1
            or dataset.dtypes != ("uint8",) or dataset.nodata != 255
            or not (bounds.left < west < east < bounds.right
                    and bounds.bottom < south < north < bounds.top)):
        raise ValueError(f"Invalid clip geometry, dtype, CRS or nodata: {dataset.name}")
    tags = dataset.tags()
    if require_tags and (tags.get("year") != str(year) or tags.get("record") != str(SOURCE_RECORD)
                         or tags.get("product") != "MCD12Q1v061 LC_Type1"):
        raise ValueError(f"Wrong year or source for {dataset.name}")
    if require_tags and not (bounds.left <= west - BUFFER_DEGREES + 1e-7
                             and bounds.bottom <= south - BUFFER_DEGREES + 1e-7
                             and bounds.right >= east + BUFFER_DEGREES - 1e-7
                             and bounds.top >= north + BUFFER_DEGREES - 1e-7):
        raise ValueError(f"Insufficient buffer in {dataset.name}")


def fetch_mosaic(year: int, output: Path, bbox=BBOX):
    name = f"lc_mcd12q1v061.t1_c_500m_s_{year}0101_{year}1231_go_epsg.4326_v20230818.tif"
    url = f"https://zenodo.org/api/records/{SOURCE_RECORD}/files/{name}/content"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".{uuid4().hex}.part")
    west, south, east, north = bbox
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="35",
                      GDAL_HTTP_MAX_RETRY="2"):
        with rasterio.open(url) as dataset:
            if dataset.crs != rasterio.crs.CRS.from_epsg(4326) or dataset.count != 1:
                raise ValueError(f"Unexpected source geometry: {url}")
            extent = from_bounds(west - BUFFER_DEGREES, south - BUFFER_DEGREES,
                                 east + BUFFER_DEGREES, north + BUFFER_DEGREES, dataset.transform)
            col, row = floor(extent.col_off), floor(extent.row_off)
            window = Window(col, row, ceil(extent.col_off + extent.width) - col,
                            ceil(extent.row_off + extent.height) - row).intersection(
                                Window(0, 0, dataset.width, dataset.height))
            classes = dataset.read(1, window=window, masked=True).filled(255)
            # The source does not consistently declare nodata; IGBP LC_Type1 uses codes 1–17.
            classes[(classes < 1) | (classes > 17)] = 255
            if not ((classes >= 1) & (classes <= 17)).any():
                raise ValueError(f"No valid IGBP codes in {name} at {bbox}")
            profile = dict(driver="GTiff", count=1, width=int(window.width), height=int(window.height),
                           dtype="uint8", crs=dataset.crs, transform=dataset.window_transform(window),
                           nodata=255, tiled=True, compress="deflate")
            try:
                with rasterio.open(temporary, "w", **profile) as target:
                    target.write(classes, 1)
                    target.update_tags(year=year, product="MCD12Q1v061 LC_Type1", record=SOURCE_RECORD,
                                       source_file=name, bbox=",".join(map(str, bbox)))
                with rasterio.open(temporary) as target:
                    validate_clip(target, bbox, year, require_tags=True)
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Clip the open MCD12Q1 v061 LC_Type1 mosaic.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--bbox", type=parse_bbox)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("years", nargs="+", type=int, choices=range(2001, 2023))
    args = parser.parse_args()
    if args.bbox is not None and args.output_dir is None:
        parser.error("--bbox requires --output-dir to preserve the western US clips")
    for year in args.years:
        target = (args.output_dir or args.data / "landcover") / f"LC_Type1_{year}.tif"
        if not target.exists():
            fetch_mosaic(year, target, args.bbox or BBOX)
        with rasterio.open(target) as dataset:
            validate_clip(dataset, args.bbox or BBOX, year, require_tags=args.bbox is not None)
        print(f"{year}: {target.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
