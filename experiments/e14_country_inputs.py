"""Download official FIRMS country records and annual land-cover clips for E14."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd
import requests

from experiments.e13_nasa_ranges import byte_range
from experiments.e14_landscape_inputs import NEW, YEARS
from src.landcover.download_mosaic import fetch_mosaic

COUNTRY = {
    "canada_boreal": "Canada",
    "romania_carpathians": "Romania",
    "indonesia_sumatra": "Indonesia",
    "zambia_savanna": "Zambia",
    "australia_west": "Australia"
}
PRODUCTS = ("modis", "viirs-snpp")
HOST = "https://firms.modaps.eosdis.nasa.gov"
BLOCK = 8 * 1024 * 1024


def acquire(out, region, year, product):
    country = COUNTRY[region]
    url = f"{HOST}/data/country/{product}/{year}/{product}_{year}_{country}.csv"
    target = out / "raw" / f"{region}_{year}_{product}.csv"
    record = dict(region=region, year=year, product=product, url=url)
    if target.is_file():
        return record | dict(status="cached", bytes=target.stat().st_size)
    # Reuse a verified prefix left by an interrupted full or ranged download.
    with requests.Session() as session:
        first, total = byte_range(session, url, 0, 15)
        if not first.startswith(b"latitude,"):
            raise ValueError(f"Unexpected FIRMS header: {target.name}")
        candidates = sorted(target.parent.glob(f"{target.stem}*.part"),
                            key=lambda p: p.stat().st_size, reverse=True)
        start, prefix = 0, None
        for part in candidates:
            size = part.stat().st_size
            if not 16 <= size <= total:
                continue
            with part.open("rb") as f:
                if f.read(16) != first:
                    continue
                f.seek(size - 16)
                last = f.read()
            expected, size_remote = byte_range(session, url, size-16, size-1)
            if last == expected and size_remote == total:
                start, prefix = size, part
                break
        temp = target.with_suffix(f".{uuid4().hex}.part")
        with temp.open("wb") as f:
            if prefix:
                with prefix.open("rb") as source:
                    shutil.copyfileobj(source, f)
            while start < total:
                end = min(start + BLOCK, total) - 1
                try:
                    block, size = byte_range(session, url, start, end)
                except (requests.RequestException, ValueError):
                    block, size = byte_range(session, url, start, min(start + 1024 * 1024, total)-1)
                if size != total:
                    raise ValueError("FIRMS file changed size during download")
                f.write(block)
                start += len(block)
        if temp.stat().st_size != total:
            raise ValueError(f"Incomplete FIRMS file: {target.name}")
        temp.replace(target)
    return record | dict(status="downloaded", bytes=total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regions", nargs="+", choices=NEW, default=list(NEW))
    args = parser.parse_args()
    out = args.output
    (out / "raw").mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(acquire,out,r,y,p):(r,y,p) for r in args.regions
                   for y in YEARS for p in PRODUCTS}
        for future in as_completed(futures):
            try:
                item = future.result()
            except (requests.RequestException, ValueError, OSError) as error:
                r,y,p = futures[future]
                item = dict(region=r, year=y, product=p, status=type(error).__name__, bytes=0)
            records.append(item)
            print(item["region"], item["year"], item["product"], item["status"], flush=True)
    pd.DataFrame(records).sort_values(["region","year","product"]).to_csv(out / "download_audit.csv", index=False)
    if any(not item["bytes"] for item in records):
        parser.exit(2, "Some FIRMS files are incomplete; rerun to resume.\n")
    for r in args.regions:
        for y in YEARS:
            target = out / "landcover" / r / f"LC_Type1_{y}.tif"
            if not target.is_file():
                fetch_mosaic(y, target, NEW[r])


if __name__ == "__main__":
    main()
