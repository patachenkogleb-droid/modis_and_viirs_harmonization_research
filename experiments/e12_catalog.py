"""Match FIRMS pilot pixels to candidate Aqua and S-NPP swath metadata."""
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, Polygon

from experiments.regional_daily_inputs import REGIONS

COLLECTIONS = {
    "aqua": ("C2278858993-LPCLOUD", "C1379841358-LAADS"),
    "snpp": ("C2734202914-LPCLOUD", "C2105092163-LAADS"),
}
PRODUCTS = {"aqua": ("MYD14", "MYD03"), "snpp": ("VNP14IMG", "VNP03IMG")}
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
WINDOW_MINUTES = 60


def cmr_search(session, collection, bbox, day):
    start = day - pd.Timedelta(minutes=WINDOW_MINUTES)
    end = day + pd.Timedelta(days=1, minutes=WINDOW_MINUTES)
    response = session.get(CMR, params={
        "collection_concept_id": collection, "bounding_box": ",".join(map(str, bbox)),
        "temporal": f"{start:%Y-%m-%dT%H:%M:%SZ},{end:%Y-%m-%dT%H:%M:%SZ}", "page_size": 500,
    }, timeout=30)
    response.raise_for_status()
    entries = response.json()["feed"]["entry"]
    if int(response.headers.get("CMR-Hits", len(entries))) > len(entries):
        raise ValueError("Incomplete CMR granule results")
    return entries


def slot(entry):
    return ".".join(entry["producer_granule_id"].split(".")[1:3])


def download_url(entry, extension):
    if entry is None:
        return None
    urls = [link["href"] for link in entry.get("links", [])
            if link.get("rel", "").endswith("/data#") and link["href"].startswith("https://")
            and urlsplit(link["href"]).path.endswith(extension)]
    return urls[0] if urls else None


def footprint(entry, lat, lon):
    """Catalog polygon inclusion; swath metadata is not a pixel-level observation."""
    rings = [ring for polygon in entry.get("polygons", []) for ring in polygon]
    for ring in rings:
        lats, lons = np.asarray(ring.split(), dtype=float).reshape(-1, 2).T
        unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lons)))
        point_lon = lon + 360 * round((unwrapped.mean() - lon) / 360)
        polygon = Polygon(zip(unwrapped, lats))
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.covers(Point(point_lon, lat)):
            return True
    return False


def query_case(item):
    (region, date), samples = item
    _, bbox = REGIONS[region]
    rows = []
    with requests.Session() as session:
        for sensor, (mask_id, geo_id) in COLLECTIONS.items():
            masks = cmr_search(session, mask_id, bbox, date)
            by_slot = {slot(entry): entry for entry in cmr_search(session, geo_id, bbox, date)}
            for p in samples.itertuples():
                role = "source" if sensor == p.sensor else "counterpart"
                for mask in masks:
                    start = pd.Timestamp(mask["time_start"]).tz_convert(None)
                    end = pd.Timestamp(mask["time_end"]).tz_convert(None)
                    minutes = max((start - p.timestamp).total_seconds(), 0,
                                  (p.timestamp - end).total_seconds()) / 60
                    if minutes >= WINDOW_MINUTES or not footprint(mask, p.latitude, p.longitude):
                        continue
                    geo = by_slot.get(slot(mask))
                    ext = ".hdf" if sensor == "aqua" else ".nc"
                    rows.append(dict(
                        sample_id=p.sample_id, sensor=sensor, role=role, slot=slot(mask),
                        mask_granule=mask["producer_granule_id"], geo_granule=(
                            geo["producer_granule_id"] if geo else None),
                        mask_url=download_url(mask, ext), geo_url=download_url(geo, ext),
                        start_utc=start, end_utc=end, min_gap_minutes=minutes,
                    ))
    return rows


def catalog(samples):
    groups = list(samples.groupby(["region", "acq_date"], sort=True))
    with ThreadPoolExecutor(max_workers=5) as pool:
        all_rows = list(pool.map(query_case, groups))
    table = pd.DataFrame([r for group in all_rows for r in group])
    if table.empty:
        raise ValueError("No CMR candidates returned for any pilot pixel")
    return table.sort_values(["sample_id", "role", "min_gap_minutes", "start_utc"])
