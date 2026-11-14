from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import Window

from src.grid.spatial_grid import ORIGIN_X, ORIGIN_Y

IGBP_GROUPS = {
    1: "needleleaf", 3: "needleleaf", 2: "evergreen_broadleaf", 4: "deciduous_broadleaf",
    5: "mixed_forest", 6: "shrub_grass", 7: "shrub_grass", 10: "shrub_grass",
    8: "savanna", 9: "savanna", 12: "cropland", 14: "cropland",
}


AUDIT_COLUMNS = ["grouped_landcover", "grouped_share", "valid_share"]


def dominant_cells(cells, path: Path, size: int, sampling_m: int = 500, *, audit=False):
    """Count IGBP labels on an aligned equal-area raster inside each requested cell."""
    if size % sampling_m:
        raise ValueError("Grid size must be a multiple of land-cover sampling resolution")
    result = []
    cells = cells.loc[:, ["cell_x", "cell_y"]].drop_duplicates()
    columns = ["cell_x", "cell_y", "landcover"]
    if audit:
        columns += AUDIT_COLUMNS
    if cells.empty:
        return pd.DataFrame(columns=columns)
    groups = tuple(dict.fromkeys([*IGBP_GROUPS.values(), "other"]))
    group_indices = [[code - 1 for code in range(1, 18) if IGBP_GROUPS.get(code, "other") == group]
                     for group in groups]
    width = size // sampling_m
    with rasterio.open(path) as dataset:
        if dataset.crs is None or dataset.count != 1:
            raise ValueError("Supply a georeferenced, single-band LC_Type1 GeoTIFF")
        xmin, xmax = int(cells.cell_x.min()), int(cells.cell_x.max())
        ymin, ymax = int(cells.cell_y.min()), int(cells.cell_y.max())
        transform = from_origin(ORIGIN_X + xmin * size, ORIGIN_Y + (ymax + 1) * size,
                                sampling_m, sampling_m)
        with WarpedVRT(dataset, crs="EPSG:6933", transform=transform,
                       width=(xmax - xmin + 1) * width, height=(ymax - ymin + 1) * width,
                       nodata=255, resampling=Resampling.nearest) as aligned:
            for x, y in cells.itertuples(index=False, name=None):
                pixels = aligned.read(1, window=Window((x - xmin) * width, (ymax - y) * width,
                                                       width, width))
                counts = np.bincount(pixels.ravel(), minlength=256)[1:18]
                if counts.sum() < pixels.size / 2:
                    label = "missing"
                else:
                    modes = np.flatnonzero(counts == counts.max()) + 1
                    label = IGBP_GROUPS.get(int(modes[0]), "other") if len(modes) == 1 else "tie"
                row = (x, y, label)
                if audit:
                    grouped_counts = np.array([counts[indices].sum() for indices in group_indices])
                    leaders = np.flatnonzero(grouped_counts == grouped_counts.max())
                    grouped = groups[leaders[0]] if len(leaders) == 1 else "tie"
                    if label == "missing":
                        grouped = "missing"
                    row += (grouped, grouped_counts.max() / pixels.size, counts.sum() / pixels.size)
                result.append(row)
    return pd.DataFrame(result, columns=columns)


def attach_landcover(pairs, raster_dir: Path, size: int, *, audit=False):
    missing = {"landcover": "missing"}
    if audit:
        missing.update(grouped_landcover="missing", grouped_share=0.0, valid_share=0.0)
    pieces = []
    for year, group in pairs.groupby(pairs.acq_date.dt.year):
        path = raster_dir / f"LC_Type1_{year}.tif"
        if path.is_file():
            cells = dominant_cells(group, path, size, audit=audit)
            pieces.append(group.merge(cells, on=["cell_x", "cell_y"], validate="many_to_one"))
        else:
            pieces.append(group.assign(**missing))
    return pd.concat(pieces, ignore_index=True) if pieces else pairs.assign(**missing)
