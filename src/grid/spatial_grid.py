import numpy as np
from pyproj import Transformer

CELL_SIZES = (3_000, 6_000, 15_000)
ORIGIN_X = 0
ORIGIN_Y = 0
_TRANSFORM = Transformer.from_crs(4326, 6933, always_xy=True)


def grid_indices(longitude, latitude, size: int):
    if size <= 0:
        raise ValueError("Grid size must be positive")
    x, y = _TRANSFORM.transform(longitude, latitude, errcheck=True)
    return (
        np.floor((x - ORIGIN_X) / size).astype("int32"),
        np.floor((y - ORIGIN_Y) / size).astype("int32"),
    )


def add_grid(frame, size: int):
    x, y = grid_indices(frame.longitude.to_numpy(), frame.latitude.to_numpy(), size)
    return frame.assign(cell_x=x, cell_y=y)
