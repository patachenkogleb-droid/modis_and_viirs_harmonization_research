NADIR_SCAN_KM = {"aqua": 1.0, "snpp": 0.375}


def filter_vegetation(frame):
    if frame["type"].isna().any():
        raise ValueError("Vegetation filtering requires type on all observations")
    return frame.loc[frame["type"].eq(0)]


def filter_scan_edge(frame, max_scan_ratio: float):
    if max_scan_ratio <= 0:
        raise ValueError("Scan ratio must be positive")
    nadir = frame["sensor"].map(NADIR_SCAN_KM)
    if nadir.isna().any():
        raise ValueError("Unrecognized sensor")
    return frame.loc[frame["scan"].le(nadir * max_scan_ratio)]


def filter_aqua_scan(frame, max_scan_km: float):
    return frame.loc[frame["sensor"].ne("aqua") | frame["scan"].le(max_scan_km)]
