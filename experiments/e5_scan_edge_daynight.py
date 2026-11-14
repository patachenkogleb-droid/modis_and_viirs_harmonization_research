from experiments.run import TRAIN, run
from src.calibration.harmonizer import summarize_ratio
from src.filters.clean_fires import filter_aqua_scan


def experiment(context):
    frame = context.load(*TRAIN)
    rows = []
    for phase in ("D", "N"):
        phase_frame = frame.loc[frame.daynight.eq(phase)]
        for cutoff in (None, 2.0):
            variant = phase_frame if cutoff is None else filter_aqua_scan(phase_frame, cutoff)
            pairs, coverage = context.pair(variant)
            rows.append(dict(daynight=phase, aqua_max_scan_km=cutoff, detection_rows=len(variant),
                             **coverage, **summarize_ratio(pairs)))
    context.write_table("e5_scan_daynight.csv", rows)
    valid = [row for row in rows if row["k"] is not None]
    if valid:
        context.plot("e5_scan_daynight.png",
                     [f"{row['daynight']} / {row['aqua_max_scan_km'] or 'all'}" for row in valid],
                     [row["k"] for row in valid], "k (VIIRS / MODIS)", "Day/night / Aqua max scan (km)")
    return {"status": "RUN" if len(valid) == len(rows) and context.archive_complete(*TRAIN) else "PARTIAL", "summary": rows,
            "years_in_input": sorted(frame.acq_date.dt.year.unique().tolist())}


if __name__ == "__main__":
    run([5])
