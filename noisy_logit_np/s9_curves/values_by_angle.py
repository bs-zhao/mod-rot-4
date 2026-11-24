"""Metric value at every angle (the 'loss' curves) in one table: results/values_by_angle.csv
rows = (db, source, metric, axes subset, angle) for the key metrics; plus the baseline R* value.  usage: python values_by_angle.py"""
import pandas as pd
from analyze_reg import load, RES, KEY
rows = []
for key, d in load().items():
    rot = d["rot"]
    for src in ("oracle", "nn"):
        for m in KEY:
            if m not in d[src].columns: continue
            v = d[src][m].values
            for i, r in rot.iterrows(): rows.append((key, src, m, r.axes_1based if r.candidate_type == "sweep" else "baseline", int(r.angle_deg), float(v[i])))
pd.DataFrame(rows, columns=["db", "source", "metric", "axes", "angle_deg", "value"]).to_csv(RES / "values_by_angle.csv", index=False)
