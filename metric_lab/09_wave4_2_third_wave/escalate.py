"""W3-05 section 23: preregistered candidate escalation AFTER all pilot blind solutions are locked and evaluated (development selection,
truth used only in this research comparison).  Writes configs/finalists.json (frozen; never edited by hand).

usage: python escalate.py            (reads results/pilot_recovery.csv and results/local_flags.csv.gz, results/oracle_summary.csv if present)
Rules (verbatim implementation of section 23):
  * E02 and X02 controls always retained (E02 = MX-B08c03@AD, X02 = MX-SL0UXUU@RBF);
  * per family (RW, MX, G/D, reduced-axis) at most three fixed configurations ranked by number of toys passing on S9_2 at pilot (a toy passes
    when every pilot search seed passes), then worst MCC gap, then worst leak excess, then lexical id; separately the best S9 configuration;
  * at most two theory candidates per structural family with every analytic local sweep perfect (24/24 libraries, Q01 pilot) and no observed
    practical global failure (oracle_summary: practical_adequate on every target where available), ranked by joint (analytic and inferred)
    marginal count and then worst analytic margin; labelled 'theory';
  * if a family's pilots do not reach the baseline (E02 at pilot, S9_2 toys), only its best mechanistically informative configuration advances.
Declared deviations (applied before ranking, never by truth): configurations that are numerically identical to a retained control by
construction are not advanced again (RW-POW-a1.0-c* = B08.M03.c*, RW-SM-a1.0-t* has weights e_i exactly -> E02 for c = .03); the pooled-HE
controls of ST-C (ST-HE-*) are controls, not structural candidates; the reduced-axis pilot (capped RA budget) is not escalated.
MX-SEL needs its component and F_w searches at every integration seed: when it advances, MX-F-w0.25 / w0.5 / w0.75 are added as support jobs.
"""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave3.common import RES, CONFIGS, leak_threshold
import numpy as np, pandas as pd


def family(cid):
    if cid.startswith("RW-"):
        return "RW"
    if cid.startswith("MX-"):
        return "MX"
    if cid == "ST-E-RA":
        return "RA"
    if cid.startswith("ST-"):
        return "GD"
    return "CTRL"


def coverage(br):
    br = br[br.panel == "primary"].copy(); br["thr"] = br.toy.map(leak_threshold); br["leak_excess"] = br.leak_max - br.thr
    g = br.groupby(["cid", "variant", "toy"]).agg(toy_pass=("pass_lab05", "all"), n=("pass_lab05", "size"), gap=("MCC_gap", "max"), lex=("leak_excess", "max")).reset_index()
    c = g.groupby(["cid", "variant"]).agg(n_toys=("toy_pass", "sum"), worst_gap=("gap", "max"), worst_leak_excess=("lex", "max"), n_cells=("n", "sum")).reset_index()
    c["family"] = c.cid.map(family); return c, g


def rank(c, variant):
    d = c[c.variant == variant].copy()
    return d.sort_values(["n_toys", "worst_gap", "worst_leak_excess", "cid"], ascending=[False, True, True, True])


def main():
    br = pd.read_csv(RES / "pilot_recovery.csv"); c, g = coverage(br); c.to_csv(RES / "pilot_coverage.csv", index=False); g.to_csv(RES / "pilot_toy_pass.csv", index=False)
    loc = pd.read_csv(RES / "local_flags.csv.gz"); loc = loc[(loc.fidelity == "pilot") & loc["mask"].isin(["FULL", "MASK-ALL", "ALLROWS"])]
    lf = loc.groupby("cid").agg(O_perf=("O_perfect", "sum"), O_marg=("O_marginal", "sum"), both=("both_marginal", "sum"), I_marg=("I_marginal", "sum"),
                                worst_O_margin=("O_min_margin", "min"), n=("toy", "size")).reset_index()
    og = pd.read_csv(RES / "oracle_summary.csv") if (RES / "oracle_summary.csv").is_file() else pd.DataFrame(columns=["cid", "practical_adequate"])
    fins = [dict(cid="E02-FIN", base="MX-B08c03@AD", role="control (E02)", family="CTRL"), dict(cid="X02-FIN", base="MX-SL0UXUU@RBF", role="control (X02)", family="CTRL")]
    chosen = {"MX-B08c03@AD", "MX-SL0UXUU@RBF"}
    equiv = {"RW-POW-a1.0-c0.03", "RW-SM-a1.0-t0.001", "RW-SM-a1.0-t0.01", "RW-SM-a1.0-t0.05", "E02", "E01", "RW-POW-a1.0-c0.01", "B07c03", "B01c03"}
    excluded = equiv | {c_ for c_ in c.cid if c_.startswith("ST-HE-") or c_ == "ST-E-RA"}
    base_n = int(c[(c.cid == "MX-B08c03@AD") & (c.variant == "s9_2")].n_toys.iloc[0])
    for fam in ("RW", "MX", "GD", "RA"):
        r = rank(c[(c.family == fam) & ~c.cid.isin(excluded)], "s9_2")
        if not len(r):
            continue
        top = [x for x in r.cid if x not in chosen][:3 if int(r.n_toys.iloc[0]) >= base_n else 1]
        for x in top:
            row = r[r.cid == x].iloc[0]
            fins.append(dict(cid=f"{x}-FIN", base=x, role=f"empirical top-3 S9_2 ({int(row.n_toys)}/12 toys)", family=fam)); chosen.add(x)
        r9 = rank(c[(c.family == fam) & ~c.cid.isin(excluded)], "s9")
        if len(r9) and r9.cid.iloc[0] not in chosen:
            x = r9.cid.iloc[0]; fins.append(dict(cid=f"{x}-FIN", base=x, role=f"best S9 ({int(r9.n_toys.iloc[0])}/12 toys)", family=fam)); chosen.add(x)
        th = lf[(lf.cid.map(family) == fam) & (lf.O_perf == lf.n) & (lf.n == 24)].copy()
        bad = set(og[og.practical_adequate == False].cid) if len(og) else set()
        th = th[~th.cid.isin(bad)].sort_values(["both", "worst_O_margin", "cid"], ascending=[False, False, True])
        for x in [x for x in th.cid if x not in chosen][:2]:
            row = th[th.cid == x].iloc[0]
            fins.append(dict(cid=f"{x}-FIN", base=x, role=f"theory (analytic perfect 24/24, joint marginal {int(row.both)}/24)", family=fam)); chosen.add(x)
    if any(f["base"] == "MX-SEL" for f in fins):
        for w in ("0.25", "0.5", "0.75"):
            if f"MX-F-w{w}" not in chosen:
                fins.append(dict(cid=f"MX-F-w{w}-FIN", base=f"MX-F-w{w}", role="support job for MX-SEL (selector pool)", family="MX")); chosen.add(f"MX-F-w{w}")
    for f in fins:
        f["iseeds"] = [31, 73, 101]; f["sseeds"] = [31, 73, 101]
    obj = dict(version="W3-FIN-1", rule="03_third_wave section 23 (escalate.py)", source_hash=hashlib.sha256((RES / "pilot_recovery.csv").read_bytes()).hexdigest()[:16],
               finalists=[f for f in fins if f["base"] != "ST-E-RA"], not_escalated_reduced_axis="ST-E-RA uses the capped RA budget only (declared pilot); 3x3 not run",
               note="development selection using T after locking; a deployed algorithm contains no T and no family-specific tuning")
    (CONFIGS / "finalists.json").write_text(json.dumps(obj, indent=1)); print(json.dumps(obj, indent=1))


if __name__ == "__main__":
    main()
