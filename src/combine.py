"""Join routes + play_clock + qb_sight, assign buckets, print the headline, rate receivers.

Inputs
  outputs/routes.csv            from src/openness.py (Yatharth)
  outputs/openness_frames.csv   from src/openness.py, only for the window_start fallback
  outputs/play_clock.csv, outputs/qb_sight_frames.csv   from src/clock.py
Outputs
  outputs/buckets.csv           one row per route, with window flags and buckets
  outputs/receiver_rating.csv   one row per receiver with >= MIN_ROUTES routes

Usage: python src/combine.py [--routes outputs/routes.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import FPS, MIN_LEAD_S, MIN_ROUTES, OUT, load_plays

KEYS = ["gameId", "playId"]
RKEYS = KEYS + ["nflId"]

OPEN_SEP = 3.0          # sep >= 3 yd = open (shared openness definition)
WINDOW_FLOOR_S = 1.0    # fallback window_start is never earlier than this
CLOSED_SEP = 3.0        # sep_at_throw below this = not open at the throw = window closed
POSITIONS = ["WR", "TE", "RB"]   # FB is folded into RB
RATED_POSITIONS = ["WR", "TE"]   # RBs excluded from the rating (short checkdowns)
NO_RATE = ["S", "R"]    # sacks and scrambles: excluded from the ignored/targeted analysis
INSIGHT_EXPECT = 0.90   # sanity floor for "targeted receivers in sight at the throw"

# routes.csv columns this script needs. window_start and proj_sep_0_7 are optional:
# if a column (or a value) is missing, the fallbacks below are used.
REQUIRED = ["gameId", "playId", "nflId", "displayName", "position", "alignment",
            "coverage_type", "t_end", "sep_at_throw", "downfield_yds", "is_target",
            "is_clear_open_downfield", "passResult", "playResult",
            "time_to_open_after_covered"]
OPTIONAL = ["window_start", "proj_sep_0_7"]

# Sub-buckets for a clear-open downfield receiver, checked in order; first match wins.
BUCKETS = [
    ("targeted", None, "QB threw to him"),
    ("pressured", "TOO_LATE", "his window opened at or after the first pressure"),
    ("late", "TOO_LATE", f"window opened < {MIN_LEAD_S}s before t_end, or after the throw"),
    ("blind", "NOT_REAL_WINDOW", "outside the QB's sight cone at window_start"),
    ("closed", "NOT_REAL_WINDOW", f"sep_at_throw < {CLOSED_SEP} yd: not open at the throw"),
    ("missed", "OPEN_WINDOW", "in time, in sight, still open: QB passed him up"),
]
SUB_TO_B3 = {b: b3 for b, b3, _ in BUCKETS if b3}
IGNORED = [b for b, b3, _ in BUCKETS if b3]
BUCKET3 = ["TOO_LATE", "NOT_REAL_WINDOW", "OPEN_WINDOW"]


def load_routes(path: Path) -> pd.DataFrame:
    r = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in r.columns]
    if missing:
        sys.exit(f"{path} is missing required columns: {missing}")
    for c in OPTIONAL:
        if c not in r.columns:
            print(f"note: {path.name} has no {c} column; using the fallback for every route")
            r[c] = np.nan
    r = r[REQUIRED + OPTIONAL].copy()
    for c in ["is_target", "is_clear_open_downfield"]:
        r[c] = r[c].astype(str).str.lower().isin(["true", "1"])
    if r.duplicated(RKEYS).any():
        sys.exit(f"{path} has duplicate (gameId, playId, nflId) rows")
    return r


def apply_filters(r: pd.DataFrame) -> pd.DataFrame:
    n0, p0 = len(r), r[KEYS].drop_duplicates().shape[0]
    plays = load_plays()
    noplay = plays.loc[plays["playDescription"].str.contains("No Play", na=False), KEYS]
    r = r.merge(noplay.assign(_np=True), on=KEYS, how="left")
    n_np = int(r["_np"].eq(True).sum())
    r = r[~r["_np"].eq(True)].drop(columns="_np")
    r["position"] = r["position"].replace({"FB": "RB"})
    n_pos = int((~r["position"].isin(POSITIONS)).sum())
    r = r[r["position"].isin(POSITIONS)]
    print(f"filters: {n0:,} routes / {p0:,} plays in; dropped {n_np:,} 'No Play' routes and "
          f"{n_pos:,} non-WR/TE/RB routes; {len(r):,} routes / "
          f"{r[KEYS].drop_duplicates().shape[0]:,} plays left")
    return r.reset_index(drop=True)


def fallback_window_start(need: pd.DataFrame) -> pd.Series:
    """max(1.0, start of the final continuous stretch of sep >= 3 at or before t_end).
    NaN if he is never open before t_end. Missing sep counts as not open."""
    path = OUT / "openness_frames.csv"
    if not path.exists():
        sys.exit(f"{path} not found (needed for the window_start fallback). "
                 f"Regenerate with: python src/openness.py")
    f = pd.read_csv(path)
    f = f.merge(need[RKEYS + ["t_end"]], on=RKEYS)
    f = f[f["t"] <= f["t_end"] + 1e-9].sort_values(RKEYS + ["t"])
    t, op = f["t"], f["sep"] >= OPEN_SEP
    grp = [f[k] for k in RKEYS]
    last_open = t.where(op).groupby(grp).transform("max")
    last_closed = t.where(~op & (t < last_open)).groupby(grp).transform("max")
    start = t.where(op & (t > last_closed.fillna(-np.inf))).groupby(grp).min()
    return np.fmax(start, WINDOW_FLOOR_S).rename("ws_fallback")


def build(r: pd.DataFrame) -> pd.DataFrame:
    clock = pd.read_csv(OUT / "play_clock.csv").rename(columns={"t_end": "t_end_clock"})
    df = r.merge(clock, on=KEYS, how="inner")
    if len(df) < len(r):
        print(f"note: {len(r) - len(df):,} routes had no play_clock row and were dropped")

    # Opening time for buckets = window_start, with the openness-frames fallback.
    need = df["window_start"].isna()
    fb = fallback_window_start(df[need])
    df = df.join(fb, on=RKEYS)
    df["window_start_src"] = np.where(~need, "routes",
                                      np.where(df["ws_fallback"].notna(), "fallback", "none"))
    df["window_start"] = df["window_start"].fillna(df["ws_fallback"])
    df = df.drop(columns="ws_fallback")

    # Closed check: he wasn't actually open at the throw.
    df["window_closed"] = df["sep_at_throw"] < CLOSED_SEP

    # Sight at the frame where t == window_start, and at the throw (end_frame).
    sight = pd.read_csv(OUT / "qb_sight_frames.csv",
                        usecols=RKEYS + ["frameId", "angle_off_qb", "in_sight"])
    df["ws_frame"] = (df["snap_frame"] + (df["window_start"] * FPS).round()).astype("Int64")
    df = df.merge(sight.rename(columns={"frameId": "ws_frame"}), on=RKEYS + ["ws_frame"],
                  how="left")
    df = df.merge(sight.rename(columns={"frameId": "end_frame", "angle_off_qb": "angle_at_throw",
                                        "in_sight": "in_sight_at_throw"}),
                  on=RKEYS + ["end_frame"], how="left")
    for c in ["in_sight", "in_sight_at_throw"]:
        df[c] = df[c].astype("Int8")
    # window_start past the clock's end frame = he opened after the QB's window: late.
    df["ws_after_end"] = (df["ws_frame"] > df["end_frame"]).fillna(False).astype(bool)
    untracked = (df["window_start"].notna() & ~df["ws_after_end"]
                 & df["angle_off_qb"].isna()).sum()
    print(f"note: {df['ws_after_end'].sum():,} routes have window_start past the clock end "
          f"frame (counted late); {untracked:,} have no sight frame otherwise (not in sight)")

    # Window components (t_end from routes.csv, the same clock window_start was built on).
    has_ws = df["window_start"].notna()
    df["before_pressure"] = has_ws & (df["time_to_pressure"].isna()
                                      | (df["window_start"] < df["time_to_pressure"]))
    df["lead_ok"] = (has_ws & ~df["ws_after_end"]
                     & (df["t_end"] - df["window_start"] >= MIN_LEAD_S - 1e-9))
    df["sighted"] = df["in_sight"].fillna(0).astype(bool)
    df["in_window"] = (df["before_pressure"] & df["lead_ok"] & df["sighted"]
                       & ~df["window_closed"])

    # Buckets: clear-open downfield receivers on non-sack, non-scramble plays.
    df["analyzed"] = df["is_clear_open_downfield"] & ~df["passResult"].isin(NO_RATE)
    conds = [df["is_target"], has_ws & ~df["before_pressure"], ~df["lead_ok"],
             ~df["sighted"], df["window_closed"], pd.Series(True, index=df.index)]
    df["bucket"] = np.select([c.to_numpy(bool) for c in conds], [b for b, _, _ in BUCKETS],
                             default="")
    df.loc[~df["analyzed"], "bucket"] = ""
    df["bucket3"] = df["bucket"].map(SUB_TO_B3).fillna("")

    cols = ["gameId", "playId", "nflId", "displayName", "position", "alignment",
            "coverage_type", "is_clear_open_downfield", "is_target", "analyzed",
            "passResult", "playResult", "downfield_yds", "sep_at_throw", "proj_sep_0_7",
            "time_to_open_after_covered", "t_end", "end_event", "time_to_pressure",
            "n_rushers", "window_start", "window_start_src", "ws_frame", "angle_off_qb",
            "in_sight", "angle_at_throw", "in_sight_at_throw", "end_frame", "ws_after_end",
            "before_pressure", "lead_ok", "window_closed", "in_window", "bucket", "bucket3"]
    return df[cols]


def sight_check(df: pd.DataFrame) -> None:
    tg = df[df["is_target"] & df["end_event"].isin(["pass_forward", "autoevent_passforward"])]
    ok = tg["in_sight_at_throw"].notna()
    rate = tg.loc[ok, "in_sight_at_throw"].astype(bool).mean()
    print(f"\nSight check: {rate:.1%} of targeted receivers are in the QB's sight at the throw "
          f"(n={ok.sum():,}; {(~ok).sum()} without a sight frame)")
    if rate < INSIGHT_EXPECT:
        print(f"  FLAG: below the expected {INSIGHT_EXPECT:.0%}. Check the angle convention "
              f"or the QB orientation before trusting the 'blind' bucket.")


def headline(df: pd.DataFrame) -> None:
    co = df[df["analyzed"]]
    n = co[KEYS].drop_duplicates().shape[0]
    if len(co) != n:
        print(f"note: {len(co) - n} plays have more than one clear-open receiver")
    ign = co[~co["is_target"]]
    print("\n=== HEADLINE ===")
    print(f"{n:,} clear-open downfield plays (sacks/scrambles and 'No Play' excluded). "
          f"The clear-open receiver was ignored on {len(ign):,} ({len(ign) / max(n, 1):.1%}).")
    print("Why he was ignored:")
    b3 = ign["bucket3"].value_counts(normalize=True)
    sub = ign["bucket"].value_counts(normalize=True)
    for big in BUCKET3:
        print(f"  {big:<16}{b3.get(big, 0):6.1%}")
        for b, bb, desc in BUCKETS:
            if bb == big:
                print(f"    {b:<12}{sub.get(b, 0):6.1%}   {desc}")

    print("\nOutcomes on clear-open downfield plays:")
    for label, g in [("targeted", co[co["is_target"]]), ("ignored ", ign)]:
        att = g["passResult"].isin(["C", "I", "IN"])
        comp = (g["passResult"] == "C").sum() / max(att.sum(), 1)
        print(f"  {label}: n={len(g):,}  completion {comp:.1%}  "
              f"yards/play {g['playResult'].mean():.2f}")


def rate(df: pd.DataFrame) -> pd.DataFrame:
    """WR and TE only; percentiles are taken within position."""
    cell = ["position", "alignment", "coverage_type"]
    df = df[df["position"].isin(RATED_POSITIONS)]
    tt = df[df["time_to_open_after_covered"].notna()].copy()
    tt["expected"] = tt.groupby(cell)["time_to_open_after_covered"].transform("median")

    per = df.groupby("nflId").agg(displayName=("displayName", "first"),
                                  position=("position", lambda s: s.mode().iat[0]),
                                  n_routes=("in_window", "size"),
                                  window_rate=("in_window", "mean"))
    per = per.join(tt.groupby("nflId").agg(
        n_opened=("time_to_open_after_covered", "size"),
        median_ttoc=("time_to_open_after_covered", "median"),
        expected_ttoc=("expected", "median")))
    per["n_opened"] = per["n_opened"].fillna(0).astype(int)
    per = per[per["n_routes"] >= MIN_ROUTES].copy()
    per["ttoe"] = per["expected_ttoc"] - per["median_ttoc"]  # positive = opens faster
    by_pos = per.groupby("position")
    per["pct_window"] = by_pos["window_rate"].rank(pct=True)
    per["pct_ttoe"] = by_pos["ttoe"].rank(pct=True)
    per["clock_rating"] = 100 * per[["pct_window", "pct_ttoe"]].mean(axis=1, skipna=False)
    per = per.sort_values(["position", "clock_rating"], ascending=[False, False]).reset_index()
    return per.round({"window_rate": 4, "median_ttoc": 2, "expected_ttoc": 2, "ttoe": 2,
                      "pct_window": 4, "pct_ttoe": 4, "clock_rating": 1})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", type=Path, default=OUT / "routes.csv")
    args = ap.parse_args()
    if not args.routes.exists():
        sys.exit(f"{args.routes} not found")

    df = build(apply_filters(load_routes(args.routes)))
    df.to_csv(OUT / "buckets.csv", index=False, float_format="%.3f")
    src = df.loc[df["analyzed"], "window_start_src"].value_counts().to_dict()
    print(f"buckets.csv: {len(df):,} routes; clear-open window_start source {src}")

    sight_check(df)
    headline(df)

    rating = rate(df)
    rating.to_csv(OUT / "receiver_rating.csv", index=False)
    counts = rating["position"].value_counts().to_dict()
    print(f"\nreceiver_rating.csv: {len(rating):,} WR/TE with >= {MIN_ROUTES} routes {counts} "
          f"({rating['ttoe'].isna().sum()} without a ttoe, so no rating)")
    show = ["displayName", "n_routes", "window_rate", "ttoe", "pct_window", "pct_ttoe",
            "clock_rating"]
    for pos, k in [("WR", 10), ("TE", 5)]:
        print(f"\nTop {k} {pos}:")
        print(rating[rating["position"] == pos].head(k)[show].to_string(index=False))


if __name__ == "__main__":
    main()
