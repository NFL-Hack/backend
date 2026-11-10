"""Join routes + play_clock + qb_sight, assign buckets, print the headline, rate receivers.

Inputs
  routes.csv (contract below), outputs/play_clock.csv, outputs/qb_sight_frames.csv
Outputs
  outputs/buckets.csv           one row per route, with window flags and bucket
  outputs/receiver_rating.csv   one row per receiver with >= MIN_ROUTES routes

Usage
  python src/combine.py --routes mock/routes.csv      # now
  python src/combine.py --routes outputs/routes.csv   # once the real file lands
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (FPS, MIN_LEAD_S, MIN_ROUTES, OUT, load_players, load_plays,
                    load_scouting)

KEYS = ["gameId", "playId"]
RKEYS = KEYS + ["nflId"]

# routes.csv contract: one row per route runner per play.
#   time_to_open     seconds after snap the receiver first became open; blank if never
#   open_on_arrival  1 if still open when a ball thrown at time_to_open would arrive
#   clear_open       1 if this was a clear-open opportunity
#   targeted         1 if this receiver was the intended target
CONTRACT = ["gameId", "playId", "nflId", "time_to_open", "open_on_arrival",
            "clear_open", "targeted"]

# Bucket rules for a clear-open receiver, checked in this order; first match wins.
# All but "targeted" mean the receiver was ignored.
BUCKETS = [
    ("targeted", "QB threw to him"),
    ("pressured", "he opened at or after the first pressure"),
    ("late", f"he opened less than {MIN_LEAD_S}s before t_end"),
    ("blind", "he was outside the QB's sight cone when he opened"),
    ("closed", "he was no longer open when the ball would have arrived"),
    ("missed", "in sight, in time, no pressure, open on arrival: QB passed him up"),
]
IGNORED = [b for b, _ in BUCKETS if b != "targeted"]


def load_routes(path: Path) -> pd.DataFrame:
    r = pd.read_csv(path)
    missing = [c for c in CONTRACT if c not in r.columns]
    if missing:
        sys.exit(f"{path} is missing contract columns: {missing}")
    r = r[CONTRACT].copy()
    for c in ["open_on_arrival", "clear_open", "targeted"]:
        r[c] = r[c].fillna(0).astype(bool)
    dups = r.duplicated(RKEYS).sum()
    if dups:
        sys.exit(f"{path} has {dups} duplicate (gameId, playId, nflId) rows")
    bad = (r["clear_open"] & r["time_to_open"].isna()).sum()
    if bad:
        print(f"WARNING: {bad} clear_open routes have no time_to_open; bucketed 'late'")
    return r


def build(routes: pd.DataFrame) -> pd.DataFrame:
    clock = pd.read_csv(OUT / "play_clock.csv")
    df = routes.merge(clock, on=KEYS, how="inner")
    lost = len(routes) - len(df)
    if lost:
        print(f"note: {lost:,} routes had no play_clock row (no snap/QB) and were dropped")

    # Sight at the frame matching time_to_open.
    df["open_frame"] = (df["snap_frame"] + (df["time_to_open"] * FPS).round()).astype("Int64")
    sight = pd.read_csv(OUT / "qb_sight_frames.csv",
                        usecols=RKEYS + ["frameId", "angle_off_qb", "in_sight"])
    df = df.merge(sight.rename(columns={"frameId": "open_frame"}),
                  on=RKEYS + ["open_frame"], how="left")
    df["in_sight"] = df["in_sight"].astype("Int8")
    no_sight = (df["time_to_open"].notna() & df["angle_off_qb"].isna()).sum()
    if no_sight:
        print(f"note: {no_sight:,} opened routes had no sight frame at time_to_open "
              f"(after end_frame or untracked); treated as not in sight")

    # Context for buckets, rating, and outcomes.
    roles = load_scouting().rename(columns={"pff_positionLinedUp": "alignment"})
    df = df.merge(roles[RKEYS + ["alignment"]], on=RKEYS, how="left")
    df = df.merge(load_players().rename(columns={"officialPosition": "position"}),
                  on="nflId", how="left")
    plays = load_plays()[KEYS + ["pff_passCoverageType", "passResult", "playResult"]]
    df = df.merge(plays.rename(columns={"pff_passCoverageType": "coverage_type"}),
                  on=KEYS, how="left")

    # Window components.
    opened = df["time_to_open"].notna()
    df["before_pressure"] = opened & (df["time_to_pressure"].isna()
                                      | (df["time_to_open"] < df["time_to_pressure"]))
    df["lead_ok"] = opened & (df["t_end"] - df["time_to_open"] >= MIN_LEAD_S - 1e-9)
    df["sighted"] = df["in_sight"].fillna(0).astype(bool)  # no sight data = not in sight
    df["in_window"] = (df["before_pressure"] & df["lead_ok"] & df["sighted"]
                       & df["open_on_arrival"])

    # Buckets (clear-open routes only).
    conds = [
        df["targeted"],
        opened & ~df["before_pressure"],
        ~df["lead_ok"],
        ~df["sighted"],
        ~df["open_on_arrival"],
        pd.Series(True, index=df.index),
    ]
    df["bucket"] = np.select([c.to_numpy(bool) for c in conds], [b for b, _ in BUCKETS],
                             default="")
    df.loc[~df["clear_open"], "bucket"] = ""

    # Play-level unit for the headline: the earliest-opening clear-open receiver.
    co = df[df["clear_open"]].sort_values(KEYS + ["time_to_open", "nflId"])
    play_targeted = co.groupby(KEYS)["targeted"].transform("any")
    co = co.assign(play_co_targeted=play_targeted)
    primary = co[~co["play_co_targeted"]].drop_duplicates(KEYS).index
    df["headline_primary"] = df.index.isin(primary)
    df = df.merge(co.groupby(KEYS)["play_co_targeted"].first().reset_index(),
                  on=KEYS, how="left")
    df["play_co_targeted"] = df["play_co_targeted"].astype("boolean")  # NA: no clear-open

    cols = ["gameId", "playId", "nflId", "displayName", "position", "alignment",
            "coverage_type", "time_to_open", "open_on_arrival", "clear_open", "targeted",
            "snap_frame", "end_frame", "t_end", "end_event", "time_to_pressure",
            "n_rushers", "open_frame", "angle_off_qb", "in_sight", "before_pressure",
            "lead_ok", "in_window", "bucket", "headline_primary", "play_co_targeted",
            "passResult", "playResult"]
    return df[cols]


def headline(df: pd.DataFrame) -> None:
    plays = df[df["clear_open"]].groupby(KEYS).agg(
        targeted=("play_co_targeted", "first"),
        passResult=("passResult", "first"),
        playResult=("playResult", "first"),
    )
    plays["targeted"] = plays["targeted"].astype(bool)
    ignored = df[df["headline_primary"]]
    n_co, n_ign = len(plays), len(ignored)
    print("\n=== HEADLINE ===")
    print(f"{n_co:,} plays had a clear-open receiver; on {n_ign:,} "
          f"({n_ign / max(n_co, 1):.1%}) no clear-open receiver was targeted.")
    if n_ign:
        print("Why the clear-open receiver was ignored (earliest-opening one per play):")
        shares = ignored["bucket"].value_counts(normalize=True)
        desc = dict(BUCKETS)
        for b in IGNORED:
            print(f"  {b:<10}{shares.get(b, 0):6.1%}   {desc[b]}")

    print("\nOutcomes on clear-open plays:")
    for label, g in [("clear-open receiver targeted", plays[plays["targeted"]]),
                     ("clear-open receiver ignored ", plays[~plays["targeted"]])]:
        att = g["passResult"].isin(["C", "I", "IN"])
        comp = (g["passResult"] == "C").sum() / max(att.sum(), 1)
        print(f"  {label}: n={len(g):,}  completion% {comp:.1%} "
              f"(of {att.sum():,} attempts)  yards/play {g['playResult'].mean():.2f}")


def rate(df: pd.DataFrame) -> pd.DataFrame:
    cell = ["position", "alignment", "coverage_type"]
    opened = df[df["time_to_open"].notna()].copy()
    opened["expected_tto"] = opened.groupby(cell, dropna=False)["time_to_open"].transform(
        "median")

    per = df.groupby("nflId").agg(displayName=("displayName", "first"),
                                  position=("position", "first"),
                                  n_routes=("in_window", "size"),
                                  window_rate=("in_window", "mean"))
    tt = opened.groupby("nflId").agg(n_opened=("time_to_open", "size"),
                                     median_tto=("time_to_open", "median"),
                                     expected_tto=("expected_tto", "median"))
    per = per.join(tt)
    per["n_opened"] = per["n_opened"].fillna(0).astype(int)
    per = per[per["n_routes"] >= MIN_ROUTES].copy()
    per["ttoe"] = per["expected_tto"] - per["median_tto"]  # positive = opens faster
    per["pct_window"] = per["window_rate"].rank(pct=True)
    per["pct_ttoe"] = per["ttoe"].rank(pct=True)
    per["clock_rating"] = 100 * per[["pct_window", "pct_ttoe"]].mean(axis=1, skipna=False)
    per = per.sort_values("clock_rating", ascending=False).reset_index()
    return per.round({"window_rate": 4, "median_tto": 2, "expected_tto": 2, "ttoe": 2,
                      "pct_window": 4, "pct_ttoe": 4, "clock_rating": 1})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", type=Path, default=OUT / "routes.csv")
    args = ap.parse_args()
    if not args.routes.exists():
        sys.exit(f"{args.routes} not found. Use --routes mock/routes.csv until the real "
                 f"file lands (python src/make_mock_routes.py builds it).")
    is_mock = "mock" in str(args.routes).lower()
    if is_mock:
        print("*** RUNNING ON MOCK routes.csv: numbers below are NOT real ***")

    df = build(load_routes(args.routes))
    df.to_csv(OUT / "buckets.csv", index=False)
    print(f"buckets.csv: {len(df):,} routes, {df['clear_open'].sum():,} clear-open")

    headline(df)

    rating = rate(df)
    rating.to_csv(OUT / "receiver_rating.csv", index=False)
    print(f"\nreceiver_rating.csv: {len(rating):,} receivers with >= {MIN_ROUTES} routes")
    print(rating.head(10)[["displayName", "position", "n_routes", "window_rate", "ttoe",
                           "clock_rating"]].to_string(index=False))
    if is_mock:
        print("*** MOCK ***")


if __name__ == "__main__":
    main()
