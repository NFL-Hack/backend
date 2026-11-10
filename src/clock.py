"""Pressure clock and QB line of sight.

Outputs
  outputs/play_clock.csv       one row per play
      gameId, playId, qb_nflId, snap_frame, end_frame, t_end, end_event,
      time_to_pressure, n_rushers
  outputs/qb_sight_frames.csv  one row per route runner per frame, snap..end_frame
      gameId, playId, nflId, frameId, t, angle_off_qb, in_sight

All times are seconds after the snap: t = (frameId - snap_frame) / 10.

Usage: python src/clock.py [--games N]   (--games limits files, for quick checks)
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import (END_EVENTS_AUTO, END_EVENTS_MANUAL, FPS, OUT, PRESSURE_RADIUS_YD,
                    SIGHT_HALF_ANGLE_DEG, SNAP_EVENTS, TRACKING, angle_diff_deg,
                    bearing_deg, load_scouting)

KEYS = ["gameId", "playId"]
FKEYS = KEYS + ["frameId"]


def _first_frame(ev: pd.DataFrame, names: list[str], after: pd.Series | None = None):
    """First frame per play whose event is in `names` (optionally strictly after `after`)."""
    e = ev[ev["event"].isin(names)]
    if after is not None:
        e = e.merge(after.rename("_after"), left_on=KEYS, right_index=True)
        e = e[e["frameId"] > e["_after"]]
    return e.groupby(KEYS)["frameId"].min()


def _end_frames(ev: pd.DataFrame, snap: pd.Series, last: pd.Series) -> pd.DataFrame:
    """End frame + end_event per play: first manual end event after the snap, else the
    pass autoevent, else the last tracked frame."""
    e = ev[ev["event"].isin(END_EVENTS_MANUAL + END_EVENTS_AUTO)]
    e = e.merge(snap.rename("snap_frame"), left_on=KEYS, right_index=True)
    e = e[e["frameId"] > e["snap_frame"]].copy()
    e["is_auto"] = e["event"].isin(END_EVENTS_AUTO)
    e = e.sort_values(KEYS + ["is_auto", "frameId"]).drop_duplicates(KEYS)
    e = e.set_index(KEYS)[["frameId", "event"]].rename(
        columns={"frameId": "end_frame", "event": "end_event"})
    out = pd.DataFrame({"snap_frame": snap}).join(last.rename("last_frame"))
    out = out.join(e)
    missing = out["end_frame"].isna()
    out.loc[missing, "end_frame"] = out.loc[missing, "last_frame"]
    out.loc[missing, "end_event"] = "tracking_end"
    out["end_frame"] = out["end_frame"].astype(int)
    return out.drop(columns="last_frame")


def process_game(path: str, roles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trk = pd.read_csv(path, usecols=["gameId", "playId", "nflId", "frameId", "team",
                                     "x", "y", "o", "event"])
    gid = int(trk["gameId"].iloc[0])
    roles = roles[roles["gameId"] == gid]

    # Events are repeated on every row of a frame; the ball rows carry one copy each.
    ev = trk.loc[trk["team"] == "football", FKEYS + ["event"]].dropna(subset=["event"])
    snap = _first_frame(ev, SNAP_EVENTS[:1])
    snap = snap.combine_first(_first_frame(ev, SNAP_EVENTS[1:]))
    last = trk.groupby(KEYS)["frameId"].max()
    clock = _end_frames(ev, snap, last)

    players = trk.dropna(subset=["nflId"]).astype({"nflId": "int64"})
    players = players.merge(roles[KEYS + ["nflId", "pff_role"]], on=KEYS + ["nflId"])
    players = players.merge(clock[["snap_frame", "end_frame"]], left_on=KEYS,
                            right_index=True)
    players = players[(players["frameId"] >= players["snap_frame"])
                      & (players["frameId"] <= players["end_frame"])]

    qb = players.loc[players["pff_role"] == "Pass", FKEYS + ["nflId", "x", "y", "o"]]
    qb = qb.rename(columns={"nflId": "qb_nflId", "x": "qb_x", "y": "qb_y", "o": "qb_o"})

    # --- time_to_pressure: first frame (strictly before end_frame) any rusher <= 2 yd ---
    rush = players[players["pff_role"] == "Pass Rush"]
    rush = rush[rush["frameId"] < rush["end_frame"]].merge(qb, on=FKEYS)
    rush["dist"] = np.hypot(rush["x"] - rush["qb_x"], rush["y"] - rush["qb_y"])
    pressured = rush[rush["dist"] <= PRESSURE_RADIUS_YD]
    clock = clock.join(pressured.groupby(KEYS)["frameId"].min().rename("pressure_frame"))
    clock["time_to_pressure"] = (clock["pressure_frame"] - clock["snap_frame"]) / FPS
    clock["t_end"] = (clock["end_frame"] - clock["snap_frame"]) / FPS
    clock = clock.join(qb.groupby(KEYS)["qb_nflId"].first())
    n_rush = roles[roles["pff_role"] == "Pass Rush"].groupby(KEYS).size()
    clock = clock.join(n_rush.rename("n_rushers"))
    clock["n_rushers"] = clock["n_rushers"].fillna(0).astype(int)
    clock = clock.reset_index()

    # --- line of sight for every route runner, snap..end_frame inclusive ---
    rr = players.loc[players["pff_role"] == "Pass Route",
                     FKEYS + ["nflId", "x", "y", "snap_frame"]].merge(qb, on=FKEYS)
    rr["t"] = (rr["frameId"] - rr["snap_frame"]) / FPS
    to_rec = bearing_deg(rr["x"] - rr["qb_x"], rr["y"] - rr["qb_y"])
    rr["angle_off_qb"] = angle_diff_deg(rr["qb_o"], to_rec).round(2)
    rr["in_sight"] = (rr["angle_off_qb"] <= SIGHT_HALF_ANGLE_DEG).astype("Int8")
    rr.loc[rr["angle_off_qb"].isna(), "in_sight"] = pd.NA  # missing QB orientation
    sight = rr[FKEYS + ["nflId", "t", "angle_off_qb", "in_sight"]]
    return clock, sight


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=None, help="only process first N games")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    files = sorted(TRACKING.glob("tracking_*.csv"))[: args.games]
    if not files:
        sys.exit(f"No tracking files found in {TRACKING}")
    roles = load_scouting()
    OUT.mkdir(exist_ok=True)

    t0 = time.time()
    clocks, sights = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(process_game, str(f), roles) for f in files]
        for i, fut in enumerate(futs, 1):
            c, s = fut.result()
            clocks.append(c)
            sights.append(s)
            if i % 20 == 0 or i == len(files):
                print(f"  {i}/{len(files)} games  ({time.time() - t0:.0f}s)")

    cols = ["gameId", "playId", "qb_nflId", "snap_frame", "end_frame", "t_end",
            "end_event", "time_to_pressure", "n_rushers"]
    clock = pd.concat(clocks, ignore_index=True)
    no_qb = clock["qb_nflId"].isna().sum()
    clock = clock.dropna(subset=["qb_nflId"]).astype({"qb_nflId": "int64"})[cols]
    clock.to_csv(OUT / "play_clock.csv", index=False)
    sight = pd.concat(sights, ignore_index=True)
    sight.to_csv(OUT / "qb_sight_frames.csv", index=False)

    print(f"play_clock.csv: {len(clock):,} plays ({no_qb} dropped: no tracked QB)")
    print(f"  pressured before end: {clock['time_to_pressure'].notna().mean():.1%}, "
          f"median time_to_pressure {clock['time_to_pressure'].median():.1f}s, "
          f"median t_end {clock['t_end'].median():.1f}s")
    print("  end_event:", clock["end_event"].value_counts().to_dict())
    print(f"qb_sight_frames.csv: {len(sight):,} rows, in_sight rate "
          f"{sight['in_sight'].mean():.1%}")


if __name__ == "__main__":
    main()
