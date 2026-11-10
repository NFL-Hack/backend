"""Write a MOCK routes.csv with the contract columns so combine.py can run before the
real routes file lands. Values are random; nothing computed from it means anything.

Usage: python src/make_mock_routes.py   -> mock/routes.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUT, REPO, load_plays, load_scouting

SEED = 7


def main() -> None:
    rng = np.random.default_rng(SEED)
    clock = pd.read_csv(OUT / "play_clock.csv", usecols=["gameId", "playId", "t_end"])
    roles = load_scouting()
    r = roles.loc[roles["pff_role"] == "Pass Route", ["gameId", "playId", "nflId"]]
    r = r.merge(clock, on=["gameId", "playId"])

    n = len(r)
    opened = rng.random(n) < 0.7
    tto = np.round(rng.uniform(0.5, np.maximum(r["t_end"], 0.6)), 1)
    r["time_to_open"] = np.where(opened, tto, np.nan)
    r["open_on_arrival"] = (opened & (rng.random(n) < 0.6)).astype(int)
    r["clear_open"] = (opened & (rng.random(n) < 0.35)).astype(int)

    # One target per pass attempt, tilted toward clear-open receivers.
    plays = load_plays()
    attempts = plays.loc[plays["passResult"].isin(["C", "I", "IN"]), ["gameId", "playId"]]
    r["_w"] = rng.random(n) * np.where(r["clear_open"] == 1, 1.5, 1.0)
    r = r.merge(attempts.assign(_att=1), on=["gameId", "playId"], how="left")
    top = r[r["_att"] == 1].sort_values("_w").groupby(["gameId", "playId"]).tail(1).index
    r["targeted"] = 0
    r.loc[top, "targeted"] = 1

    out_dir = REPO / "mock"
    out_dir.mkdir(exist_ok=True)
    cols = ["gameId", "playId", "nflId", "time_to_open", "open_on_arrival",
            "clear_open", "targeted"]
    r[cols].to_csv(out_dir / "routes.csv", index=False)
    print(f"mock/routes.csv: {len(r):,} routes (MOCK DATA)")


if __name__ == "__main__":
    main()
