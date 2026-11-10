"""Shared paths, constants, and the angle convention used by clock.py and combine.py."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"

# Raw Big Data Bowl data. Override with NFL_DATA_DIR if your copy lives elsewhere.
DATA = Path(
    os.environ.get(
        "NFL_DATA_DIR",
        REPO.parent / "nfl-big-data-bowl-regional-event-data-main" / "data",
    )
)
TRACKING = DATA / "tracking"

FPS = 10  # tracking is sampled at 10 Hz; t = (frameId - snap_frame) / FPS

# ---- Thresholds (assumptions, see README) ----
PRESSURE_RADIUS_YD = 2.0   # a Pass Rush player this close to the QB = pressure
SIGHT_HALF_ANGLE_DEG = 90  # receiver is "in sight" if within this many degrees of QB 'o'
MIN_LEAD_S = 0.4           # receiver must open at least this long before t_end
MIN_ROUTES = 30            # minimum routes for receiver_rating.csv

SNAP_EVENTS = ["ball_snap", "autoevent_ballsnap"]  # preference order
# Events that end the QB's decision window. Manual tags win over autoevents.
END_EVENTS_MANUAL = ["pass_forward", "qb_sack", "qb_strip_sack", "run",
                     "handoff", "lateral", "fumble"]
END_EVENTS_AUTO = ["autoevent_passforward"]


# ---- Angle convention ----
# Tracking 'o' and 'dir' are compass-style: 0 deg points along +y, angles increase
# clockwise, so 90 deg points along +x. Bearings computed from positions use the same
# convention: bearing(dx, dy) = atan2(dx, dy) in degrees, wrapped to [0, 360).
# Both live in raw field coordinates, so no playDirection flip is needed to compare them.
def bearing_deg(dx, dy):
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two compass angles, in [0, 180]."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def load_scouting() -> pd.DataFrame:
    return pd.read_csv(DATA / "pffScoutingData.csv",
                       usecols=["gameId", "playId", "nflId", "pff_role",
                                "pff_positionLinedUp"])


def load_plays() -> pd.DataFrame:
    return pd.read_csv(DATA / "plays.csv")


def load_players() -> pd.DataFrame:
    return pd.read_csv(DATA / "players.csv", usecols=["nflId", "officialPosition",
                                                      "displayName"])
