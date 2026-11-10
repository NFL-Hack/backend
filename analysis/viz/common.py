"""Shared palette, data loaders and Kaplan-Meier pocket curves for all visuals."""
import os as _os
ANALYSIS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repo/analysis
import os
import sys

import numpy as np
import pandas as pd

ROOT = ANALYSIS
DATA = _os.environ.get("NFL_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(ANALYSIS)), "nfl-big-data-bowl-regional-event-data-main", "data"))
OPEN_DIR = f"{ROOT}/openness"
CLOCK_DIR = f"{ROOT}/clock"
OUT = os.path.dirname(os.path.abspath(__file__))
HORIZON = 0.7

# ------------------------------------------------------------------ palette (fixed across every visual)
POS_COL = {"WR": "#1f6fb4", "TE": "#8e44ad", "RB": "#e67e22"}
OFF_COL = "#1d3557"        # offense: navy discs
DEF_COL = "#f4f4f4"        # defense: white discs
OPEN_COL = "#2ca02c"       # open = green
COVER_COL = "#d62728"      # covered = red
MID_COL = "#f1c40f"        # in between
FIELD_COL = "#3a7d44"
INK = "#222222"
MUTED = "#777777"
FONT = "Helvetica, Arial, sans-serif"


def openness_color(o):
    """0 yds -> red, 1.5 -> yellow, 3+ -> green (hex)."""
    stops = [(0.0, (214, 39, 40)), (1.5, (241, 196, 15)), (3.0, (44, 160, 44))]
    o = float(np.clip(o, 0, 3))
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if o <= b:
            f = (o - a) / (b - a)
            c = [int(round(ca[i] + f * (cb[i] - ca[i]))) for i in range(3)]
            return "#%02x%02x%02x" % tuple(c)
    return "#2ca02c"


def load_openness():
    return pd.read_csv(f"{OPEN_DIR}/receiver_openness.csv")


def load_player_summary():
    return pd.read_csv(f"{OPEN_DIR}/player_summary.csv")


def clock_available():
    return all(os.path.exists(f"{CLOCK_DIR}/{f}") for f in ["routes_clock.csv", "player_clock.csv", "cache_clock.pkl"])


# ------------------------------------------------------------------ Kaplan-Meier pocket curves
def km(time, event):
    time, event = np.asarray(time, float), np.asarray(event, bool)
    ut = np.unique(time[event])
    n_at = np.array([(time >= u).sum() for u in ut])
    d = np.array([((time == u) & event).sum() for u in ut])
    return np.r_[0.0, ut], np.r_[1.0, np.cumprod(1 - d / n_at)]


def km_eval(curve, t):
    times, S = curve
    idx = np.searchsorted(times, np.asarray(t, float), side="right") - 1
    return S[np.clip(idx, 0, len(S) - 1)]


def km_median(curve):
    times, S = curve
    b = np.where(S <= 0.5)[0]
    return times[b[0]] if len(b) else np.nan


def pocket_plays():
    """Per-dropback pocket table rebuilt from the clock teammate's cache (same definitions as clock.py)."""
    _, pp = pd.read_pickle(f"{CLOCK_DIR}/cache_clock.pkl")
    plays = pd.read_csv(f"{DATA}/plays.csv")
    plays = plays[~plays.playDescription.str.contains("No Play", na=False)]
    pp = pp.merge(plays[["gameId", "playId", "pff_passCoverageType"]], on=["gameId", "playId"])
    press_ok = pp.press.notna() & (pp.press <= pp.end)
    sacked = pp.sack.notna() & (pp.sack <= pp.end)
    pp["pocket_event"] = press_ok | sacked
    ev = np.fmin(pp.press.where(press_ok), pp.sack.where(sacked))
    pp["pocket_t"] = (ev.fillna(pp.end) - pp.snap) / 10
    pp["blitz"] = pp.n_rush >= 5
    return pp


def pocket_curves():
    pp = pocket_plays()
    c = {"All": km(pp.pocket_t, pp.pocket_event),
         "Blitz": km(pp[pp.blitz].pocket_t, pp[pp.blitz].pocket_event),
         "No blitz": km(pp[~pp.blitz].pocket_t, pp[~pp.blitz].pocket_event)}
    for cov in ["Man", "Zone"]:
        s = pp[pp.pff_passCoverageType == cov]
        c[cov] = km(s.pocket_t, s.pocket_event)
    return c, pp
