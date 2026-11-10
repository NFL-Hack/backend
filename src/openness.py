"""Receiver openness pipeline (team-agreed shared definitions).

Run:  python src/openness.py --games 2   -> outputs/*_2games.csv
      python src/openness.py             -> outputs/routes.csv, outputs/openness_frames.csv

Self-contained: target parser (match_targets & helpers) copied from openness/openness.py.
Everything per frame / per route is vectorized (merge + groupby); the only Python loop
is match_targets' loop over plays (copied verbatim, ~8.5k plays, runs once).
"""
import argparse
import glob
import os
import re
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Raw data: override with NFL_DATA_DIR (same convention as common.py).
DATA = os.environ.get("NFL_DATA_DIR", os.path.join(os.path.dirname(REPO), "nfl-big-data-bowl-regional-event-data-main", "data"))
OUT = os.path.join(REPO, "outputs")
COLS = ["gameId", "playId", "nflId", "frameId", "team", "playDirection", "x", "y", "s", "o", "dir", "event"]

OPEN_SEP = 3.0          # OPEN = nearest Coverage defender >= 3.0 yd
BALL_SPEED = 20.0       # yd/s, flight_time = dist(QB, receiver) / 20
CLOSE_RATE = 7.0        # arrival radius = 7 * flight_time + 1.5
CLOSE_BASE = 1.5
DOWNFIELD_MIN = 5.0     # clear-open-downfield: >= 5 yd past LOS
CLEAR_MARGIN = 2.0      # ... and >= 2 yd more open than the next route runner

SNAP_EVENTS = ["ball_snap", "autoevent_ballsnap"]
PASS_EVENTS = ["pass_forward", "autoevent_passforward"]
OTHER_END = ["qb_sack", "qb_strip_sack", "run"]

ROUTE_COLS = ["gameId", "playId", "nflId", "displayName", "position", "alignment", "coverage_type", "t_end",
              "sep_at_throw", "downfield_yds", "time_to_open", "open_on_arrival", "is_target",
              "is_clear_open_downfield", "passResult", "playResult", "time_to_open_after_covered",
              "window_start", "proj_sep_0_7"]
FRAME_COLS = ["gameId", "playId", "nflId", "t", "sep"]
K = ["gameId", "playId", "nflId"]
WINDOW_FALLBACK = 1.0   # window_start floor (s) when the receiver never gets DOWNFIELD_MIN past the LOS
DROP_POS = ["QB", "C", "T"]
ORIG = os.path.join(REPO, "data_ref", "receiver_openness_ref.csv")   # original analysis: 0.7s projected openness + clear-open-downfield flag


# ----------------------------------------------------------------------------- tracking helpers
def normalize(t):
    """Shared definition: offense moves toward +x."""
    left = t.playDirection.values == "left"
    t["x"] = np.where(left, 120 - t.x, t.x)
    t["y"] = np.where(left, 53.3 - t.y, t.y)
    t["o"] = np.where(left, (t.o + 180) % 360, t.o)
    t["dir"] = np.where(left, (t.dir + 180) % 360, t.dir)
    return t


def first_event(ev, names):
    """First frame per play of the highest-priority event name present (names in priority order)."""
    e = ev[ev.event.isin(names)].copy()
    e["pri"] = e.event.map({n: i for i, n in enumerate(names)})
    return e.sort_values(["playId", "pri", "frameId"]).drop_duplicates("playId").set_index("playId").frameId


def play_frames(t):
    """snap frame, end frame and end_event ('pass' / 'sack' / 'run') per play."""
    ev = t.loc[t.event.notna() & (t.event != "None"), ["playId", "frameId", "event"]].drop_duplicates()
    snap = first_event(ev, SNAP_EVENTS).rename("snap")
    cand = pd.concat([
        first_event(ev, PASS_EVENTS).rename("f").to_frame().assign(kind="pass", pri=0),
        first_event(ev, ["qb_sack"]).rename("f").to_frame().assign(kind="sack", pri=1),
        first_event(ev, ["qb_strip_sack"]).rename("f").to_frame().assign(kind="sack", pri=2),
        first_event(ev, ["run"]).rename("f").to_frame().assign(kind="run", pri=3),
    ]).reset_index()
    pf = cand.merge(snap.reset_index(), on="playId")
    pf = pf[pf.f >= pf.snap]                                   # end must not precede the snap
    pf = pf.sort_values(["playId", "f", "pri"]).drop_duplicates("playId")   # earliest; ties -> pass first
    return pf.rename(columns={"f": "end", "kind": "end_event"})[["playId", "snap", "end", "end_event"]]


# ----------------------------------------------------------------------------- per game
_PFF = None
_DROP = None   # DataFrame of (gameId, playId, nflId) frame rows to drop (No Play plays, QB/C/T route runners)


def _init(pff, drop):
    global _PFF, _DROP
    _PFF = pff
    _DROP = drop


def game_worker(path):
    t = pd.read_csv(path, usecols=COLS)
    gid = int(t.gameId.iloc[0])
    t = normalize(t)
    pf = play_frames(t)
    n_plays_trk = t.playId.nunique()
    t = t.merge(pf, on="playId")
    t = t[(t.frameId >= t.snap) & (t.frameId <= t.end)]

    # line of scrimmage = ball x at the snap frame (normalized)
    los = t[(t.team == "football") & (t.frameId == t.snap)].groupby("playId").x.first().rename("los_x")

    pff = _PFF[_PFF.gameId == gid]
    pl = t[t.nflId.notna()].drop(columns=["event", "team", "playDirection"])
    pl = pl.merge(pff[["playId", "nflId", "pff_role"]], on=["playId", "nflId"], how="left")
    pl["t"] = (pl.frameId - pl.snap) / 10.0

    rec = pl[pl.pff_role == "Pass Route"]
    cov = pl[pl.pff_role == "Coverage"][["playId", "frameId", "x", "y"]]

    # ---- sep at every frame: receiver x Coverage defender, min distance
    m = rec[["playId", "nflId", "frameId", "x", "y"]].merge(cov, on=["playId", "frameId"], suffixes=("", "_d"))
    m["d"] = np.hypot(m.x_d.values - m.x.values, m.y_d.values - m.y.values)
    sep = m.groupby(["playId", "nflId", "frameId"], sort=False).d.min().rename("sep")
    rec = rec.join(sep, on=["playId", "nflId", "frameId"])        # NaN if no Coverage defender in frame
    rec = rec.sort_values(["playId", "nflId", "frameId"])
    rec["gameId"] = gid
    rec["nflId"] = rec.nflId.astype("int64")
    frames = rec[FRAME_COLS].copy()
    n_frames_raw = len(frames)
    d = _DROP[_DROP.gameId == gid]
    frames = frames[~pd.MultiIndex.from_frame(frames[["playId", "nflId"]]).isin(
        pd.MultiIndex.from_frame(d[["playId", "nflId"]]))]

    # ---- window_start = max(start of final open stretch ending at the end frame, first t >= 5 yd past LOS)
    rk = [rec.playId, rec.nflId]
    op = rec.sep >= OPEN_SEP                                   # NaN sep -> not open
    last_closed = rec.t.where(~op).groupby(rk).transform("max").fillna(-np.inf)
    a_start = rec.t.where(op & (rec.t > last_closed)).groupby(rk).min()
    open_end = op[rec.frameId == rec.end].groupby([rec.playId[rec.frameId == rec.end],
                                                   rec.nflId[rec.frameId == rec.end]]).any()
    a_start = a_start.where(open_end.reindex(a_start.index, fill_value=False))   # NaN unless open at end
    dfx = rec.x - rec.playId.map(los)
    b_dn = rec.t.where(dfx >= DOWNFIELD_MIN).groupby(rk).min()
    win = pd.DataFrame({"a": a_start}).join(b_dn.rename("b"))
    win["window_fallback"] = win.a.notna() & win.b.isna()
    win["window_start"] = np.where(win.b.notna(), np.fmax(win.a, win.b), np.fmax(win.a, WINDOW_FALLBACK))
    win.loc[win.a.isna(), "window_start"] = np.nan
    win.index.names = ["playId", "nflId"]

    # ---- time_to_open and time_to_open_after_covered
    is_open = rec.sep >= OPEN_SEP
    covered = (rec.sep < OPEN_SEP).astype(int)
    prior_cov = covered.groupby([rec.playId, rec.nflId]).cumsum() - covered   # covered at an earlier frame
    tto = rec.loc[is_open].groupby(["playId", "nflId"]).t.min().rename("time_to_open")
    ttoc = rec.loc[is_open & (prior_cov > 0)].groupby(["playId", "nflId"]).t.min().rename("time_to_open_after_covered")

    # ---- end-frame state
    end = rec[rec.frameId == rec.end][["playId", "nflId", "x", "y", "s", "dir", "t", "sep", "end_event"]]
    end = end.rename(columns={"t": "t_end", "sep": "sep_at_throw"}).join(los, on="playId")
    end["downfield_yds"] = end.x - end.los_x

    # ---- open on arrival (pass plays only)
    qb = pl[(pl.pff_role == "Pass") & (pl.frameId == pl.end)].groupby("playId")[["x", "y"]].first()
    a = end[end.end_event == "pass"].join(qb, on="playId", rsuffix="_qb")
    ft = np.hypot(a.x - a.x_qb, a.y - a.y_qb) / BALL_SPEED
    rad = np.radians(a.dir)
    a["px"] = a.x + a.s * np.sin(rad) * ft
    a["py"] = a.y + a.s * np.cos(rad) * ft
    a["radius"] = CLOSE_RATE * ft + CLOSE_BASE
    cov_end = pl[(pl.pff_role == "Coverage") & (pl.frameId == pl.end)][["playId", "x", "y"]]
    am = a[["playId", "nflId", "px", "py", "radius"]].merge(cov_end, on="playId")
    am["d"] = np.hypot(am.x - am.px, am.y - am.py)
    dmin = am.groupby(["playId", "nflId"]).d.min().rename("arr_d")
    a = a.join(dmin, on=["playId", "nflId"])
    ok = a.arr_d.notna() & a.radius.notna()
    a["open_on_arrival"] = pd.array(np.where(ok, a.arr_d > a.radius, False), dtype="boolean")
    a.loc[~ok, "open_on_arrival"] = pd.NA
    end = end.merge(a[["playId", "nflId", "open_on_arrival"]], on=["playId", "nflId"], how="left")

    # ---- route table: every pff 'Pass Route' on plays with a valid snap & end frame
    routes = pff[(pff.pff_role == "Pass Route") & pff.playId.isin(pf.playId)][["playId", "nflId", "pff_positionLinedUp"]]
    routes = routes.merge(pf[["playId", "end_event"]], on="playId")
    routes = routes.merge(end.drop(columns=["end_event"]), on=["playId", "nflId"], how="left")
    routes = routes.join(tto, on=["playId", "nflId"]).join(ttoc, on=["playId", "nflId"])
    routes = routes.join(win[["window_start", "window_fallback"]], on=["playId", "nflId"])
    routes["gameId"] = gid
    stats = {"plays_tracking": n_plays_trk, "plays_valid": len(pf),
             "plays_pff": pff.playId.nunique(),
             "frames_raw": n_frames_raw, "frames_kept": len(frames)}
    return frames, routes, stats


# ----------------------------------------------------------------------------- targets (copied from openness/openness.py)
END = re.compile(r"\s+(?:to|for|pushed|ran|caught|at)\s|\s*[\(\[,]|\.\s+(?:[A-Z]{2,}|Penalty|The|PENALTY)|\s+(?:INTERCEPTED|TOUCHDOWN)")
PASS = re.compile(r"pass (incomplete )?(short|deep) (left|middle|right)(?: intended for| to) ")
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\.?$", re.I)


def parse_desc(d):
    m = PASS.search(d or "")
    if not m:
        return None, None, None
    rest = d[m.end():]
    name = END.split(rest, maxsplit=1)[0].strip().rstrip(".").strip()
    return name, m.group(2), m.group(3)


def norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def split_abbrev(name):
    """'A.St. Brown' -> ('a', 'stbrown');  'D.Smith-Schuster' -> ('d', 'smithschuster')."""
    if "." not in name:
        return None, None
    first, last = name.split(".", 1)
    last = SUFFIX.sub("", last.strip()).strip()
    return norm(first), norm(last)


def player_keys(display):
    parts = display.split()
    first = norm(parts[0]) if parts else ""
    last = SUFFIX.sub("", " ".join(parts[1:])).strip()
    return first, norm(last)


def match_targets(plays, routes, players):
    keys = players.set_index("nflId").displayName.apply(player_keys)
    routes = routes.copy()
    routes["pfirst"] = routes.nflId.map(lambda i: keys.get(i, ("", ""))[0])
    routes["plast"] = routes.nflId.map(lambda i: keys.get(i, ("", ""))[1])
    by_play = {k: g for k, g in routes.groupby(["gameId", "playId"])}
    out = []
    for r in plays.itertuples():
        name, depth, side = parse_desc(r.playDescription)
        tid = np.nan
        if name:
            f, l = split_abbrev(name)
            g = by_play.get((r.gameId, r.playId))
            if g is not None and l:
                cand = g[g.plast == l]
                if len(cand) > 1:
                    cand = cand[cand.pfirst.str.startswith(f)]
                if len(cand) == 0:   # last-name prefix fallback (e.g. dropped suffix/odd spacing)
                    cand = g[g.plast.str.startswith(l) | pd.Series([l.startswith(x) and len(x) > 3 for x in g.plast], index=g.index)]
                    cand = cand[cand.pfirst.str.startswith(f)]
                if len(cand) == 1:
                    tid = cand.nflId.iloc[0]
        out.append((r.gameId, r.playId, name, depth, side, tid))
    return pd.DataFrame(out, columns=["gameId", "playId", "targetName", "passDepth", "passSide", "targetId"])


# ----------------------------------------------------------------------------- assemble
def assemble(routes, plays, players, pff):
    keys = ["gameId", "playId"]
    played = routes[keys].drop_duplicates()
    pl = plays.merge(played, on=keys)
    tg = match_targets(pl, pff[pff.pff_role == "Pass Route"][K], players)
    named = int(tg.targetName.notna().sum())
    matched = int(tg.targetId.notna().sum())

    r = routes.merge(tg[keys + ["targetId"]], on=keys, how="left")
    r["is_target"] = r.nflId == r.targetId
    r = r.merge(plays[keys + ["pff_passCoverageType", "passResult", "playResult"]], on=keys, how="left")
    r = r.merge(players[["nflId", "displayName", "officialPosition"]], on="nflId", how="left")
    r = r.rename(columns={"pff_positionLinedUp": "alignment", "pff_passCoverageType": "coverage_type",
                          "officialPosition": "position"})

    # v2: clear-open downfield = ORIGINAL 2,236-play definition, joined from openness/receiver_openness.csv:
    # clear_open_dn (most open by projected 0.7 s separation among receivers 5+ yd downfield, by 2+ yd over the
    # next downfield receiver) AND play_has_target. proj_sep_0_7 = the original `openness` column.
    orig = pd.read_csv(ORIG, usecols=K + ["openness", "clear_open_dn", "play_has_target"])
    orig["nflId"] = orig.nflId.astype("int64")
    orig["is_clear_open_downfield"] = orig.clear_open_dn.astype(bool) & orig.play_has_target.astype(bool)
    r["nflId"] = r.nflId.astype("int64")
    r = r.merge(orig[K + ["openness", "is_clear_open_downfield"]].rename(columns={"openness": "proj_sep_0_7"}),
                on=K, how="left")
    r["is_clear_open_downfield"] = r.is_clear_open_downfield.eq(True)
    info = {"named": named, "matched": matched, "plays": len(tg),
            "orig_clear_plays": int(orig[orig.is_clear_open_downfield][["gameId", "playId"]].drop_duplicates().shape[0])}

    # v2 filters: drop No Play plays, FB -> RB, drop QB/C/T route-runner rows
    info["rows_before"] = len(r)
    info["plays_before"] = r[keys].drop_duplicates().shape[0]
    info["clear_before"] = clear_counts(r)
    noplay = plays.loc[plays.playDescription.str.contains("No Play", na=False), keys]
    r = r.merge(noplay.assign(_np=True), on=keys, how="left")
    info["rows_noplay"] = int(r._np.eq(True).sum())
    r = r[~r._np.eq(True)].drop(columns="_np")
    r["position"] = r.position.replace({"FB": "RB"})
    info["rows_pos"] = int(r.position.isin(DROP_POS).sum())
    r = r[~r.position.isin(DROP_POS)]
    return r.sort_values(K)[ROUTE_COLS + ["end_event", "window_fallback"]].reset_index(drop=True), info


def clear_counts(r):
    """(plays with a clear-open-dn route, same among plays with a matched target, targeted share among those)."""
    keys = ["gameId", "playId"]
    ht = r.groupby(keys).is_target.transform("any")
    c = r[r.is_clear_open_downfield]
    ct = r[r.is_clear_open_downfield & ht]
    return (c[keys].drop_duplicates().shape[0], ct[keys].drop_duplicates().shape[0],
            float(ct.is_target.mean()) if len(ct) else np.nan)


def sanity(r, info, frames_n):
    rr = r
    print("\n==== v2 sanity ====")
    print(f"routes rows: before filters {info['rows_before']:,} ({info['plays_before']:,} plays); "
          f"No Play rows dropped {info['rows_noplay']:,}; QB/C/T rows dropped {info['rows_pos']:,}; "
          f"after {len(rr):,} ({rr[['gameId','playId']].drop_duplicates().shape[0]:,} plays)")
    print(f"frames rows: before {info['frames_raw']:,}, after {frames_n:,}")
    print("position value counts:", rr.position.value_counts(dropna=False).to_dict())
    w = rr.groupby("position").agg(n=("nflId", "size"), win_nonnull=("window_start", lambda s: s.notna().mean()),
                                   win_median=("window_start", "median"),
                                   fallback_n=("window_fallback", "sum"),
                                   fallback_pct_of_nonnull=("window_fallback", lambda s: s.sum() / max(1, s.size)))
    w["fallback_pct_of_nonnull"] = w.fallback_n / (w.win_nonnull * w.n).clip(lower=1)
    print("window_start by position:\n", w.round(3).to_string())
    print(f"window_start overall: non-null {rr.window_start.notna().mean():.1%}, median {rr.window_start.median():.2f}, "
          f"fallback used {int(rr.window_fallback.sum())} routes")
    cb = info["clear_before"]
    ca = clear_counts(rr)
    print(f"original clear-open-dn plays in receiver_openness.csv: {info['orig_clear_plays']}")
    print(f"clear-open-dn plays BEFORE filters: {cb[0]} (with matched target {cb[1]}), targeted {cb[2]:.1%}")
    print(f"clear-open-dn plays AFTER filters:  {ca[0]} (with matched target {ca[1]}), targeted {ca[2]:.1%}")
    pj = rr.proj_sep_0_7.notna()
    print(f"proj_sep_0_7 coverage: {pj.mean():.1%} of routes ({pj.sum():,}); pass plays "
          f"{rr[rr.end_event=='pass'].proj_sep_0_7.notna().mean():.1%}; "
          f"corr with sep_at_throw {rr.proj_sep_0_7.corr(rr.sep_at_throw):.3f}")
    print("==== legacy checks ====")
    print(f"\nroutes rows: {len(rr):,}   frames rows: {frames_n:,}   plays: {rr[['gameId','playId']].drop_duplicates().shape[0]:,}")
    print("end_event (plays):", rr.drop_duplicates(["gameId", "playId"]).end_event.value_counts().to_dict())
    print(f"target match rate: {info['matched']}/{info['named']} = {info['matched']/max(info['named'],1):.1%} "
          f"(plays naming a receiver, of {info['plays']} tracked plays)")
    print(f"routes ever open: {rr.time_to_open.notna().mean():.1%}")
    print(f"time_to_open == 0 (share of all routes): {(rr.time_to_open == 0).mean():.1%};  "
          f"of ever-open routes: {(rr.time_to_open == 0).sum()/rr.time_to_open.notna().sum():.1%}")
    print(f"routes ever covered then opened (after_covered notna): {rr.time_to_open_after_covered.notna().mean():.1%}")
    pos = rr.position
    tab = rr.assign(pos=pos).groupby("pos").agg(n=("nflId", "size"), tto_med=("time_to_open", "median"),
                                                 ttoc_med=("time_to_open_after_covered", "median"),
                                                 ever_open=("time_to_open", lambda s: s.notna().mean()),
                                                 snap_open=("time_to_open", lambda s: (s == 0).mean()))
    print("median time_to_open by position (raw / after-covered):\n", tab.round(3).to_string())
    pa = rr[rr.end_event == "pass"]
    print(f"\npass-play routes: downfield_yds > 0 at throw: {(pa.downfield_yds > 0).mean():.1%}; "
          f"median downfield {pa.downfield_yds.median():.2f}; targets median downfield {pa[pa.is_target].downfield_yds.median():.2f}")
    print(f"LOS missing (ball not tracked at snap): {rr.downfield_yds.isna().mean():.2%} of routes")
    oa = pa[pa.open_on_arrival.notna()]
    print(f"open_on_arrival rate: targets {oa[oa.is_target].open_on_arrival.astype(bool).mean():.1%} (n={oa.is_target.sum()}), "
          f"non-targets {oa[~oa.is_target].open_on_arrival.astype(bool).mean():.1%} (n={(~oa.is_target).sum()})")
    tt = oa[oa.is_target]
    comp = tt.groupby(tt.open_on_arrival.astype(bool)).passResult.agg(n="size", comp=lambda s: (s == "C").mean())
    print("targets: completion % by open_on_arrival\n", comp.round(3).to_string())
    print(f"open_on_arrival NA on pass plays: {pa.open_on_arrival.isna().mean():.2%}; "
          f"sep_at_throw NA: {rr.sep_at_throw.isna().mean():.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=None)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    suffix = f"_{a.games}games" if a.games else ""
    files = sorted(glob.glob(f"{DATA}/tracking/tracking_*.csv"))[: a.games]

    plays = pd.read_csv(f"{DATA}/plays.csv")
    pff = pd.read_csv(f"{DATA}/pffScoutingData.csv",
                      usecols=["gameId", "playId", "nflId", "pff_role", "pff_positionLinedUp"])
    players = pd.read_csv(f"{DATA}/players.csv")

    # frame rows to drop: route runners on No Play plays, and QB/C/T route runners
    rr_ = pff.loc[pff.pff_role == "Pass Route", K].dropna()
    noplay = plays.loc[plays.playDescription.str.contains("No Play", na=False), ["gameId", "playId"]]
    badpos = players.loc[players.officialPosition.isin(DROP_POS), "nflId"]
    drop = pd.concat([rr_.merge(noplay, on=["gameId", "playId"]), rr_[rr_.nflId.isin(badpos)]]).drop_duplicates()
    drop["nflId"] = drop.nflId.astype("int64")

    fpath = os.path.join(OUT, f"openness_frames{suffix}.csv")
    if os.path.exists(fpath):
        os.remove(fpath)
    routes, stats, nfr = [], [], 0
    with Pool(a.procs, initializer=_init, initargs=(pff, drop)) as pool:
        for i, (fr, ro, st) in enumerate(pool.imap(game_worker, files)):
            fr.to_csv(fpath, mode="a", header=(i == 0), index=False, float_format="%.3f")
            nfr += len(fr)
            routes.append(ro)
            stats.append(st)
    routes = pd.concat(routes, ignore_index=True)
    st = pd.DataFrame(stats).sum()
    print(f"plays: tracking {st.plays_tracking}, with valid snap+end {st.plays_valid}, pff {st.plays_pff}")

    r, info = assemble(routes, plays, players, pff)
    r[ROUTE_COLS].to_csv(os.path.join(OUT, f"routes{suffix}.csv"), index=False, float_format="%.3f")
    info["frames_raw"] = int(st.frames_raw)
    sanity(r, info, nfr)
    print(f"\nelapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
