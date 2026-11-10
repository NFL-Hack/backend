"""Clock Rating -- can a receiver get open before the pocket dies?

Run:  python clock.py            (all games, caches per-route/per-play frame results)
      python clock.py --games 5  (quick test, no cache)

time_to_open : seconds from snap until a receiver, AFTER first being covered (nearest defender
               <= 3 yds), reaches projected separation >= 3 yds (0.7 s constant-velocity projection,
               same as ../openness/openness.py) for 2 consecutive frames. Censored at the throw.
S_pocket(t)  : Kaplan-Meier P(no rusher within 2 yds of QB and no sack by t); censored at throw.
beat_clock   : S_pocket(time_to_open) using the play's rush-type curve (blitz = 5+ rushers);
               0 when covered and never open before the throw; NaN for free releases.
"""
import os as _os
ANALYSIS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repo/analysis
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, f"{ANALYSIS}/openness")
from openness import match_targets  # noqa: E402  (teammate's target parser, 99.7% match)

DATA = _os.environ.get("NFL_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(ANALYSIS)), "nfl-big-data-bowl-regional-event-data-main", "data"))
OUT = os.path.dirname(os.path.abspath(__file__))
HORIZON = 0.7        # s, projection (consistent with openness.py)
COVER_YDS = 3.0      # nearest defender within this => covered
OPEN_YDS = 3.0       # projected separation at/above this => open
PERSIST = 2          # frames open must persist (0.2 s) to suppress flicker
PRESS_YDS = 2.0      # rusher within this of QB => pocket broken
MIN_ROUTES = 50
COLS = ["gameId", "playId", "nflId", "frameId", "team", "s", "x", "y", "dir", "event"]
POS_COL = {"WR": "#2a6f97", "TE": "#6a994e", "RB": "#bc4749"}

_PFF = None


def _init():
    global _PFF
    p = pd.read_csv(f"{DATA}/pffScoutingData.csv", usecols=["gameId", "playId", "nflId", "pff_role"])
    _PFF = p


def first_event(ev, names):
    """First frame per play of the first-priority event name available."""
    e = ev[ev.event.isin(names)].copy()
    e["pri"] = e.event.map({n: i for i, n in enumerate(names)})
    e = e.sort_values(["playId", "pri", "frameId"]).drop_duplicates("playId")
    return e.set_index("playId").frameId


# ----------------------------------------------------------------------------- per game
def game_worker(path):
    t = pd.read_csv(path, usecols=COLS)
    gid = t.gameId.iloc[0]
    ev = t.loc[t.event.notna(), ["playId", "frameId", "event"]].drop_duplicates()
    fr = pd.DataFrame({"snap": first_event(ev, ["ball_snap", "autoevent_ballsnap"]),
                       "throw": first_event(ev, ["pass_forward", "autoevent_passforward"]),
                       "sack": first_event(ev, ["qb_sack", "qb_strip_sack"])})
    fr["last"] = t.groupby("playId").frameId.max()
    fr = fr[fr.snap.notna()]
    # window end: throw, else sack, else end of tracking (scrambles, etc.)
    fr["end"] = fr.throw.fillna(fr.sack).fillna(fr["last"])
    fr = fr.reset_index()

    t = t[t.nflId.notna()].merge(fr[["playId", "snap", "end"]], on="playId")
    t = t[(t.frameId >= t.snap) & (t.frameId <= t.end)]
    rad = np.radians(t["dir"].values)
    t["vx"], t["vy"] = t.s * np.sin(rad), t.s * np.cos(rad)   # verified NGS convention
    t = t.merge(_PFF[_PFF.gameId == gid].drop(columns="gameId"), on=["playId", "nflId"], how="left")

    k = ["playId", "frameId"]
    rec = t[t.pff_role == "Pass Route"][k + ["nflId", "x", "y", "vx", "vy", "snap", "end"]]
    dfn = t[t.pff_role.isin(["Coverage", "Pass Rush"])][k + ["x", "y", "vx", "vy"]]
    m = rec.merge(dfn, on=k, suffixes=("", "_d"))
    dx, dy = (m.x_d - m.x).values, (m.y_d - m.y).values
    rvx, rvy = (m.vx_d - m.vx).values, (m.vy_d - m.vy).values
    m["sep"] = np.hypot(dx, dy)
    m["proj"] = np.hypot(dx + rvx * HORIZON, dy + rvy * HORIZON)
    g = m.groupby(["playId", "nflId", "frameId"], sort=True).agg(sep=("sep", "min"), proj=("proj", "min"),
                                                                  snap=("snap", "first"), end=("end", "first"))
    g = g.reset_index()
    rk = ["playId", "nflId"]
    g["covered"] = g.sep <= COVER_YDS
    g["cov_before"] = g.groupby(rk).covered.transform(lambda s: s.shift(1, fill_value=False).cummax())
    op = g.proj >= OPEN_YDS
    persist = op.copy()
    for i in range(1, PERSIST):
        nxt = g.groupby(rk).proj.shift(-i)
        persist &= (nxt >= OPEN_YDS) | nxt.isna() & (g.frameId + i > g.end)  # allow at the very end
    g["open_ok"] = g.cov_before & persist
    routes = g.groupby(rk).agg(snap=("snap", "first"), end=("end", "first"),
                               sep_snap=("sep", "first"), ever_cov=("covered", "any"))
    routes["cov_frame"] = g[g.covered].groupby(rk).frameId.min()
    routes["open_frame"] = g[g.open_ok].groupby(rk).frameId.min()
    routes = routes.reset_index()
    routes.insert(0, "gameId", gid)

    # ---- pocket: rusher-to-QB distance
    qb = t[t.pff_role == "Pass"][k + ["x", "y"]].drop_duplicates(k)
    ru = t[t.pff_role == "Pass Rush"][k + ["x", "y"]]
    q = qb.merge(ru, on=k, suffixes=("", "_r"))
    q["d"] = np.hypot(q.x - q.x_r, q.y - q.y_r)
    dmin = q.groupby(k).d.min().reset_index()
    press = dmin[dmin.d <= PRESS_YDS].groupby("playId").frameId.min().rename("press")
    fr = fr.merge(press, left_on="playId", right_index=True, how="left")
    fr["n_rush"] = fr.playId.map(_PFF[(_PFF.gameId == gid) & (_PFF.pff_role == "Pass Rush")].groupby("playId").size())
    fr.insert(0, "gameId", gid)
    return routes, fr


def load(n_games=None):
    cache = os.path.join(OUT, "cache_clock.pkl")
    if n_games is None and os.path.exists(cache):
        return pd.read_pickle(cache)
    files = sorted(glob.glob(f"{DATA}/tracking/tracking_*.csv"))[: n_games or None]
    with Pool(min(8, os.cpu_count()), initializer=_init) as p:
        res = p.map(game_worker, files)
    out = (pd.concat([r[0] for r in res], ignore_index=True), pd.concat([r[1] for r in res], ignore_index=True))
    if n_games is None:
        pd.to_pickle(out, cache)
    return out


# ----------------------------------------------------------------------------- survival
def km(time, event):
    """Kaplan-Meier. Returns (times, S) step function: S[i] holds on [times[i], times[i+1])."""
    time, event = np.asarray(time, float), np.asarray(event, bool)
    ut = np.unique(time[event])
    n_at = np.array([(time >= u).sum() for u in ut])
    d = np.array([((time == u) & event).sum() for u in ut])
    S = np.cumprod(1 - d / n_at)
    return np.r_[0.0, ut], np.r_[1.0, S]


def km_eval(curve, t):
    times, S = curve
    idx = np.searchsorted(times, np.asarray(t, float), side="right") - 1
    return S[np.clip(idx, 0, len(S) - 1)]


def km_median(curve):
    times, S = curve
    below = np.where(S <= 0.5)[0]
    return times[below[0]] if len(below) else np.nan


# ----------------------------------------------------------------------------- logistic
def logit(X, y, iters=30):
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        b += np.linalg.solve(H, X.T @ (y - p))
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return b, se


# ----------------------------------------------------------------------------- labels
def repel_labels(ax, xs, ys, names, color, min_gap_frac=0.045):
    """Place labels to the right of points, nudging vertically so none overlap."""
    if not len(xs):
        return
    y0, y1 = ax.get_ylim()
    gap = (y1 - y0) * min_gap_frac
    order = np.argsort(ys)
    ly = np.array(ys, float)[order].copy()
    for _ in range(200):
        moved = False
        for i in range(1, len(ly)):
            if ly[i] - ly[i - 1] < gap:
                mid = (ly[i] + ly[i - 1]) / 2
                ly[i - 1], ly[i] = mid - gap / 2 - 1e-9, mid + gap / 2 + 1e-9
                moved = True
        if not moved:
            break
    top, bot = y1 - gap * 0.6, y0 + gap * 0.6       # keep labels inside the axes
    for i in range(len(ly) - 1, -1, -1):
        ly[i] = min(ly[i], top - (len(ly) - 1 - i) * gap)
    for i in range(len(ly)):
        ly[i] = max(ly[i], bot + i * gap)
    x0, x1 = ax.get_xlim()
    dxl = (x1 - x0) * 0.05
    for j, i in enumerate(order):
        ax.annotate(names[i], (xs[i], ys[i]), xytext=(xs[i] + dxl, ly[j]), fontsize=8, color="black",
                    va="center", arrowprops=dict(arrowstyle="-", color=color, lw=0.6, alpha=0.7))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=None)
    a = ap.parse_args()

    plays = pd.read_csv(f"{DATA}/plays.csv")
    pff = pd.read_csv(f"{DATA}/pffScoutingData.csv")
    players = pd.read_csv(f"{DATA}/players.csv")
    nullified = plays.playDescription.str.contains("No Play", na=False)
    plays = plays[~nullified]

    routes, pp = load(a.games)
    key = ["gameId", "playId"]
    pp = pp.merge(plays[key + ["pff_passCoverageType", "passResult"]], on=key)
    routes = routes.merge(pp[key], on=key)

    # ---------------- pocket clock
    pp["press_ok"] = pp.press.notna() & (pp.press <= pp.end)
    pp["sacked"] = pp.sack.notna() & (pp.sack <= pp.end)
    pp["pocket_event"] = pp.press_ok | pp.sacked
    ev_frame = np.fmin(pp.press.where(pp.press_ok), pp.sack.where(pp.sacked))
    pp["pocket_t"] = (ev_frame.fillna(pp.end) - pp.snap) / 10
    pp["blitz"] = pp.n_rush >= 5
    pp["t_throw"] = (pp.throw - pp.snap) / 10
    curves = {"All plays": km(pp.pocket_t, pp.pocket_event),
              "Blitz (5+ rushers)": km(pp[pp.blitz].pocket_t, pp[pp.blitz].pocket_event),
              "No blitz (<=4)": km(pp[~pp.blitz].pocket_t, pp[~pp.blitz].pocket_event)}
    for c in ["Man", "Zone"]:
        s = pp[pp.pff_passCoverageType == c]
        curves[c + " coverage"] = km(s.pocket_t, s.pocket_event)
    print(f"plays: {len(pp)}; pressure/sack observed {pp.pocket_event.mean():.1%}; "
          f"blitz rate {pp.blitz.mean():.1%}; median time to throw {pp.t_throw.median():.2f}s")
    pocket_stats = {}
    for n, c in curves.items():
        pocket_stats[n] = dict(median=km_median(c), S2=km_eval(c, 2.0), S25=km_eval(c, 2.5), S3=km_eval(c, 3.0))
        print(f"  {n:20s} KM median {pocket_stats[n]['median']:.1f}s  S(2.0)={pocket_stats[n]['S2']:.2f} "
              f"S(2.5)={pocket_stats[n]['S25']:.2f}  S(3.0)={pocket_stats[n]['S3']:.2f}")

    # ---------------- routes
    routes["free_release"] = ~routes.ever_cov
    routes["opened"] = routes.open_frame.notna()
    routes["time_to_cover"] = (routes.cov_frame - routes.snap) / 10
    routes["time_to_open"] = (routes.open_frame - routes.snap) / 10
    routes["window"] = (routes.end - routes.snap) / 10
    routes["tto_obs"] = routes.time_to_open.fillna(routes.window)          # for censored KM
    routes = routes.merge(pp[key + ["blitz", "pff_passCoverageType", "passResult", "throw"]], on=key)
    bc = np.where(routes.blitz, km_eval(curves["Blitz (5+ rushers)"], routes.tto_obs),
                  km_eval(curves["No blitz (<=4)"], routes.tto_obs))
    routes["beat_clock"] = np.where(routes.free_release, np.nan, np.where(routes.opened, bc, 0.0))
    routes = routes.merge(pff[["gameId", "playId", "nflId", "pff_positionLinedUp"]], on=key + ["nflId"], how="left")
    routes = routes.merge(players[["nflId", "displayName", "officialPosition"]], on="nflId", how="left")
    routes["pos"] = routes.officialPosition.replace({"FB": "RB"})
    routes = routes[routes.pos.isin(["WR", "TE", "RB"])]
    al = routes.pff_positionLinedUp.fillna("")
    routes["align"] = np.select([al.isin(["LWR", "RWR"]), al.str.startswith("S"), al.str.startswith("TE"),
                                 al.str.match(r"^(HB|FB)")], ["Wide", "Slot", "Inline TE", "Backfield"], "Other")

    # targets
    tg = match_targets(plays, pff[pff.pff_role == "Pass Route"][key + ["nflId"]], players)
    routes = routes.merge(tg[key + ["targetId"]], on=key, how="left")
    routes["targeted"] = routes.nflId == routes.targetId
    routes = routes.merge(plays[key + ["playResult", "possessionTeam"]], on=key, how="left")
    routes["complete"] = routes.passResult == "C"
    routes.drop(columns=["ever_cov"]).to_csv(os.path.join(OUT, "routes_clock.csv"), index=False)

    cov = routes[~routes.free_release]
    print(f"\nroutes: {len(routes)}; free releases {routes.free_release.mean():.1%}")
    print(routes.groupby("pos").agg(n=("nflId", "size"), free=("free_release", "mean")).round(3))
    print(cov.groupby("pos").agg(n=("nflId", "size"), open_pct=("opened", "mean"),
                                 med_tto=("time_to_open", "median"), beat=("beat_clock", "mean")).round(3))
    print(cov.groupby(["pos", "pff_passCoverageType"]).agg(n=("nflId", "size"), open_pct=("opened", "mean"),
                                                           med_tto=("time_to_open", "median"),
                                                           beat=("beat_clock", "mean")).round(3))
    print(cov.groupby(["pos", "align"]).agg(n=("nflId", "size"), open_pct=("opened", "mean"),
                                            med_tto=("time_to_open", "median"), beat=("beat_clock", "mean")).round(3))
    # censor-aware time to open by position
    tto_km = {p: km(cov[cov.pos == p].tto_obs, cov[cov.pos == p].opened) for p in ["WR", "TE", "RB"]}
    for p, c in tto_km.items():
        print(f"  {p} KM median time-to-open {km_median(c):.2f}s")

    # ---------------- players
    def agg(df):
        c = df[~df.free_release]
        t = df[df.targeted]
        out = {"routes": len(df), "free_release_pct": df.free_release.mean(), "covered_routes": len(c),
               "open_pct": c.opened.mean(), "median_tto": c.time_to_open.median(),
               "km_median_tto": km_median(km(c.tto_obs, c.opened)) if len(c) else np.nan,
               "beat_clock": c.beat_clock.mean(), "targets": len(t),
               "comp_pct_targeted": t.complete.mean() if len(t) else np.nan,
               "yds_per_target": t.playResult.mean() if len(t) else np.nan}
        for cv in ["Man", "Zone"]:
            s = c[c.pff_passCoverageType == cv]
            out[f"n_{cv.lower()}"] = len(s)
            out[f"open_pct_{cv.lower()}"] = s.opened.mean() if len(s) else np.nan
            out[f"beat_clock_{cv.lower()}"] = s.beat_clock.mean() if len(s) else np.nan
        out["main_align"] = df["align"].mode().iloc[0]
        return pd.Series(out)

    ps = routes.groupby(["nflId", "displayName", "pos"]).apply(agg, include_groups=False).reset_index()
    team = routes.groupby("nflId").possessionTeam.agg(lambda s: s.mode().iloc[0])
    ps["team"] = ps.nflId.map(team)
    ps = ps[ps.routes >= MIN_ROUTES].copy()
    ps["zone_minus_man"] = ps.beat_clock_zone - ps.beat_clock_man
    ps["clock_rank_in_pos"] = ps.groupby("pos").beat_clock.rank(ascending=False)
    ps = ps.sort_values(["pos", "beat_clock"], ascending=[True, False])
    ps.to_csv(os.path.join(OUT, "player_clock.csv"), index=False)
    show = ["displayName", "team", "routes", "free_release_pct", "open_pct", "median_tto", "beat_clock",
            "beat_clock_man", "beat_clock_zone", "comp_pct_targeted", "yds_per_target"]
    for p in ["WR", "TE", "RB"]:
        s = ps[ps.pos == p]
        print(f"\n=== {p} (n={len(s)}) top Beat-the-Clock")
        print(s[show].head(8).round(3).to_string(index=False))
        print("--- bottom")
        print(s[show].tail(4).round(3).to_string(index=False))
        z = s[(s.n_man >= 15) & (s.n_zone >= 15)]
        print("--- zone-only (biggest zone minus man)")
        print(z.sort_values("zone_minus_man", ascending=False)[show + ["n_man"]].head(5).round(3).to_string(index=False))
    corr = ps[["beat_clock", "comp_pct_targeted", "yds_per_target"]].corr().round(3)
    print("\nplayer-level corr\n", corr)

    # ---------------- validation (targeted receivers on plays with a throw)
    tv = routes[routes.targeted & routes.throw.notna()].copy()
    tv["cat"] = np.where(tv.free_release, "Free release",
                         np.where(~tv.opened, "Never open", pd.cut(tv.beat_clock, [0, .6, .8, .9, 1.01],
                                                                   labels=["<0.6", "0.6-0.8", "0.8-0.9", "0.9+"]).astype(str)))
    order = ["Never open", "<0.6", "0.6-0.8", "0.8-0.9", "0.9+", "Free release"]
    val = tv.groupby("cat").agg(n=("complete", "size"), comp=("complete", "mean"), yds=("playResult", "mean")).reindex(order)
    print("\nVALIDATION by target beat_clock\n", val.round(3))
    tb = tv[~tv.free_release]
    tbins = pd.cut(tb.tto_obs, [0, 1.5, 2.0, 2.5, 3.0, 3.5, 99], right=False)
    valt = tb[tb.opened].groupby(pd.cut(tb[tb.opened].time_to_open, [0, 1.5, 2.0, 2.5, 3.0, 3.5, 99], right=False),
                                 observed=True).agg(n=("complete", "size"), comp=("complete", "mean"), yds=("playResult", "mean"))
    print("\nVALIDATION by target time_to_open (opened only)\n", valt.round(3))
    b, se = logit(tb.beat_clock.values, tb.complete.values.astype(float))
    print(f"\nlogit complete ~ beat_clock (covered targets, n={len(tb)}): b={b[1]:.3f} se={se[1]:.3f} z={b[1]/se[1]:.1f}")
    b2, se2 = logit(tb.opened.values.astype(float), tb.complete.values.astype(float))
    print(f"logit complete ~ opened: b={b2[1]:.3f} se={se2[1]:.3f}")
    val.to_csv(os.path.join(OUT, "validation_by_beat_clock.csv"))

    # ---------------- charts
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    # pocket survival
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
    groups = [("All plays", ["All plays"], ["#333333"]),
              ("Blitz vs. no blitz", ["No blitz (<=4)", "Blitz (5+ rushers)"], ["#2a6f97", "#e76f51"]),
              ("Man vs. zone coverage", ["Zone coverage", "Man coverage"], ["#6a994e", "#9c27b0"])]
    for ax, (title, names, cols) in zip(axes, groups):
        for n, col in zip(names, cols):
            times, S = curves[n]
            ax.step(times, S, where="post", color=col, lw=2.2,
                    label=f"{n} (median {pocket_stats[n]['median']:.1f}s, S(2.5s)={pocket_stats[n]['S25']:.2f})")
        ax.axvline(2.5, color="grey", ls=":", lw=1)
        ax.set_xlim(0, 6); ax.set_ylim(0, 1.02)
        ax.set_title(title); ax.set_xlabel("Seconds after snap")
        ax.legend(loc="lower left", fontsize=9, frameon=False)
    axes[0].set_ylabel("P(pocket still intact)  [Kaplan-Meier]")
    fig.suptitle(f"The pocket clock: probability no rusher is within {PRESS_YDS:g} yds of the QB (and no sack) "
                 f"t seconds after the snap\n(plays without pressure are censored at the throw; n={len(pp)} dropbacks)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pocket_survival.png"), dpi=150); plt.close(fig)

    # leaderboard
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    for ax, p in zip(axes, ["WR", "TE", "RB"]):
        s = ps[ps.pos == p]
        sc = ax.scatter(s.median_tto, s.beat_clock, s=s.routes / 2.5, c=s.open_pct * 100, cmap="viridis",
                        alpha=0.8, edgecolor="white", lw=0.5, vmin=40, vmax=100)
        ax.axhline(s.beat_clock.mean(), color="grey", ls="--", lw=0.8)
        ax.axvline(s.median_tto.median(), color="grey", ls="--", lw=0.8)
        ax.set_xlim(s.median_tto.min() - 0.1, s.median_tto.max() + 0.6)
        yr = s.beat_clock.max() - s.beat_clock.min()
        ax.set_ylim(s.beat_clock.min() - 0.06 * yr, s.beat_clock.max() + 0.12 * yr)
        lab = pd.concat([s.nlargest(6, "beat_clock"), s.nsmallest(2, "beat_clock")]).drop_duplicates("nflId")
        repel_labels(ax, lab.median_tto.values, lab.beat_clock.values, lab.displayName.values, POS_COL[p])
        ax.set_title(f"{p}s (>= {MIN_ROUTES} routes, n={len(s)})")
        ax.set_xlabel("Median time to open (s after snap, routes that got open; lower = faster)")
        ax.set_ylabel("Beat-the-Clock score (mean P(pocket intact when open))")
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02); cb.set_label("% of covered routes he got open")
    fig.suptitle("Clock Rating: who gets open before the pocket collapses?  (dot size = routes; dashed = position average)",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "clock_leaderboard.png"), dpi=150); plt.close(fig)

    # validation
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    v = val.dropna()
    colors = ["#bc4749"] + ["#2a6f97"] * (len(v) - 2) + ["#999999"]
    ax.bar(v.index, v.comp * 100, color=colors)
    for i, (c, n, y) in enumerate(zip(v.comp, v.n, v.yds)):
        ax.text(i, c * 100 + 1.5, f"{c:.0%}\n{y:.1f} yds\nn={n}", ha="center", fontsize=9)
    ax.set_ylim(0, 105); ax.set_ylabel("Completion % when targeted")
    ax.set_xlabel("Targeted receiver's beat_clock (P(pocket intact when he got open))")
    ax.set_title("Targets who never got open: 43% completion vs ~70%+")
    ax = axes[1]
    x = [f"{iv.left:g}-{iv.right:g}s" if iv.right < 99 else f"{iv.left:g}s+" for iv in valt.index]
    ax.bar(x, valt.comp * 100, color="#2a6f97")
    for i, (c, n) in enumerate(zip(valt.comp, valt.n)):
        ax.text(i, c * 100 + 1.5, f"{c:.0%}\nn={n}", ha="center", fontsize=9)
    ax2 = ax.twinx(); ax2.plot(x, valt.yds, color="#e76f51", marker="o", lw=2)
    ax2.set_ylabel("Yards per target", color="#e76f51"); ax2.spines["top"].set_visible(False)
    ax.set_ylim(0, 105); ax.set_ylabel("Completion % when targeted")
    ax.set_xlabel("Targeted receiver's time to open (s after snap; opened routes only)")
    ax.set_title("Completion % holds until 3.5s+; later separation = deeper, bigger gains")
    fig.suptitle(f"Validation on {len(tv)} targeted routes. Logistic: complete ~ beat_clock, "
                 f"coef {b[1]:.2f} (z={b[1]/se[1]:.1f})", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "clock_validation.png"), dpi=150); plt.close(fig)
    print("\ncharts written")


if __name__ == "__main__":
    main()
