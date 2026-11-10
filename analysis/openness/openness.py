"""The Open Man They Ignored -- receiver openness at the moment of the throw.

Run:  python openness.py            (all games)
      python openness.py --games 5  (quick test)

Openness (yards) = projected separation from the nearest defender HORIZON sec after
the throw, assuming every player keeps his current velocity (NGS dir: 0 deg = +y,
clockwise => vx = s*sin(dir), vy = s*cos(dir)). A defender closing fast shrinks it.
"""
import os as _os
ANALYSIS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repo/analysis
import argparse
import glob
import os
import re
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = _os.environ.get("NFL_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(ANALYSIS)), "nfl-big-data-bowl-regional-event-data-main", "data"))
OUT = os.path.dirname(os.path.abspath(__file__))
HORIZON = 0.7          # seconds of constant-velocity projection
CLEAR_MARGIN = 2.0     # yards more open than the 2nd most open route runner
MIN_ROUTES = 50
COLS = ["gameId", "playId", "nflId", "frameId", "team", "playDirection", "x", "y", "s", "dir", "event"]


# ----------------------------------------------------------------------------- tracking
def throw_frames_one_game(path):
    """Return the snap-frame ball position and all players at the throw frame for one game."""
    t = pd.read_csv(path, usecols=COLS)
    ev = t.loc[t.event.isin(["pass_forward", "autoevent_passforward"]), ["playId", "frameId", "event"]].drop_duplicates()
    ev["pri"] = (ev.event != "pass_forward").astype(int)     # prefer pass_forward
    ev = ev.sort_values(["playId", "pri", "frameId"]).drop_duplicates("playId")[["playId", "frameId"]]
    ev = ev.rename(columns={"frameId": "throwFrame"})
    # ball position at frame 1 = line of scrimmage
    ball0 = t[(t.team == "football") & (t.frameId == 1)][["playId", "x"]].rename(columns={"x": "los_x"})
    t = t.merge(ev, on="playId")
    snap = t[(t.frameId == t.throwFrame) & (t.team != "football")].drop(columns=["event"])
    return snap.merge(ball0, on="playId", how="left")


def load_throw_frames(n_games=None):
    cache = os.path.join(OUT, "cache_throw_frames.pkl")
    files = sorted(glob.glob(f"{DATA}/tracking/tracking_*.csv"))
    if n_games:
        files = files[:n_games]
    if os.path.exists(cache) and n_games is None:
        return pd.read_pickle(cache)
    with Pool(min(8, os.cpu_count())) as p:
        df = pd.concat(p.map(throw_frames_one_game, files), ignore_index=True)
    if n_games is None:
        df.to_pickle(cache)
    return df


def normalize(df):
    """Flip so offense always moves toward +x."""
    left = df.playDirection == "left"
    df.loc[left, "x"] = 120 - df.loc[left, "x"]
    df.loc[left, "y"] = 160 / 3 - df.loc[left, "y"]
    df.loc[left, "los_x"] = 120 - df.loc[left, "los_x"]
    df.loc[left, "dir"] = (df.loc[left, "dir"] + 180) % 360
    rad = np.radians(df["dir"])
    df["vx"] = df.s * np.sin(rad)
    df["vy"] = df.s * np.cos(rad)
    return df


# ----------------------------------------------------------------------------- targets
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


# ----------------------------------------------------------------------------- openness
def compute_openness(fr, plays, pff):
    fr = fr.merge(plays[["gameId", "playId", "possessionTeam", "defensiveTeam"]], on=["gameId", "playId"])
    role = pff[["gameId", "playId", "nflId", "pff_role", "pff_positionLinedUp"]]
    fr = fr.merge(role, on=["gameId", "playId", "nflId"], how="left")
    rec = fr[fr.pff_role == "Pass Route"]
    dfn = fr[fr.team == fr.defensiveTeam]
    m = rec.merge(dfn[["gameId", "playId", "nflId", "x", "y", "vx", "vy"]], on=["gameId", "playId"], suffixes=("", "_d"))
    dx, dy = m.x_d - m.x, m.y_d - m.y
    m["sep_now"] = np.hypot(dx, dy)
    # closing speed: defender velocity relative to receiver, projected onto the defender->receiver line
    rvx, rvy = m.vx_d - m.vx, m.vy_d - m.vy
    m["closing"] = -(rvx * dx + rvy * dy) / m.sep_now.clip(lower=0.1)
    m["sep_proj"] = np.hypot(dx + rvx * HORIZON, dy + rvy * HORIZON)
    keys = ["gameId", "playId", "nflId"]
    near_now = m.loc[m.groupby(keys).sep_now.idxmin(), keys + ["sep_now", "closing", "nflId_d"]]
    near_proj = m.groupby(keys).sep_proj.min().rename("openness").reset_index()
    out = rec.merge(near_now, on=keys).merge(near_proj, on=keys)
    out["depth"] = out.x - out.los_x
    out = out.rename(columns={"nflId_d": "nearestDefId"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=None)
    a = ap.parse_args()

    plays = pd.read_csv(f"{DATA}/plays.csv")
    plays = plays[~plays.playDescription.str.contains("No Play", na=False)]   # penalty-nullified plays
    pff = pd.read_csv(f"{DATA}/pffScoutingData.csv")
    players = pd.read_csv(f"{DATA}/players.csv")

    fr = normalize(load_throw_frames(a.games))
    print(f"throw frames: {fr[['gameId','playId']].drop_duplicates().shape[0]} plays")
    ro = compute_openness(fr, plays, pff)

    routes = pff[pff.pff_role == "Pass Route"][["gameId", "playId", "nflId"]]
    tg = match_targets(plays, routes, players)
    nplays = len(tg)
    named = tg.targetName.notna().sum()
    matched = tg.targetId.notna().sum()
    print(f"target parse: {named}/{nplays} plays name a receiver; matched {matched}/{named} = {matched/named:.1%}")
    unmatched = tg[tg.targetName.notna() & tg.targetId.isna()]
    print("unmatched examples:", unmatched.targetName.value_counts().head(10).to_dict())

    ro = ro.merge(tg, on=["gameId", "playId"], how="left")
    ro = ro.merge(plays[["gameId", "playId", "passResult", "playResult", "down", "yardsToGo",
                         "pff_passCoverage", "pff_passCoverageType"]], on=["gameId", "playId"])
    ro = ro.merge(players[["nflId", "displayName", "officialPosition"]], on="nflId", how="left")
    ro["targeted"] = ro.nflId == ro.targetId
    ro["play_has_target"] = ro.targetId.notna() & ro.targetId.isin(ro.nflId)   # target present at throw frame
    g = ro.groupby(["gameId", "playId"]).openness
    ro["rank_open"] = g.rank(ascending=False, method="first")
    ro["most_open"] = ro.rank_open == 1
    second = ro[ro.rank_open == 2].set_index(["gameId", "playId"]).openness.rename("second_open")
    ro = ro.join(second, on=["gameId", "playId"])
    ro["open_margin"] = np.where(ro.most_open, ro.openness - ro.second_open, np.nan)
    ro["clear_most_open"] = ro.most_open & (ro.open_margin >= CLEAR_MARGIN)
    ro["ignored_open"] = ro.clear_most_open & ~ro.targeted & ro.play_has_target
    # downfield variant: only route runners >= 5 yds past the LOS compete for "most open"
    dn = ro[ro.depth >= 5].copy()
    dn["r"] = dn.groupby(["gameId", "playId"]).openness.rank(ascending=False, method="first")
    sec = dn[dn.r == 2].set_index(["gameId", "playId"]).openness.rename("s2")
    dn = dn.join(sec, on=["gameId", "playId"])
    dn["clear_open_dn"] = (dn.r == 1) & ((dn.openness - dn.s2.fillna(0)) >= CLEAR_MARGIN)
    ro = ro.merge(dn[["gameId", "playId", "nflId", "clear_open_dn"]], on=["gameId", "playId", "nflId"], how="left")
    ro["clear_open_dn"] = ro.clear_open_dn.astype("boolean").fillna(False).astype(bool)
    ro["ignored_open_dn"] = ro.clear_open_dn & ~ro.targeted & ro.play_has_target
    ro["complete"] = ro.passResult == "C"
    ro["pos"] = ro.officialPosition.replace({"FB": "RB"})
    ro = ro[ro.pos.isin(["WR", "TE", "RB"])]
    ro.to_csv(os.path.join(OUT, "receiver_openness.csv"), index=False)

    # ---------------------------------------------------------------- per-play target vs most open
    tp = ro[ro.play_has_target]
    tgt = tp[tp.targeted]
    print(f"\nplays with matched target at throw frame: {tgt.shape[0]}")
    print(f"QB threw to the most-open route runner: {tp[tp.most_open].targeted.mean():.1%}")
    cl = tp[tp.clear_most_open]
    print(f"plays with a CLEAR most-open man (+{CLEAR_MARGIN}yd): {len(cl)}; he was targeted {cl.targeted.mean():.1%}")
    bins = [0, 1, 2, 3, 4, 5, 7, 10, 100]
    tgt = tgt.assign(obin=pd.cut(tgt.openness, bins))
    val = tgt.groupby("obin", observed=True).agg(n=("complete", "size"), comp=("complete", "mean"),
                                                 yds=("playResult", "mean"))
    print("\nVALIDATION: target openness vs outcome\n", val.round(3))
    val.to_csv(os.path.join(OUT, "validation_by_openness.csv"))

    # outcome on 'ignored' plays vs plays where the clear open man was hit
    cl_play = cl.assign(hit=cl.targeted)[["gameId", "playId", "hit"]]
    po = plays.merge(cl_play, on=["gameId", "playId"])
    po["complete"] = po.passResult == "C"
    print("\nclear-open plays: outcome when QB hit him vs threw elsewhere\n",
          po.groupby("hit").agg(n=("complete", "size"), comp=("complete", "mean"), yds=("playResult", "mean")).round(3))
    # what did the ignored man look like vs the actual target on those plays
    ig = tp[tp.ignored_open][["gameId", "playId", "openness", "depth"]]
    tt = tp[tp.targeted][["gameId", "playId", "openness", "depth"]]
    cmpd = ig.merge(tt, on=["gameId", "playId"], suffixes=("_ignored", "_target"))
    cd = tp[tp.clear_open_dn]
    print(f"DOWNFIELD (>=5yd) clear most-open man: {len(cd)} plays; targeted {cd.targeted.mean():.1%}")
    print("\nignored plays: ignored man vs actual target (medians)\n", cmpd.median(numeric_only=True).round(2))

    # ---------------------------------------------------------------- per player
    def agg(df):
        t = df[df.targeted]
        return pd.Series({
            "routes": len(df),
            "avg_openness": df.openness.mean(),
            "avg_sep_now": df.sep_now.mean(),
            "most_open_share": df.most_open.mean(),
            "clear_open_share": df.clear_most_open.mean(),
            "ignored_open_rate": df[df.play_has_target].ignored_open.mean(),
            "ignored_open_n": df.ignored_open.sum(),
            "ignored_open_dn_rate": df[df.play_has_target].ignored_open_dn.mean(),
            "target_rate": df[df.play_has_target].targeted.mean(),
            "targets": len(t),
            "comp_pct_when_targeted": t.complete.mean() if len(t) else np.nan,
            "yds_per_target": t.playResult.mean() if len(t) else np.nan,
            "avg_openness_when_targeted": t.openness.mean() if len(t) else np.nan,
            "avg_depth": df.depth.mean(),
        })
    ps = ro.groupby(["nflId", "displayName", "pos", "possessionTeam"]).apply(agg, include_groups=False).reset_index()
    ps = ps.sort_values("routes", ascending=False).drop_duplicates("nflId")   # keep main team
    ps = ps[ps.routes >= MIN_ROUTES].sort_values("ignored_open_rate", ascending=False)
    ps.to_csv(os.path.join(OUT, "player_summary.csv"), index=False)
    show = ["displayName", "possessionTeam", "routes", "avg_openness", "most_open_share", "ignored_open_rate",
            "ignored_open_dn_rate", "target_rate", "comp_pct_when_targeted", "yds_per_target"]
    for p in ["WR", "TE", "RB"]:
        s = ps[ps.pos == p]
        print(f"\n=== {p} (n={len(s)}) top ignored-open rate")
        print(s[show].head(8).round(3).to_string(index=False))
        print(f"--- {p} top avg openness")
        print(s.sort_values("avg_openness", ascending=False)[show].head(5).round(3).to_string(index=False))

    # ---------------------------------------------------------------- teams / QBs
    plays["qb"] = plays.playDescription.str.extract(r"\)\s*([A-Z][A-Za-z']*\.\s?[A-Za-z'\.\- ]+?) pass")[0]
    tq = cl.merge(plays[["gameId", "playId", "qb"]], on=["gameId", "playId"])
    for key in ["possessionTeam", "qb"]:
        s = tq.groupby(key).agg(clear_open_plays=("targeted", "size"), hit_rate=("targeted", "mean"))
        s = s[s.clear_open_plays >= 30].sort_values("hit_rate")
        print(f"\n{key}: hit rate on clear-open plays (lowest = ignores most)\n", s.head(6).round(3), "\n...\n", s.tail(3).round(3))
        s.to_csv(os.path.join(OUT, f"ignore_by_{key}.csv"))

    charts(ro, tgt, val, ps, fr)


# ----------------------------------------------------------------------------- charts
def charts(ro, tgt, val, ps, fr):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    # 1. validation
    fig, ax = plt.subplots(figsize=(8, 5))
    x = [f"{iv.left:g}-{iv.right:g}" if iv.right < 100 else f"{iv.left:g}+" for iv in val.index]
    ax.bar(x, val.comp * 100, color="#2a6f97")
    for i, (c, n) in enumerate(zip(val.comp, val.n)):
        ax.text(i, c * 100 + 1, f"{c:.0%}\nn={n}", ha="center", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(x, val.yds, color="#e76f51", marker="o", lw=2)
    ax2.set_ylabel("Yards per target", color="#e76f51")
    ax2.spines["top"].set_visible(False)
    ax.set_ylim(0, 100)
    ax.set_xlabel(f"Target's openness at throw (yds to nearest defender, projected {HORIZON}s ahead)")
    ax.set_ylabel("Completion %")
    ax.set_title("Openness at the throw predicts completion")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "01_validation_completion_vs_openness.png"), dpi=150); plt.close(fig)

    # 2. leaderboard scatter per position
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    colors = {"WR": "#2a6f97", "TE": "#6a994e", "RB": "#bc4749"}
    # WR/TE: downfield definition (>=5 yds past LOS); RB: all-route definition (checkdowns)
    metric = {"WR": "ignored_open_dn_rate", "TE": "ignored_open_dn_rate", "RB": "ignored_open_rate"}
    for ax, p in zip(axes, ["WR", "TE", "RB"]):
        s = ps[ps.pos == p]
        m = metric[p]
        ax.scatter(s.avg_openness, s[m] * 100, s=s.routes / 3, alpha=0.55, color=colors[p])
        lab = pd.concat([s[s.routes >= 100].nlargest(6, m), s.nlargest(3, "avg_openness")]).drop_duplicates("nflId")
        for r in lab.itertuples():
            ax.annotate(r.displayName, (r.avg_openness, getattr(r, m) * 100), fontsize=8,
                        xytext=(4, 3), textcoords="offset points")
        ax.axhline(s[m].mean() * 100, color="grey", ls="--", lw=0.8)
        ax.set_title(f"{p}s (>= {MIN_ROUTES} routes, n={len(s)})")
        ax.set_xlabel("Avg openness at throw (yds, dot size = routes)")
        ax.set_ylabel("% of routes: clearly most open DOWNFIELD (5+ yds), not targeted" if p != "RB"
                      else "% of routes: clearly most open on play, not targeted")
    fig.suptitle("The Open Man They Ignored: who gets open and doesn't get the ball", fontsize=14)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "02_leaderboard_ignored_vs_openness.png"), dpi=150); plt.close(fig)

    # 3. single play diagram: biggest openness gap between ignored man and actual target
    tp = ro[ro.play_has_target]
    ig = tp[tp.ignored_open & (tp.depth > 5)][["gameId", "playId", "nflId", "openness"]]
    tt = tp[tp.targeted][["gameId", "playId", "nflId", "openness", "passResult"]]
    c = ig.merge(tt, on=["gameId", "playId"], suffixes=("_i", "_t"))
    c = c[c.passResult != "C"]
    c["gap"] = c.openness_i - c.openness_t
    best = c.sort_values("gap", ascending=False).iloc[0]
    gid, pid = best.gameId, best.playId
    f = fr[(fr.gameId == gid) & (fr.playId == pid)]
    rp = ro[(ro.gameId == gid) & (ro.playId == pid)]
    off = rp.possessionTeam.iloc[0]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_facecolor("#e9f5e9")
    los = f.los_x.iloc[0]
    for xl in range(10, 111, 5):
        ax.axvline(xl, color="white", lw=0.8, zorder=0)
    ax.axvline(los, color="#1d3557", lw=1.5, ls="--", zorder=0)
    for r in f.itertuples():
        isoff = r.team == off
        ax.scatter(r.x, r.y, s=90, color="#1d3557" if isoff else "#e63946", zorder=3)
        ax.arrow(r.x, r.y, r.vx * HORIZON, r.vy * HORIZON, head_width=0.4, color="grey", zorder=2,
                 length_includes_head=True)
    names = rp.set_index("nflId")
    for nid, col, tag in [(best.nflId_i, "#f4a261", "MOST OPEN (ignored)"), (best.nflId_t, "#7209b7", "TARGETED")]:
        r = names.loc[nid]
        ax.scatter(r.x, r.y, s=500, facecolors="none", edgecolors=col, lw=3, zorder=4)
        ax.annotate(f"{tag}\n{r.displayName}: {r.openness:.1f} yds open", (r.x, r.y), xytext=(10, 12),
                    textcoords="offset points", fontsize=10, color=col, fontweight="bold")
    ax.set_xlim(min(f.x.min(), los) - 5, f.x.max() + 8); ax.set_ylim(0, 160 / 3)
    desc = rp.iloc[0]
    pl = pd.read_csv(f"{DATA}/plays.csv").set_index(["gameId", "playId"]).loc[(gid, pid), "playDescription"]
    ax.set_title(f"{off} vs {desc.defensiveTeam}, game {gid} play {pid} (offense moving ->)\n{pl[:110]}", fontsize=10)
    ax.set_xlabel("Field x (normalized, yds)"); ax.set_ylabel("y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "03_play_diagram.png"), dpi=150); plt.close(fig)
    print(f"\nplay diagram: game {gid} play {pid}: {pl}")


if __name__ == "__main__":
    main()
