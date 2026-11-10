"""Pressure split for "The Open Man They Ignored".

Run:  python pressure.py      (~10 s; reuses ../openness/receiver_openness.csv and ../clock/cache_clock.pkl)

Per throw play:
  press_A : tracking pressure -- a 'Pass Rush' defender within 2 yds of the QB ('Pass') at or before the
            throw frame, or a sack at/before the throw (clock.py definition; frames from cache_clock.pkl).
  press_B : PFF pressure -- any player on the play with pff_hit / pff_hurry / pff_sack == 1.
  t_throw : (throw frame - snap frame) / 10 s.     blitz : 5+ pass rushers (PFF roles).
  clean_ontime : not A, not B, t_throw >= 2.5 s.
"""
import os as _os
ANALYSIS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repo/analysis
import os
import sys

import numpy as np
import pandas as pd

BASE = ANALYSIS
DATA = _os.environ.get("NFL_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(ANALYSIS)), "nfl-big-data-bowl-regional-event-data-main", "data"))
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{BASE}/clock")
from clock import load as load_clock  # noqa: E402  (returns cache_clock.pkl; recomputes from tracking if missing)

PRESS_YDS = 2.0       # documentation only: the cache was built with clock.PRESS_YDS = 2.0
ONTIME_S = 2.5
MIN_ROUTES = 50
MIN_QB = 40
K = ["gameId", "playId"]


def pct(x):
    return np.nan if len(x) == 0 else float(np.mean(x))


def spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def main():
    ro = pd.read_csv(f"{BASE}/openness/receiver_openness.csv")
    plays = pd.read_csv(f"{DATA}/plays.csv")
    pff = pd.read_csv(f"{DATA}/pffScoutingData.csv")

    # ------------------------------------------------------------------ baseline reproduction
    tp = ro[ro.play_has_target]
    cl = tp[tp.clear_most_open]
    cd = tp[tp.clear_open_dn]
    pr = plays.set_index(K)[["passResult", "playResult"]]
    cdp = cd.set_index(K)[["targeted"]].join(pr)
    print("BASELINE")
    print(f"  clear most-open (all depth): {len(cl)} plays, targeted {cl.targeted.mean():.1%}")
    print(f"  clear most-open downfield 5+: {len(cd)} plays, ignored {1 - cd.targeted.mean():.1%}")
    print(cdp.groupby("targeted").agg(n=("passResult", "size"), comp=("passResult", lambda s: (s == "C").mean()),
                                      yds=("playResult", "mean")).round(3).to_string())

    # ------------------------------------------------------------------ per-play labels
    _, pp = load_clock()
    pp = pp.copy()
    pp["t_throw"] = (pp.throw - pp.snap) / 10
    pp["press_A"] = (pp.press.notna() & (pp.press <= pp.throw)) | (pp.sack.notna() & (pp.sack <= pp.throw))
    pp["t_pressA"] = np.where(pp.press_A, (np.fmin(pp.press.where(pp.press <= pp.throw),
                                                    pp.sack.where(pp.sack <= pp.throw)) - pp.snap) / 10, np.nan)
    pp["blitz"] = pp.n_rush >= 5
    pp = pp[pp.throw.notna()]

    pb = pff.groupby(K).agg(pff_hit=("pff_hit", "max"), pff_hurry=("pff_hurry", "max"), pff_sack=("pff_sack", "max"))
    pb = pb.fillna(0).astype(int)
    pb["press_B"] = pb.max(axis=1) == 1

    # play-level openness facts (only plays with a matched target present at the throw frame are analysed)
    g = ro.groupby(K)
    play = g.agg(throwFrame=("throwFrame", "first"), possessionTeam=("possessionTeam", "first"),
                 play_has_target=("play_has_target", "first"), n_routes=("nflId", "size"),
                 most_open_openness=("openness", "max")).reset_index()
    tgt = ro[ro.targeted].drop_duplicates(K).set_index(K)
    play = play.join(tgt[["nflId", "openness", "depth", "pos"]].rename(
        columns={"nflId": "targetId", "openness": "target_openness", "depth": "target_depth", "pos": "target_pos"}), on=K)
    c1 = ro[ro.clear_most_open].set_index(K)
    play = play.join(c1[["nflId", "openness", "depth", "targeted"]].rename(
        columns={"nflId": "clear_open_id", "openness": "clear_open_openness", "depth": "clear_open_depth",
                 "targeted": "clear_open_hit"}), on=K)
    c2 = ro[ro.clear_open_dn].set_index(K)
    play = play.join(c2[["nflId", "openness", "depth", "targeted"]].rename(
        columns={"nflId": "clear_dn_id", "openness": "clear_dn_openness", "depth": "clear_dn_depth",
                 "targeted": "clear_dn_hit"}), on=K)
    play["has_clear_open"] = play.clear_open_id.notna()
    play["has_clear_dn"] = play.clear_dn_id.notna()
    for c in ["clear_open_hit", "clear_dn_hit"]:
        play[c] = play[c].astype("boolean")

    plays["qb"] = plays.playDescription.str.extract(r"\)\s*([A-Z][A-Za-z']*\.\s?[A-Za-z'\.\- ]+?) pass")[0]
    play = play.merge(plays[K + ["qb", "passResult", "playResult", "down", "yardsToGo", "pff_passCoverageType"]],
                      on=K, how="left")
    play = play.merge(pp[K + ["snap", "throw", "t_throw", "n_rush", "blitz", "press_A", "t_pressA"]], on=K, how="left")
    play = play.merge(pb.reset_index(), on=K, how="left")
    play["complete"] = play.passResult == "C"
    play["clean_A"] = play.press_A == False  # noqa: E712  (NaN-safe: missing clock -> False)
    play["clean_B"] = play.press_B == False  # noqa: E712
    play["clean_both"] = play.clean_A & play.clean_B
    play["press_either"] = (play.press_A == True) | (play.press_B == True)  # noqa: E712
    play["clean_ontime"] = play.clean_both & (play.t_throw >= ONTIME_S)
    play["has_clock"] = play.press_A.notna()
    play.to_csv(os.path.join(OUT, "play_pressure.csv"), index=False)

    # ------------------------------------------------------------------ sanity checks
    a = play[play.play_has_target & play.has_clock & play.press_B.notna()].copy()
    a["press_A"] = a.press_A.astype(bool)
    a["press_B"] = a.press_B.astype(bool)
    print(f"\nSANITY (throw plays with a matched target, n={len(a)}; all {len(play)} throw plays in openness file)")
    print(f"  pressured A (tracking): {a.press_A.mean():.1%}   pressured B (PFF): {a.press_B.mean():.1%}   "
          f"blitz: {a.blitz.mean():.1%}")
    ct = pd.crosstab(a.press_A, a.press_B, margins=True)
    print("  crosstab rows=A cols=B\n", ct.to_string())
    agree = (a.press_A == a.press_B).mean()
    po = ((a.press_A == a.press_B) & a.press_A).sum() / (a.press_A | a.press_B).sum()
    pe = a.press_A.mean() * a.press_B.mean() + (1 - a.press_A.mean()) * (1 - a.press_B.mean())
    kappa = (agree - pe) / (1 - pe)
    print(f"  agreement {agree:.1%}; Cohen kappa {kappa:.2f}; Jaccard (both | either) {po:.1%}; "
          f"P(B|A) {a[a.press_A].press_B.mean():.1%}; P(A|B) {a[a.press_B].press_A.mean():.1%}")
    for d in ["press_A", "press_B"]:
        s = a.groupby(d).t_throw.agg(["median", "mean", "size"])
        print(f"  time to throw by {d}:\n{s.round(2).to_string()}")
    print(f"  pressured-A: median time of pressure {a.t_pressA.median():.2f}s; "
          f"median gap pressure->throw {(a.t_throw - a.t_pressA).median():.2f}s")
    print(f"  pressure rate by blitz: A {a.groupby('blitz').press_A.mean().round(3).to_dict()} "
          f"B {a.groupby('blitz').press_B.mean().round(3).to_dict()}")
    print(f"  clean&on-time share: {a.clean_ontime.mean():.1%}; clean-both share {a.clean_both.mean():.1%}")
    sanity = pd.DataFrame({"metric": ["n_plays", "press_A_rate", "press_B_rate", "agree", "kappa", "jaccard",
                                      "P_B_given_A", "P_A_given_B", "tthrow_med_pressA", "tthrow_med_cleanA",
                                      "tthrow_med_pressB", "tthrow_med_cleanB"],
                           "value": [len(a), a.press_A.mean(), a.press_B.mean(), agree, kappa, po,
                                     a[a.press_A].press_B.mean(), a[a.press_B].press_A.mean(),
                                     a[a.press_A].t_throw.median(), a[~a.press_A].t_throw.median(),
                                     a[a.press_B].t_throw.median(), a[~a.press_B].t_throw.median()]})
    sanity.to_csv(os.path.join(OUT, "sanity_checks.csv"), index=False)

    # ------------------------------------------------------------------ buckets
    T = play[play.play_has_target]
    buckets = {
        "all": T.index == T.index,
        "clean_A": T.clean_A, "pressured_A": T.press_A == True,  # noqa: E712
        "clean_B": T.clean_B, "pressured_B": T.press_B == True,  # noqa: E712
        "clean_both": T.clean_both, "pressured_either": T.press_either,
        "clean_ontime": T.clean_ontime,
        "clean_both_quick_lt2.5": T.clean_both & (T.t_throw < ONTIME_S),
        "blitz": T.blitz == True, "no_blitz": T.blitz == False,  # noqa: E712
    }
    rows = []
    for name, m in buckets.items():
        b = T[np.asarray(m, bool)]
        c = b[b.has_clear_open]
        d = b[b.has_clear_dn]
        hit, els = d[d.clear_dn_hit == True], d[d.clear_dn_hit == False]  # noqa: E712
        ig = d[d.clear_dn_hit == False]  # noqa: E712
        r = dict(bucket=name, n_plays=len(b), share_of_plays=len(b) / len(T), median_t_throw=b.t_throw.median(),
                 comp_pct=b.complete.mean(), yds_play=b.playResult.mean(),
                 n_clear=len(c), pct_clear_targeted=pct(c.clear_open_hit.astype(bool)),
                 n_clear_dn=len(d), pct_clear_dn_targeted=pct(d.clear_dn_hit.astype(bool)),
                 dn_hit_n=len(hit), dn_hit_comp=pct(hit.complete), dn_hit_yds=hit.playResult.mean(),
                 dn_else_n=len(els), dn_else_comp=pct(els.complete), dn_else_yds=els.playResult.mean())
        r["dn_gap_comp"] = r["dn_hit_comp"] - r["dn_else_comp"]
        r["dn_gap_yds"] = r["dn_hit_yds"] - r["dn_else_yds"]
        r.update(med_target_open_all=b.target_openness.median(), med_most_open_all=b.most_open_openness.median(),
                 med_target_open_clear_ignored=c[c.clear_open_hit == False].target_openness.median(),  # noqa: E712
                 med_clear_open_ignored=c[c.clear_open_hit == False].clear_open_openness.median(),  # noqa: E712
                 med_target_open_dn_ignored=ig.target_openness.median(),
                 med_dn_open_ignored=ig.clear_dn_openness.median(),
                 med_dn_open_depth_ignored=ig.clear_dn_depth.median(), med_target_depth_dn_ignored=ig.target_depth.median())
        rows.append(r)
    bs = pd.DataFrame(rows)
    bs.to_csv(os.path.join(OUT, "bucket_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\nBUCKETS")
    print(bs[["bucket", "n_plays", "share_of_plays", "median_t_throw", "n_clear", "pct_clear_targeted", "n_clear_dn",
              "pct_clear_dn_targeted"]].round(3).to_string(index=False))
    print(bs[["bucket", "dn_hit_n", "dn_hit_comp", "dn_hit_yds", "dn_else_n", "dn_else_comp", "dn_else_yds",
              "dn_gap_comp", "dn_gap_yds"]].round(3).to_string(index=False))
    print(bs[["bucket", "med_target_open_all", "med_most_open_all", "med_target_open_clear_ignored",
              "med_clear_open_ignored", "med_target_open_dn_ignored", "med_dn_open_ignored",
              "med_dn_open_depth_ignored", "med_target_depth_dn_ignored"]].round(2).to_string(index=False))

    # ------------------------------------------------------------------ players
    rr = ro[ro.play_has_target].merge(play[K + ["clean_B", "clean_both"]], on=K)
    team = ro.groupby("nflId").possessionTeam.agg(lambda s: s.mode().iloc[0])
    nroutes = ro.groupby("nflId").size()  # same >=50 route gate as openness.py (all routes)

    def rates(df, tag):
        return df.groupby("nflId").agg(**{f"n_{tag}": ("ignored_open", "size"),
                                          f"ign_{tag}": ("ignored_open", "mean"),
                                          f"ign_dn_{tag}": ("ignored_open_dn", "mean")})
    ps = ro.drop_duplicates("nflId").set_index("nflId")[["displayName", "pos"]]
    ps["team"] = team
    ps["routes"] = nroutes
    ps = ps.join(rates(rr, "all")).join(rates(rr[rr.clean_B], "cleanB")).join(rates(rr[rr.clean_both], "cleanAB"))
    ps = ps[ps.routes >= MIN_ROUTES].reset_index()
    # leaderboard metric as in openness FINDINGS: WR/TE downfield variant, RB all-depth variant
    isrb = ps.pos == "RB"
    for t in ["all", "cleanB", "cleanAB"]:
        ps[f"metric_{t}"] = np.where(isrb, ps[f"ign_{t}"], ps[f"ign_dn_{t}"])
    for t in ["all", "cleanB", "cleanAB"]:
        ps[f"rank_{t}"] = ps.groupby("pos")[f"metric_{t}"].rank(ascending=False, method="min")
    ps = ps.sort_values(["pos", "metric_cleanB"], ascending=[True, False])
    ps.to_csv(os.path.join(OUT, "player_clean_ignored.csv"), index=False)
    topn = {"WR": 10, "TE": 5, "RB": 5}
    print("\nPLAYERS (metric: WR/TE = ignored while clearly most open 5+ yds downfield; RB = all depth; "
          "denominator = routes on plays with a matched target)")
    lb_rows = []
    for p, n in topn.items():
        s = ps[ps.pos == p]
        cols = ["displayName", "team", "routes", "n_all", "metric_all", "rank_all", "n_cleanB", "metric_cleanB",
                "rank_cleanB", "n_cleanAB", "metric_cleanAB", "rank_cleanAB"]
        print(f"\n=== {p} (n={len(s)}) top {n} by clean-pocket (B) rate")
        print(s.sort_values("metric_cleanB", ascending=False)[cols].head(n).round(3).to_string(index=False))
        for t in ["cleanB", "cleanAB"]:
            top_o = set(
                s.sort_values("metric_all", ascending=False).head(n).displayName)
            top_c = set(s.sort_values(f"metric_{t}", ascending=False).head(n).displayName)
            rho = spearman(s.metric_all, s[f"metric_{t}"])
            print(f"  {t}: Spearman vs original (all {len(s)} {p}s) = {rho:.2f}; "
                  f"drop out of top {n}: {sorted(top_o - top_c)}; new: {sorted(top_c - top_o)}")
            lb_rows.append(dict(pos=p, version=t, spearman=rho, dropped=sorted(top_o - top_c), new=sorted(top_c - top_o)))
    pd.DataFrame(lb_rows).to_csv(os.path.join(OUT, "leaderboard_changes.csv"), index=False)

    # ------------------------------------------------------------------ QBs / teams
    q = T[T.has_clear_open]
    out = []
    for key in ["qb", "possessionTeam"]:
        allr = q.groupby(key).clear_open_hit.agg(clear_plays_all="size", hit_rate_all="mean")
        for t in ["clean_B", "clean_both"]:
            s = q[q[t]].groupby(key).clear_open_hit.agg(**{f"n_{t}": "size", f"hit_{t}": "mean"})
            allr = allr.join(s)
        dn = T[T.has_clear_dn & T.clean_B].groupby(key).clear_dn_hit.agg(n_dn_clean_B="size", hit_dn_clean_B="mean")
        allr = allr.join(dn).reset_index().rename(columns={key: "name"})
        allr.insert(0, "level", key)
        out.append(allr)
    qc = pd.concat(out, ignore_index=True)
    for c in qc.columns[2:]:
        qc[c] = pd.to_numeric(qc[c])
    qc = qc[qc.n_clean_B >= MIN_QB].sort_values(["level", "hit_clean_B"])
    qc.to_csv(os.path.join(OUT, "qb_clean.csv"), index=False)
    print(f"\nQB / TEAM clean-pocket (B) hit rate on clear-open plays (>= {MIN_QB} such plays)")
    print(qc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
