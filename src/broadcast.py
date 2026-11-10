"""Broadcast graphics, NFL theme on white (static PNG, 200 dpi).

Usage:  python src/broadcast.py --tracking data/tracking [--play 2021101711,231] [--min-routes 100]
Reads outputs/routes.csv, play_clock.csv, buckets.csv, receiver_rating.csv plus raw tracking for the
freeze-frame. Saves figures/b1..b5.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle, Wedge

# NFL shield colors: navy + red on white, silver for neutral
BG, PANEL, INK, SUB = "#FFFFFF", "#F2F4F7", "#013369", "#5A6475"
OPENC, PRESS, DEF, GREY, LINE = "#013369", "#D50A0A", "#D50A0A", "#A5ACAF", "#C9CED6"
SRC = "Data: NFL Next Gen Stats + PFF, 2021 Wk 1-8"
OPEN_YD, DPI, KEY = 3.0, 200, ["gameId", "playId"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.weight": "bold", "text.color": INK, "axes.labelcolor": SUB,
                     "xtick.color": SUB, "ytick.color": INK, "axes.labelweight": "bold",
                     "figure.facecolor": BG, "axes.facecolor": BG})
last = lambda n: n.split()[-1].upper()
truthy = lambda s: s.astype(str).str.lower().isin(["true", "1", "1.0"])


def load(outputs):
    d = {n: pd.read_csv(os.path.join(outputs, n + ".csv")) for n in ["routes", "play_clock", "buckets", "receiver_rating"]}
    for n in ["routes", "buckets"]:
        for c in ["is_target", "is_clear_open_downfield"]:
            d[n][c] = truthy(d[n][c])
    return d


def play_frames(tracking_dir, gid, pid, pc):
    """Normalized tracking (offense moves +x) from the snap to the end frame."""
    t = pd.read_csv(os.path.join(tracking_dir, f"tracking_{gid}.csv"))
    g = t[(t.playId == pid) & (t.frameId >= pc.snap_frame) & (t.frameId <= pc.end_frame)].copy()
    left = g.playDirection == "left"
    g.loc[left, "x"] = 120 - g.loc[left, "x"]
    g.loc[left, "y"] = 53.3 - g.loc[left, "y"]
    g.loc[left, "o"] = (g.loc[left, "o"] + 180) % 360
    g["t"] = ((g.frameId - pc.snap_frame) / 10).round(1)
    return g


def save(fig, figures, name, n_text):
    fig.text(0.03, 0.025, f"{n_text}   |   {SRC}", color=SUB, fontsize=11, ha="left", va="bottom")
    path = os.path.join(figures, name)
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    print("saved", path)


def title(fig, text, sub=None, y=0.93, size=26):
    fig.text(0.03, y, text, color=INK, fontsize=size, ha="left", va="center")
    fig.add_artist(Line2D([0.03, 0.09], [y - 0.045, y - 0.045], color=PRESS, lw=5))   # red accent rule
    if sub:
        fig.text(0.03, y - 0.085, sub, color=SUB, fontsize=14, ha="left", va="center", fontweight="normal")


def legend(fig, handles, **kw):
    opts = dict(loc="lower right", bbox_to_anchor=(0.97, 0.012), ncol=len(handles), frameon=False, fontsize=12)
    opts.update(kw)
    for tx in fig.legend(handles=handles, **opts).get_texts():
        tx.set_color(INK)


def pick_play(d, n=5):
    b = d["buckets"]
    c = b[(b.bucket3 == "OPEN_WINDOW") & b.time_to_pressure.notna() & b.downfield_yds.between(8, 20) & (b.t_end < 4.5)]
    c = c.sort_values("sep_at_throw", ascending=False).head(n)
    print("EXAMPLE-PLAY CANDIDATES (use --play gameId,playId):")
    print(c[KEY + ["displayName", "position", "sep_at_throw", "downfield_yds", "window_start", "time_to_pressure", "t_end",
                   "passResult", "playResult"]].to_string(index=False))
    return int(c.gameId.iloc[0]), int(c.playId.iloc[0])


def play_ctx(d, play):
    gid, pid = play
    b = d["buckets"][(d["buckets"].gameId == gid) & (d["buckets"].playId == pid)].copy()
    pc = d["play_clock"][(d["play_clock"].gameId == gid) & (d["play_clock"].playId == pid)].iloc[0]
    ign = b[b.is_clear_open_downfield & ~b.is_target]
    hero = ign.iloc[0] if len(ign) else b.sort_values("sep_at_throw", ascending=False).iloc[0]
    clock = pc.time_to_pressure if pd.notna(pc.time_to_pressure) else pc.t_end
    return b, pc, hero, clock


# ---- 1. telestrator freeze-frame at window_start ----------------------------------------
def telestrator(d, play, tracking_dir, figures):
    b, pc, hero, clock = play_ctx(d, play)
    g = play_frames(tracking_dir, *play, pc)
    t0 = round(float(min(hero.window_start, g.t.max())), 1)
    f = g[g.t == t0]
    p = f[f.team != "football"]
    qb = p[p.nflId == pc.qb_nflId].iloc[0]
    off, dfn = p[p.team == qb.team], p[p.team != qb.team]
    me = p[p.nflId == hero.nflId].iloc[0]
    rusher = dfn.assign(dq=np.hypot(dfn.x - qb.x, dfn.y - qb.y)).sort_values("dq").iloc[0]
    los = g[(g.team == "football") & (g.t == 0)].x.iloc[0]

    x0 = max(0, min(p.x.min(), los) - 7)
    x1 = min(120, max(p.x.max() + 7, x0 + 62))
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.03, 0.13, 0.94, 0.66])
    ax.set_xlim(x0, x1)
    ax.set_ylim(0, 53.3)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.add_patch(Rectangle((x0, 0), x1 - x0, 53.3, facecolor=PANEL, edgecolor=INK, linewidth=2, zorder=0))
    for yl in range(10, 111):
        if not (x0 < yl < x1):
            continue
        if yl % 5 == 0:
            ax.plot([yl, yl], [0, 53.3], color=LINE, lw=1.5, zorder=1)
            if yl % 10 == 0 and 10 < yl < 110 and yl < x1 - 2:
                n = yl - 10 if yl <= 60 else 110 - yl
                for yy in (6, 47.3):
                    ax.text(yl, yy, str(n), color=LINE, fontsize=24, ha="center", va="center", zorder=1)
        else:
            for ya in (0.5, 23.2, 29.6, 52.0):
                ax.plot([yl, yl], [ya, ya + 0.8], color=LINE, lw=1, zorder=1)
    ax.plot([los, los], [0, 53.3], color=INK, lw=2.5, ls=(0, (6, 3)), zorder=2)

    # QB sight cone: +/- 90 deg around 'o' (0 deg = +y, 90 deg = +x)
    ax.add_patch(Wedge((qb.x, qb.y), 30, -qb.o, 180 - qb.o, facecolor=INK, alpha=0.07, edgecolor=INK, linewidth=1, zorder=3))
    for rad, al in ((3.0, 0.10), (2.3, 0.14), (1.7, 0.20), (1.25, 0.28)):
        ax.add_patch(Circle((me.x, me.y), rad, facecolor=OPENC, alpha=al, edgecolor="none", zorder=4))
    ax.add_patch(Circle((me.x, me.y), OPEN_YD, facecolor="none", edgecolor=OPENC, linewidth=3, linestyle="--", zorder=5))
    ax.scatter(dfn.x, dfn.y, s=300, color=DEF, edgecolor="white", linewidth=2, zorder=7)
    ax.scatter(off.x, off.y, s=300, color="white", edgecolor=INK, linewidth=2, zorder=7)
    ax.scatter([me.x], [me.y], s=380, color=OPENC, edgecolor="white", linewidth=2.5, zorder=8)
    ax.scatter([qb.x], [qb.y], s=380, color="white", edgecolor=INK, linewidth=3, marker="s", zorder=8)
    for row in pd.concat([off, dfn]).itertuples():
        if pd.notna(row.jerseyNumber):
            white_txt = row.team != qb.team or row.nflId == hero.nflId
            ax.text(row.x, row.y, str(int(row.jerseyNumber)), color="white" if white_txt else INK, fontsize=8.5,
                    ha="center", va="center", zorder=9)
    ax.add_patch(FancyArrowPatch((rusher.x, rusher.y), (qb.x, qb.y), arrowstyle="-|>", mutation_scale=30, color=PRESS,
                                 linewidth=4, shrinkA=8, shrinkB=10, zorder=12))
    lx = min(me.x + 9, x1 - 7)
    ly = me.y - 6 if me.y > 12 else me.y + 6
    ax.annotate(f"{last(hero.displayName)}  OPEN", (me.x, me.y), xytext=(lx, ly), color="white", fontsize=18, ha="center",
                va="center", zorder=10, arrowprops=dict(arrowstyle="-", color=OPENC, lw=2, shrinkB=16),
                bbox=dict(boxstyle="round,pad=0.3", facecolor=OPENC, edgecolor=OPENC, linewidth=2))

    word = "PRESSURE" if pd.notna(pc.time_to_pressure) else "THROW"
    for i, (lab, val, col) in enumerate([("GETS OPEN", f"{hero.window_start:.1f}s", OPENC), (word, f"{clock:.1f}s", PRESS),
                                         ("WINDOW", f"{clock - hero.window_start:+.1f}s", INK)]):
        cx = 0.52 + i * 0.155
        fig.add_artist(FancyBboxPatch((cx, 0.835), 0.14, 0.115, boxstyle="round,pad=0.004,rounding_size=0.012",
                                      transform=fig.transFigure, facecolor=BG, edgecolor=col, linewidth=3))
        fig.text(cx + 0.07, 0.91, val, color=col, fontsize=28, ha="center", va="center")
        fig.text(cx + 0.07, 0.856, lab, color=SUB, fontsize=11, ha="center", va="center")
    fig.text(0.03, 0.925, f"{last(hero.displayName)} IS OPEN.", fontsize=30, va="center")
    fig.add_artist(Line2D([0.03, 0.09], [0.885, 0.885], color=PRESS, lw=5))
    fig.text(0.03, 0.85, "In the quarterback's view, before the rush gets home.", fontsize=14,
             color=SUB, va="center", fontweight="normal")
    handles = [Line2D([0], [0], marker="o", color=BG, markerfacecolor="white", markeredgecolor=INK, markeredgewidth=2, markersize=13, label="Offense"),
               Line2D([0], [0], marker="s", color=BG, markerfacecolor="white", markeredgecolor=INK, markeredgewidth=2, markersize=13, label="QB"),
               Line2D([0], [0], marker="o", color=BG, markerfacecolor=DEF, markersize=13, label="Defense"),
               Line2D([0], [0], marker="o", color=BG, markerfacecolor=OPENC, markersize=13, label="Open receiver + 3-yd ring"),
               Patch(facecolor=INK, alpha=0.15, label="QB field of view"),
               Line2D([0], [0], color=PRESS, lw=4, label="Nearest rusher"),
               Line2D([0], [0], color=INK, lw=2.5, ls=(0, (4, 2)), label="Line of scrimmage")]
    legend(fig, handles, loc="lower center", bbox_to_anchor=(0.5, 0.058), fontsize=11.5, handletextpad=0.5, columnspacing=1.6)
    save(fig, figures, "b1_telestrator.png", f"n = 1 play, frozen {t0:.1f}s after the snap")


# ---- 2. race chart, lower third ---------------------------------------------------------
def lower_third(d, play, figures):
    b, pc, hero, clock = play_ctx(d, play)
    t_end = pc.t_end
    r = b.copy()
    r["never"] = r.window_start.isna() | (r.window_start >= t_end) | (r.sep_at_throw < OPEN_YD)
    r["start"] = np.where(r.never, t_end, r.window_start)
    r = r.sort_values(["start", "sep_at_throw"], ascending=[False, True]).reset_index(drop=True)
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_axes([0.2, 0.2, 0.76, 0.5])
    for i, row in r.iterrows():
        ax.barh(i, row.start, color=GREY, height=0.68, zorder=2)
        if not row.never:
            ax.barh(i, t_end - row.start, left=row.start, color=OPENC, height=0.68, zorder=2)
            wide = t_end - row.start > 0.75
            ax.text(row.start + 0.04 if wide else row.start - 0.04, i, f"OPEN {row.start:.1f}s", color="white" if wide else INK,
                    fontsize=13, va="center", ha="left" if wide else "right", zorder=4)
        else:
            ax.text(t_end + 0.06, i, "NOT OPEN", color=SUB, fontsize=13, va="center", zorder=4)
    ax.set_yticks(r.index)
    ax.set_yticklabels([last(n) + ("  ◀ TARGET" if tg else "") for n, tg in zip(r.displayName, r.is_target)], fontsize=17)
    if pd.notna(pc.time_to_pressure):
        ax.axvline(pc.time_to_pressure, color=PRESS, lw=6, zorder=5)
        ax.text(pc.time_to_pressure, len(r) - 0.42, "PRESSURE ", color=PRESS, fontsize=13, ha="right", va="bottom")
    ax.axvline(t_end, color="black", lw=4, zorder=5)
    ax.text(t_end, len(r) - 0.42, " THROW", color="black", fontsize=13, ha="left", va="bottom")
    ax.set_xlim(0, max(t_end, clock) + 0.55)
    ax.set_ylim(-0.6, len(r) - 0.1)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=13)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.grid(True, color=LINE, alpha=0.7, zorder=0)
    ax.set_xlabel("Seconds after the snap", fontsize=13)
    what = "PRESSURE" if pd.notna(pc.time_to_pressure) else "THE THROW"
    title(fig, f"{last(hero.displayName)} OPEN {clock - hero.window_start:.1f}s BEFORE {what}", y=0.9)
    legend(fig, [Patch(color=GREY, label="Covered"), Patch(color=OPENC, label="Open (3+ yd)"),
                 Line2D([0], [0], color=PRESS, lw=6, label="Pressure"), Line2D([0], [0], color="black", lw=4, label="Throw")])
    save(fig, figures, "b2_race_lower_third.png", f"n = {len(r)} receivers, 1 play")


# ---- 3. player cards --------------------------------------------------------------------
def player_cards(d, figures, min_routes):
    rr = d["receiver_rating"]
    rr = rr[rr.n_routes >= min_routes].copy()
    rr["pct_routes"] = rr.groupby("position").n_routes.rank(pct=True)
    bars = [("window_rate", "pct_window", "IN-TIME WINDOW RATE", lambda v: f"{v * 100:.0f}%"),
            ("ttoe", "pct_ttoe", "TIME TO OPEN VS EXPECTED", lambda v: f"{v:.2f}s faster" if v >= 0 else f"{-v:.2f}s slower"),
            ("n_routes", "pct_routes", "ROUTES", lambda v: f"{int(v)}")]
    groups = [(pos, name) for pos, name in (("WR", "WIDE RECEIVERS"), ("TE", "TIGHT ENDS"), ("RB", "RUNNING BACKS"))
              if (rr.position == pos).any()]
    H = 4.3 * len(groups) + 1.9
    fig = plt.figure(figsize=(16, H))
    top, bottom = 1 - 1.55 / H, 0.55 / H
    rowh = (top - bottom) / len(groups)
    for gi, (pos, name) in enumerate(groups):
        y0 = top - (gi + 1) * rowh
        fig.text(0.03, y0 + rowh - 0.012, name, color=SUB, fontsize=15, va="top")
        for ci, p in enumerate(rr[rr.position == pos].nlargest(3, "clock_rating").itertuples()):
            ax = fig.add_axes([0.03 + ci * 0.318, y0 + 0.012, 0.3, rowh - 0.065])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.04", facecolor=PANEL, edgecolor="none"))
            ax.add_patch(Rectangle((0, 0.94), 1, 0.06, facecolor=PRESS if ci == 0 else INK, edgecolor="none"))
            ax.text(0.06, 0.8, f"#{ci + 1}  {last(p.displayName)}", fontsize=21, va="center")
            ax.text(0.06, 0.66, p.displayName, fontsize=11, color=SUB, va="center", fontweight="normal")
            ax.text(0.94, 0.72, f"{p.clock_rating:.0f}", fontsize=58, color=INK, ha="right", va="center")
            ax.text(0.94, 0.52, "CLOCK RATING", fontsize=10, color=SUB, ha="right", va="center")
            for bi, (c, pc_col, lab, fmt) in enumerate(bars):
                yb = 0.4 - bi * 0.135
                ax.text(0.06, yb + 0.045, lab, fontsize=9.5, color=SUB, va="center")
                ax.text(0.94, yb + 0.045, fmt(getattr(p, c)), fontsize=11, ha="right", va="center")
                ax.add_patch(Rectangle((0.06, yb - 0.03), 0.88, 0.04, facecolor=LINE, edgecolor="none"))
                ax.add_patch(Rectangle((0.06, yb - 0.03), 0.88 * getattr(p, pc_col), 0.04, facecolor=INK, edgecolor="none"))
    fig.text(0.03, 1 - 0.45 / H, "WHO GETS OPEN IN TIME", fontsize=28, va="center")
    fig.add_artist(Line2D([0.03, 0.09], [1 - 0.78 / H, 1 - 0.78 / H], color=PRESS, lw=5))
    fig.text(0.03, 1 - 1.08 / H, "Top 3 by Clock Rating. Bars show rank within position.", fontsize=13, color=SUB, va="center",
             fontweight="normal")
    counts = ", ".join(f"{(rr.position == pos).sum()} {pos}" for pos, _ in groups)
    save(fig, figures, "b3_player_cards.png", f"n = {len(rr)} receivers with {min_routes}+ routes ({counts})")


# ---- 4. bucket waffle -------------------------------------------------------------------
def waffle(d, figures):
    b = d["buckets"].dropna(subset=["bucket3"])
    order = [("OPEN_WINDOW", "OPEN WINDOW, MISSED", "open before pressure, in sight, still open at the throw", OPENC),
             ("TOO_LATE", "TOO LATE", "pressure came first, or open under 0.4s before the throw", PRESS),
             ("NOT_REAL_WINDOW", "NOT A REAL WINDOW", "outside the QB's view, or the window closed", GREY)]
    share = np.array([(b.bucket3 == k).mean() * 100 for k, *_ in order])
    cells = np.floor(share).astype(int)
    for i in np.argsort(-(share - cells))[: 100 - cells.sum()]:
        cells[i] += 1
    colors = sum([[c] * n for (_, _, _, c), n in zip(order, cells)], [])
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.03, 0.1, 0.42, 0.66])
    for i, c in enumerate(colors):
        ax.add_patch(FancyBboxPatch((i % 10 + 0.08, 9 - i // 10 + 0.08), 0.84, 0.84, boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=c, edgecolor="none"))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    for i, ((_, lab, note, col), s) in enumerate(zip(order, share)):
        y = 0.66 - i * 0.2
        fig.text(0.5, y, f"{s:.0f}%", fontsize=72, color=col, va="center", ha="left")
        fig.text(0.68, y + 0.025, lab, fontsize=22, va="center")
        fig.text(0.68, y - 0.03, note, fontsize=12.5, color=SUB, va="center", fontweight="normal")
    title(fig, f"{share[0]:.0f}% OF IGNORED OPEN RECEIVERS WERE A REAL, MISSED WINDOW",
          "Throws where the clearly-open downfield receiver was not the target. One square = 1% of those plays.")
    save(fig, figures, "b4_bucket_waffle.png", f"n = {len(b):,} plays")


# ---- 5. validation ----------------------------------------------------------------------
def validation(d, figures):
    r = d["routes"]
    t = r[r.is_target & r.passResult.isin(["C", "I", "IN"])].copy()
    names = [f"{i}-{i + 1}" for i in range(10)] + ["10+"]
    t["bin"] = pd.cut(t.sep_at_throw, bins=list(range(11)) + [np.inf], labels=names, right=False)
    g = t.groupby("bin", observed=False).agg(n=("passResult", "size"), comp=("passResult", lambda s: (s == "C").mean() * 100))
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.07, 0.17, 0.9, 0.56])
    ax.bar(names, g.comp, color=[GREY] * 3 + [OPENC] * 8, width=0.74, zorder=2)
    for i, (c, n) in enumerate(zip(g.comp, g.n)):
        ax.text(i, c + 1.8, f"{c:.0f}%", ha="center", fontsize=20, zorder=3)
        ax.text(i, 3, f"n={n:,}", ha="center", fontsize=10.5, color="white", zorder=3)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="both", labelsize=14, length=0)
    ax.tick_params(axis="y", colors=SUB)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, color=LINE, alpha=0.7, zorder=0)
    ax.set_ylabel("Completion %", fontsize=14)
    ax.set_xlabel("Target's separation at the throw (yards from nearest coverage defender)", fontsize=14, labelpad=10)
    title(fig, f"MORE OPEN, MORE COMPLETIONS: {g.comp.iloc[0]:.0f}% AT 0-1 YD, {g.comp.iloc[-1]:.0f}% AT 10+ YD",
          "Completion rate on targeted throws, by how open the target was when the ball left the quarterback's hand.")
    legend(fig, [Patch(color=GREY, label="Covered (under 3 yd)"), Patch(color=OPENC, label="Open (3+ yd)")], fontsize=13)
    save(fig, figures, "b5_validation.png", f"n = {len(t):,} targeted throws")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--tracking", default="data/tracking")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--play", default=None, help="gameId,playId (default: 2nd-best auto candidate is printed, 1st is used)")
    ap.add_argument("--min-routes", type=int, default=100, help="minimum routes to appear on a player card")
    a = ap.parse_args()
    os.makedirs(a.figures, exist_ok=True)
    d = load(a.outputs)
    auto = pick_play(d)
    play = tuple(int(v) for v in a.play.split(",")) if a.play else auto
    for name, fn in [("telestrator", lambda: telestrator(d, play, a.tracking, a.figures)),
                     ("lower third", lambda: lower_third(d, play, a.figures)),
                     ("player cards", lambda: player_cards(d, a.figures, a.min_routes)),
                     ("waffle", lambda: waffle(d, a.figures)),
                     ("validation", lambda: validation(d, a.figures))]:
        try:
            fn()
        except Exception as e:
            print(f"FAILED {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
