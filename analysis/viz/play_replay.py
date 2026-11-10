"""Visual 1: animated play replay (snap -> throw) with openness rings and the pocket clock.

Outputs: play_replay.html (Plotly animation), play_replay.gif (matplotlib frames), play_replay_throw.png (still).
"""
import os as _os
ANALYSIS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repo/analysis
import os

import matplotlib
matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.patches import Circle, Rectangle

from common import (COVER_COL, DATA, DEF_COL, FIELD_COL, FONT, HORIZON, OFF_COL, OPEN_COL, OUT, km_eval,
                    load_openness, openness_color, pocket_curves)

# Hero play: PHI @ CAR, Wk 5. Clean pocket, 3.0 s: Hurts throws to a covered Ertz while Quez Watkins is ~10 yds open.
# (First choice CIN/DET 2021101704/1935 was dropped: Chase was on the sideline, y = 47.4, at the throw.)
GAME, PLAY = 2021101001, 2250
QB = "Hurts"
# label placement at the throw (dx, dy in yds, vertical anchor) so labels clear each other and nearby players
LABEL_POS = {"tgt": (-5.0, -3.4, "top"), "mo": (4.0, 3.6, "bottom")}
PRESSURE_CSV = f"{ANALYSIS}/pressure/play_pressure.csv"
POST_THROW = 0                          # animation runs snap -> throw
W = 160 / 3


def play_context(plays):
    """Week, pressure flags and title/subtitle text for the hero play."""
    g = pd.read_csv(f"{DATA}/games.csv").set_index("gameId").loc[GAME]
    pr = pd.read_csv(PRESSURE_CSV)
    pr = pr[(pr.gameId == GAME) & (pr.playId == PLAY)].iloc[0]
    where = "at" if g.homeTeamAbbr == plays.defensiveTeam else "vs"
    ordn = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[int(plays.down)]
    res = {"C": f"complete for {plays.playResult} yds", "I": "incomplete", "IN": "intercepted"}.get(plays.passResult, plays.passResult)
    sub = (f"{plays.possessionTeam} {where} {plays.defensiveTeam}, Week {g.week} 2021 · {ordn} & {plays.yardsToGo} · "
           f"{plays.pff_passCoverage} · {res}")
    return dict(clean=bool(pr.clean_both), t_throw=float(pr.t_throw), sub=sub)


def hero_title(ctx, mo):
    lead = f"Clean pocket, {ctx['t_throw']:.1f} seconds. " if ctx["clean"] else ""
    return f"{lead}{mo.displayName} was {mo.openness:.1f} yards open. {QB} threw to the covered man."


def closing_defender(t, throw, tgt_id):
    """Defender whose projected position (0.7 s ahead) is nearest the target's: the one shrinking his openness."""
    th = t[t.frameId == throw]
    r = th[th.nflId == tgt_id].iloc[0]
    d = th[th.side == "def"].copy()
    d["proj"] = np.hypot(d.x - r.x + (d.vx - r.vx) * HORIZON, d.y - r.y + (d.vy - r.vy) * HORIZON)
    return d.sort_values("proj").iloc[0], r


# ------------------------------------------------------------------ data
def load_play():
    t = pd.read_csv(f"{DATA}/tracking/tracking_{GAME}.csv")
    t = t[t.playId == PLAY].copy()
    plays = pd.read_csv(f"{DATA}/plays.csv").set_index(["gameId", "playId"]).loc[(GAME, PLAY)]
    pff = pd.read_csv(f"{DATA}/pffScoutingData.csv")
    pff = pff[(pff.gameId == GAME) & (pff.playId == PLAY)][["nflId", "pff_role"]]
    players = pd.read_csv(f"{DATA}/players.csv")[["nflId", "displayName", "officialPosition"]]
    ev = t[t.event.notna() & (t.event != "None")].drop_duplicates("frameId").set_index("event").frameId
    snap = ev.get("ball_snap", ev.get("autoevent_ballsnap"))
    throw = ev.get("pass_forward", ev.get("autoevent_passforward"))
    los = t[(t.team == "football") & (t.frameId == 1)].x.iloc[0]
    if t.playDirection.iloc[0] == "left":
        t["x"], t["y"], t["dir"] = 120 - t.x, W - t.y, (t["dir"] + 180) % 360
        los = 120 - los
    rad = np.radians(t["dir"])
    t["vx"], t["vy"] = t.s * np.sin(rad), t.s * np.cos(rad)
    t = t.merge(pff, on="nflId", how="left").merge(players, on="nflId", how="left")
    t = t[(t.frameId >= snap) & (t.frameId <= throw + POST_THROW)]
    off = plays.possessionTeam
    t["side"] = np.where(t.team == "football", "ball", np.where(t.team == off, "off", "def"))
    # openness per frame (identical definition to openness.py)
    rec = t[t.pff_role == "Pass Route"]
    dfn = t[t.side == "def"]
    m = rec.merge(dfn[["frameId", "x", "y", "vx", "vy"]], on="frameId", suffixes=("", "_d"))
    m["proj"] = np.hypot(m.x_d - m.x + (m.vx_d - m.vx) * HORIZON, m.y_d - m.y + (m.vy_d - m.vy) * HORIZON)
    op = m.groupby(["frameId", "nflId"]).proj.min().rename("openness")
    t = t.join(op, on=["frameId", "nflId"])
    return t, plays, int(snap), int(throw), los


def pocket_survival(snap, frames):
    curves, pp = pocket_curves()
    row = pp[(pp.gameId == GAME) & (pp.playId == PLAY)]
    blitz = bool(row.blitz.iloc[0]) if len(row) else False
    c = curves["Blitz" if blitz else "No blitz"]
    return {f: float(km_eval(c, (f - snap) / 10)) for f in frames}, blitz


def throw_info(t, throw):
    ro = load_openness()
    r = ro[(ro.gameId == GAME) & (ro.playId == PLAY)]
    tgt = r[r.targeted].iloc[0]
    mo = r[r.clear_open_dn].iloc[0]
    return tgt, mo


# ------------------------------------------------------------------ plotly
def field_shapes(x0, x1, los, first_down):
    sh = [dict(type="rect", x0=x0, x1=x1, y0=0, y1=W, fillcolor=FIELD_COL, line_width=0, layer="below")]
    for xl in range(int(np.ceil(x0 / 5) * 5), int(x1) + 1, 5):
        if 10 <= xl <= 110:
            sh.append(dict(type="line", x0=xl, x1=xl, y0=0, y1=W, line=dict(color="rgba(255,255,255,0.55)", width=1.5 if xl % 10 == 0 else 0.8), layer="below"))
    for xl in range(int(x0), int(x1) + 1):          # hash marks
        if 10 < xl < 110 and xl % 5:
            for yh in (0.5, 23.36, 29.97, W - 0.5):
                sh.append(dict(type="line", x0=xl, x1=xl, y0=yh - 0.4, y1=yh + 0.4, line=dict(color="rgba(255,255,255,0.45)", width=0.8), layer="below"))
    sh.append(dict(type="line", x0=los, x1=los, y0=0, y1=W, line=dict(color="#4ea8ff", width=3), layer="below"))
    sh.append(dict(type="line", x0=first_down, x1=first_down, y0=0, y1=W, line=dict(color="#ffd60a", width=3), layer="below"))
    sh.append(dict(type="rect", x0=x0, x1=x1, y0=W + 1.2, y1=W + 3.4, fillcolor="#dddddd", line_width=0, layer="below"))   # timer track
    return sh


def yard_labels(x0, x1, los_abs_offset):
    ann = []
    for xl in range(int(np.ceil(x0 / 10) * 10), int(x1) + 1, 10):
        if 20 <= xl <= 100:
            yd = xl - 10 if xl <= 60 else 110 - xl
            for yy in (5, W - 5):
                ann.append(dict(x=xl, y=yy, text=str(yd), showarrow=False, font=dict(color="rgba(255,255,255,0.7)", size=16, family=FONT)))
    return ann


def build_plotly(t, plays, snap, throw, los, surv, blitz, tgt, mo, ctx):
    frames_ids = sorted(t.frameId.unique())
    first_down = los + plays.yardsToGo
    x0, x1 = los - 14, los + 30
    names = t.drop_duplicates("nflId").set_index("nflId")

    def traces(f):
        d = t[t.frameId == f]
        rec = d[d.pff_role == "Pass Route"]
        off, dfn, ball = d[d.side == "off"], d[d.side == "def"], d[d.side == "ball"]
        el = (f - snap) / 10
        s = surv[f]
        after = f >= throw
        tr = [
            go.Scatter(x=rec.x, y=rec.y, mode="markers", hoverinfo="text",
                       text=[f"{n}: {o:.1f} yds open" for n, o in zip(rec.displayName, rec.openness)],
                       marker=dict(size=34, color="rgba(0,0,0,0)", line=dict(width=5, color=[openness_color(o) for o in rec.openness]))),
            go.Scatter(x=dfn.x, y=dfn.y, mode="markers+text", text=dfn.jerseyNumber.astype(int).astype(str), hoverinfo="skip",
                       textfont=dict(color="#111", size=10, family=FONT), marker=dict(size=19, color=DEF_COL, line=dict(width=1.5, color="#111"))),
            go.Scatter(x=off.x, y=off.y, mode="markers+text", text=off.jerseyNumber.astype(int).astype(str), hoverinfo="text",
                       hovertext=off.displayName, textfont=dict(color="white", size=10, family=FONT),
                       marker=dict(size=19, color=OFF_COL, line=dict(width=1.5, color="white"))),
            go.Scatter(x=ball.x, y=ball.y, mode="markers", hoverinfo="skip", marker=dict(size=9, color="#8b4513", symbol="diamond", line=dict(width=1, color="white"))),
            # pocket timer bar: width = survival probability, color fades from green to red
            go.Scatter(x=[x0, x0 + (x1 - x0) * s], y=[W + 2.3] * 2, mode="lines", hoverinfo="skip",
                       line=dict(width=14, color=openness_color(3 * s if s < 1 else 3))),
            go.Scatter(x=[(x0 + x1) / 2], y=[W + 5.4], mode="text", hoverinfo="skip",
                       text=[f"<b>{el:.1f} s after snap</b>   ·   pocket still intact on {s:.0%} of {'blitz' if blitz else '4-man-rush'} dropbacks at this point" + ("   ·   <b>BALL OUT</b>" if after else "")],
                       textfont=dict(size=15, color="#222", family=FONT)),
        ]
        if after:
            tr.append(go.Scatter(
                x=[tgt.x, mo.x], y=[tgt.y, mo.y], mode="markers+text", hoverinfo="skip",
                text=[f"<b>TARGETED</b>: {tgt.displayName}<br>{tgt.openness:.1f} yds open",
                      f"<b>MOST OPEN</b>: {mo.displayName}<br>{mo.openness:.1f} yds — ignored"],
                textposition=["bottom left", "top right"],
                textfont=dict(size=16, color=["#ffd0d0", "#d6ffd6"], family=FONT, shadow="1px 1px 2px #000, -1px -1px 2px #000, 1px -1px 2px #000, -1px 1px 2px #000", weight="bold"),
                marker=dict(size=48, color="rgba(0,0,0,0)", line=dict(width=3, color=["#ffffff", "#ffffff"]))))
        else:
            tr.append(go.Scatter(x=[None], y=[None], mode="markers"))
        return tr

    # the static throw-frame positions are used for the annotation (from openness csv); use tracked positions at the throw
    th = t[t.frameId == throw].set_index("nflId")
    tgt = tgt.copy(); mo = mo.copy()
    tgt["x"], tgt["y"] = th.loc[tgt.nflId, ["x", "y"]]
    mo["x"], mo["y"] = th.loc[mo.nflId, ["x", "y"]]

    frames = [go.Frame(data=traces(f), name=str(f)) for f in frames_ids]
    fig = go.Figure(data=traces(frames_ids[0]), frames=frames)
    title = (f"<b>{hero_title(ctx, mo)}</b><br>"
             f"<span style='font-size:13px;color:#555'>{ctx['sub']} · {plays.playDescription.split(') ', 1)[-1][:80]}</span>")
    steps = [dict(method="animate", label=f"{(f - snap) / 10:.1f}s" if (f - snap) % 5 == 0 or f == frames_ids[-1] else "",
                  args=[[str(f)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
             for f in frames_ids]
    legend_ann = []
    if ctx["clean"]:
        legend_ann.append(dict(x=x1, y=W + 0.6, xanchor="right", yanchor="bottom", showarrow=False,
                               text="No pressure on this play (tracking + PFF)", font=dict(size=12, family=FONT, color="#2a7a2a")))
    cd, tr_ = closing_defender(t, throw, tgt.nflId)
    legend_ann.append(dict(x=tr_.x, y=tr_.y, ax=cd.x, ay=cd.y, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=2, arrowwidth=1.6, arrowcolor="#ffd0d0", standoff=12, startstandoff=8, text=""))
    legend_ann += [
        dict(x=x0, y=-1.0, xanchor="left", yanchor="top", align="left", showarrow=False, font=dict(size=12, family=FONT, color="#333"),
             text=f"<span style='color:{OFF_COL}'>●</span> offense ({plays.possessionTeam})   "
                  f"<span style='color:#999'>○</span> defense ({plays.defensiveTeam})   "
                  f"ring = openness (yds to nearest defender, {HORIZON}s ahead): "
                  f"<span style='color:{COVER_COL}'>■ covered 0</span> → <span style='color:#d4a90a'>■ 1.5</span> → "
                  f"<span style='color:{OPEN_COL}'>■ open 3+</span><br>"
                  f"<span style='color:#4ea8ff'>━</span> line of scrimmage  <span style='color:#d4a90a'>━</span> 1st-down line"),
    ]
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=20, family=FONT)),
        shapes=field_shapes(x0, x1, los, first_down), annotations=yard_labels(x0, x1, 0) + legend_ann,
        xaxis=dict(range=[x0, x1], showgrid=False, zeroline=False, visible=False, fixedrange=True),
        yaxis=dict(range=[-4, W + 7], showgrid=False, zeroline=False, visible=False, scaleanchor="x", fixedrange=True),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False, height=1080, width=980,
        margin=dict(l=20, r=20, t=90, b=90), font=dict(family=FONT),
        updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=-0.02, xanchor="left", yanchor="top", direction="left",
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=110, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, steps=steps, x=0.16, len=0.84, y=-0.02, yanchor="top",
                      currentvalue=dict(visible=False), pad=dict(t=10))],
    )
    fig.write_html(os.path.join(OUT, "play_replay.html"), include_plotlyjs="cdn", auto_play=False)
    return fig


# ------------------------------------------------------------------ matplotlib (gif + still)
def draw_mpl(ax, t, plays, f, snap, throw, los, surv, blitz, tgt, mo, x0, x1, ctx):
    ax.clear()
    ax.add_patch(Rectangle((x0, 0), x1 - x0, W, color=FIELD_COL, zorder=0))
    for xl in range(int(np.ceil(x0 / 5) * 5), int(x1) + 1, 5):
        ax.plot([xl, xl], [0, W], color="white", alpha=0.55, lw=1.4 if xl % 10 == 0 else 0.7, zorder=1)
        if xl % 10 == 0 and 20 <= xl <= 100:
            yd = xl - 10 if xl <= 60 else 110 - xl
            for yy in (5, W - 5):
                ax.text(xl, yy, str(yd), color="white", alpha=0.7, ha="center", va="center", fontsize=13, zorder=1)
    ax.plot([los, los], [0, W], color="#4ea8ff", lw=2.5, zorder=1)
    ax.plot([los + plays.yardsToGo] * 2, [0, W], color="#ffd60a", lw=2.5, zorder=1)
    d = t[t.frameId == f]
    for r in d[d.pff_role == "Pass Route"].itertuples():
        ax.add_patch(Circle((r.x, r.y), 1.9, fill=False, lw=4, ec=openness_color(r.openness), zorder=2))
    for r in d[d.side != "ball"].itertuples():
        o = r.side == "off"
        ax.scatter(r.x, r.y, s=230, color=OFF_COL if o else DEF_COL, edgecolor="white" if o else "#111", lw=1.2, zorder=3)
        ax.text(r.x, r.y, str(int(r.jerseyNumber)), color="white" if o else "#111", ha="center", va="center", fontsize=7.5,
                fontweight="bold", zorder=4)
    b = d[d.side == "ball"]
    ax.scatter(b.x, b.y, s=40, marker="D", color="#8b4513", edgecolor="white", zorder=5)
    s, el = surv[f], (f - snap) / 10
    ax.add_patch(Rectangle((x0, W + 1.2), x1 - x0, 2.2, color="#dddddd", zorder=1, clip_on=False))
    ax.add_patch(Rectangle((x0, W + 1.2), (x1 - x0) * s, 2.2, color=openness_color(3 * s if s < 1 else 3), zorder=2, clip_on=False))
    ax.text((x0 + x1) / 2, W + 5.3, f"{el:.1f} s after snap  ·  pocket still intact on {s:.0%} of "
            f"{'blitz' if blitz else '4-man-rush'} dropbacks" + ("  ·  BALL OUT" if f >= throw else ""),
            ha="center", va="center", fontsize=12, fontweight="bold", color="#222")
    if ctx["clean"]:
        ax.text(x1, W + 0.5, "No pressure on this play (tracking + PFF)", ha="right", va="bottom", fontsize=9.5,
                color="#2a7a2a", fontweight="bold")
    if f >= throw:
        th = t[t.frameId == throw].set_index("nflId")
        cd, tr_ = closing_defender(t, throw, tgt.nflId)
        ax.annotate("", xy=(tr_.x, tr_.y), xytext=(cd.x, cd.y), zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color="#ffb3b3", lw=1.6, shrinkA=6, shrinkB=9))
        for key, row, col, txt in [("tgt", tgt, "#ff6b6b", f"TARGETED: {tgt.displayName}\n{tgt.openness:.1f} yds open"),
                                   ("mo", mo, "#7CFC7C", f"MOST OPEN: {mo.displayName}\n{mo.openness:.1f} yds — ignored")]:
            dx, dy, va = LABEL_POS[key]
            px, py = th.loc[row.nflId, ["x", "y"]]
            ax.add_patch(Circle((px, py), 2.9, fill=False, lw=2.5, ec="white", zorder=5))
            ax.text(px + dx, py + dy, txt, color=col, ha="center", va=va, fontsize=11.5, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#111111", alpha=0.75, ec="none"))
    ax.set_xlim(x0, x1); ax.set_ylim(-1, W + 7.5); ax.set_aspect("equal"); ax.axis("off")


def build_mpl(t, plays, snap, throw, los, surv, blitz, tgt, mo, ctx):
    x0, x1 = los - 14, los + 30
    frames_ids = sorted(t.frameId.unique())
    fig, ax = plt.subplots(figsize=(9.6, 10.4), dpi=80)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.89, bottom=0.07)
    ht = hero_title(ctx, mo).replace(" open. ", " open.\n")
    fig.suptitle(ht, fontsize=14.5, fontweight="bold", x=0.03, ha="left", y=0.988, va="top", linespacing=1.25)
    fig.text(0.03, 0.918, ctx["sub"], fontsize=11, color="#555")
    fig.text(0.03, 0.015, f"Navy = offense, white = defense. Ring = openness (yds to nearest defender, projected {HORIZON}s ahead):\n"
             "red = covered (0) → yellow (1.5) → green = open (3+).  Blue line = line of scrimmage, yellow line = 1st down.\n"
             "Top bar = Kaplan-Meier share of dropbacks with no rusher within 2 yds of the QB yet.", fontsize=9.5, color="#333")
    imgs = []
    for f in frames_ids:
        draw_mpl(ax, t, plays, f, snap, throw, los, surv, blitz, tgt, mo, x0, x1, ctx)
        fig.canvas.draw()
        imgs.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    dur = [0.11] * len(imgs)
    dur[frames_ids.index(throw)] = 0.6
    dur[-1] = 2.5
    imageio.mimsave(os.path.join(OUT, "play_replay.gif"), imgs, duration=dur, loop=0)
    draw_mpl(ax, t, plays, frames_ids[-1], snap, throw, los, surv, blitz, tgt, mo, x0, x1, ctx)
    fig.savefig(os.path.join(OUT, "play_replay_throw.png"), dpi=150)
    plt.close(fig)


def main():
    t, plays, snap, throw, los = load_play()
    surv, blitz = pocket_survival(snap, sorted(t.frameId.unique()))
    tgt, mo = throw_info(t, throw)
    print(f"play {GAME}/{PLAY}: snap {snap}, throw {throw} ({(throw - snap) / 10:.1f}s), blitz={blitz}; "
          f"target {tgt.displayName} {tgt.openness:.2f}, most open {mo.displayName} {mo.openness:.2f}; "
          f"recomputed at throw: {t[(t.frameId == throw) & t.openness.notna()].set_index('displayName').openness.round(2).to_dict()}")
    ctx = play_context(plays)
    print(hero_title(ctx, mo), "|", ctx["sub"])
    build_plotly(t, plays, snap, throw, los, surv, blitz, tgt, mo, ctx)
    build_mpl(t, plays, snap, throw, los, surv, blitz, tgt, mo, ctx)


if __name__ == "__main__":
    main()
