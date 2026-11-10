"""Visual 2: The Race -- pocket survival vs. how fast receivers get open.

Outputs: the_race.png, the_race.html
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common import CLOCK_DIR, FONT, INK, MUTED, OUT, POS_COL, km, km_eval, km_median, pocket_curves

MIN_ROUTES = 150
TMAX = 4.0


def data():
    curves, pp = pocket_curves()
    r = pd.read_csv(f"{CLOCK_DIR}/routes_clock.csv")
    cov = r[~r.free_release]
    pos_med = {p: km_median(km(cov[cov.pos == p].tto_obs, cov[cov.pos == p].opened)) for p in ["WR", "TE", "RB"]}
    pc = pd.read_csv(f"{CLOCK_DIR}/player_clock.csv")
    pc = pc[(pc.routes >= MIN_ROUTES) & pc.km_median_tto.notna()]
    # top receivers by Beat-the-Clock score, 4 WR + 3 TE + 3 RB so every position is represented
    top = pd.concat([pc[pc.pos == p].nlargest(n, "beat_clock") for p, n in [("WR", 4), ("TE", 3), ("RB", 3)]])
    top = top.sort_values(["km_median_tto", "beat_clock"], ascending=[False, True])
    return curves, pp, pos_med, top, pc


def rows(pos_med, top):
    """Bottom-panel rows: position medians (bold) then the named players, fastest at top."""
    rr = [dict(label=f"All {p}s (median)", t=pos_med[p], pos=p, bold=True, bc=None) for p in ["WR", "RB", "TE"]]
    rr += [dict(label=f"{r.displayName} ({r.team})", t=r.km_median_tto, pos=r.pos, bold=False, bc=r.beat_clock)
           for r in top.itertuples()]
    return rr


def mpl(curves, pos_med, top):
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 10.5), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.45], hspace=0.08))
    med_all, med_bl = km_median(curves["All"]), km_median(curves["Blitz"])
    tt = np.linspace(0, TMAX, 401)
    a1.plot(tt, km_eval(curves["All"], tt) * 100, color=INK, lw=3, label="All dropbacks")
    a1.plot(tt, km_eval(curves["Blitz"], tt) * 100, color=INK, lw=2.2, ls="--", label="Blitz (5+ rushers)")
    for ax in (a1, a2):
        ax.axvspan(med_all, TMAX, color="#d62728", alpha=0.07, lw=0)
        ax.axvline(med_all, color="#d62728", lw=1.4)
        ax.axvline(med_bl, color="#d62728", lw=1.2, ls="--")
    a1.text(med_all + 0.03, 96, f"Half of pockets\nbroken: {med_all:.1f}s", color="#d62728", fontsize=10.5, va="top", fontweight="bold")
    a1.text(med_bl - 0.03, 96, f"vs. blitz:\n{med_bl:.1f}s", color="#d62728", fontsize=10.5, va="top", ha="right")
    a1.set_ylim(0, 104); a1.set_ylabel("Pockets still intact (%)\n(no rusher within 2 yds of QB, Kaplan–Meier)")
    a1.legend(loc="lower left", frameon=False)
    a1.grid(axis="y", color="#eeeeee")

    rr = rows(pos_med, top)
    ys = np.arange(len(rr))[::-1]
    for y, r in zip(ys, rr):
        c = POS_COL[r["pos"]]
        a2.barh(y, r["t"], height=0.55 if r["bold"] else 0.38, color=c, alpha=0.95 if r["bold"] else 0.75)
        a2.scatter(r["t"], y, s=70, color=c, edgecolor="white", zorder=3)
        txt = f"{r['t']:.1f}s" + ("" if r["bc"] is None else f"   beat-the-clock {r['bc']:.2f}")
        a2.text(r["t"] + 0.05, y, txt, va="center", fontsize=9.5, color=INK)
    a2.axhline(ys[2] - 0.6, color="#cccccc", lw=1)
    a2.set_yticks(ys, [r["label"] for r in rr])
    for lab, r in zip(a2.get_yticklabels(), rr):
        lab.set_color(POS_COL[r["pos"]]); lab.set_fontweight("bold" if r["bold"] else "normal")
    a2.set_xlim(0, TMAX); a2.set_xlabel("Seconds after the snap")
    a2.set_title(f"Median time to get open after being covered (Kaplan–Meier; top receivers by Beat-the-Clock score, ≥{MIN_ROUTES} routes)",
                 fontsize=10.5, color=MUTED, loc="left")
    a2.text(med_bl - 0.05, ys[-1] - 0.75, "← open before the pocket breaks: beat the clock", ha="right", va="center", fontsize=10.5, color="#2ca02c", fontweight="bold")
    a2.text(med_all + 0.05, ys[-1] - 0.75, "too late", ha="left", va="center", fontsize=10.5, color="#d62728", fontweight="bold")
    a2.set_ylim(ys[-1] - 1.2, ys[0] + 0.6)
    fig.suptitle(f"The Race: TEs and RBs shake coverage in ~{pos_med['TE']:.1f}–{pos_med['RB']:.1f}s; the typical WR needs "
                 f"{pos_med['WR']:.1f}s,\nonly {med_bl - pos_med['WR']:.1f}s before a blitz has broken half of all pockets",
                 x=0.02, ha="left", fontsize=15, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.9, bottom=0.07)
    fig.savefig(os.path.join(OUT, "the_race.png"), dpi=150)
    plt.close(fig)


def interactive(curves, pos_med, pc):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.35, 0.65], vertical_spacing=0.04)
    tt = np.linspace(0, TMAX, 401)
    for k, dash in [("All", "solid"), ("Blitz", "dash"), ("No blitz", "dot")]:
        fig.add_trace(go.Scatter(x=tt, y=km_eval(curves[k], tt) * 100, name=f"Pocket: {k}", line=dict(color=INK, dash=dash, width=3 if k == "All" else 2),
                                 hovertemplate="%{x:.1f}s: %{y:.0f}% intact"), 1, 1)
    rng = np.random.default_rng(0)
    for p in ["WR", "TE", "RB"]:
        s = pc[pc.pos == p]
        fig.add_trace(go.Scatter(x=s.km_median_tto + rng.uniform(-0.04, 0.04, len(s)), y=s.beat_clock, mode="markers", name=p,
                                 marker=dict(color=POS_COL[p], size=np.sqrt(s.routes) * 0.9, opacity=0.75, line=dict(width=0.5, color="white")),
                                 text=s.displayName + " (" + s.team + ")", customdata=np.c_[s.routes, s.km_median_tto],
                                 hovertemplate="<b>%{text}</b><br>median time to open %{customdata[1]:.1f}s<br>Beat-the-Clock %{y:.2f}<br>%{customdata[0]} routes<extra></extra>"), 2, 1)
        fig.add_vline(x=pos_med[p], line=dict(color=POS_COL[p], width=2, dash="dot"), row=2, col=1)
    med = km_median(curves["All"])
    fig.add_vrect(x0=med, x1=TMAX, fillcolor="#d62728", opacity=0.07, line_width=0)
    fig.add_vline(x=med, line=dict(color="#d62728", width=2))
    fig.update_xaxes(range=[0, TMAX], title_text="Seconds after the snap (dot jittered ±0.04s)", row=2, col=1)
    fig.update_yaxes(title_text="Pockets intact (%)", row=1, col=1)
    fig.update_yaxes(title_text="Beat-the-Clock score", row=2, col=1)
    fig.update_layout(title=dict(text=f"<b>The Race</b>: every receiver with ≥{MIN_ROUTES} routes — median time to get open vs. the pocket clock "
                                      f"(red = after half of pockets are broken, {med:.1f}s)", font=dict(size=16)),
                      template="simple_white", font=dict(family=FONT), height=780, width=1100, hovermode="closest")
    fig.write_html(os.path.join(OUT, "the_race.html"), include_plotlyjs="cdn")


def main():
    curves, pp, pos_med, top, pc = data()
    print("position KM medians:", {k: round(v, 2) for k, v in pos_med.items()},
          "pocket medians:", km_median(curves["All"]), km_median(curves["Blitz"]))
    mpl(curves, pos_med, top)
    interactive(curves, pos_med, pc)


if __name__ == "__main__":
    main()
