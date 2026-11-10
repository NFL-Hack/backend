"""Visual 3: The Cost of Ignoring -- outcome when the QB hits the clearly-most-open downfield man vs. throws elsewhere.

Recomputed from pressure/play_pressure.csv (same clearly-open-downfield flag as openness/), split into all plays vs.
clean pocket & on time. Output: cost_of_ignoring.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd

from common import COVER_COL, INK, MUTED, OPEN_COL, OUT, ROOT


def stats():
    pp = pd.read_csv(f"{ROOT}/pressure/play_pressure.csv")
    cd = pp[pp.play_has_target.astype(bool) & pp.has_clear_dn.astype(bool)].copy()   # one row per play
    cd["hit"] = cd.clear_dn_hit.astype(bool)
    out = []
    for lab, s in [("All plays", cd), ("Clean pocket, 2.5 s+ to throw", cd[cd.clean_ontime.astype(bool)])]:
        g = s.groupby("hit").agg(n=("complete", "size"), comp=("complete", "mean"), yds=("playResult", "mean"))
        out.append((lab, len(s), 1 - s.hit.mean(), g.loc[True], g.loc[False]))
    clean, pres = cd[cd.clean_both.astype(bool)], cd[~cd.clean_both.astype(bool)]
    return out, clean.hit.mean(), pres.hit.mean()


def main():
    rows, hr_clean, hr_pres = stats()
    for lab, n, ign, hit, els in rows:
        print(f"{lab}: n {n}, ignored {ign:.1%}; hit {hit.comp:.1%} {hit.yds:.1f}; elsewhere {els.comp:.1%} {els.yds:.1f}")
    print(f"hit rate clean {hr_clean:.1%} vs pressured {hr_pres:.1%}")
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.2))
    labels = ["Threw to the\nopen downfield man", "Threw\nelsewhere"]
    cols = [OPEN_COL, COVER_COL]
    for r, (lab, n, ign, hit, els) in enumerate(rows):
        panels = [("Yards per play", [hit.yds, els.yds], "{:.1f}", 18, f"+{hit.yds - els.yds:.1f} yds"),
                  ("Completion rate", [hit.comp * 100, els.comp * 100], "{:.0f}%", 100, f"+{(hit.comp - els.comp) * 100:.1f} pts")]
        for c, (title, vals, fmt, ymax, gap) in enumerate(panels):
            ax = axes[r, c]
            bars = ax.bar([0, 1], vals, color=cols, width=0.62)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.02, fmt.format(v), ha="center", va="bottom",
                        fontsize=26, fontweight="bold", color=b.get_facecolor())
            ax.set_xticks([0, 1], labels if r == 1 else ["", ""], fontsize=12.5, color=INK)
            ax.set_ylim(0, ymax * 1.3)
            if r == 0:
                ax.set_title(title, fontsize=16, fontweight="bold", color=INK, pad=6)
            ax.text(0.5, ymax * 1.2, gap, ha="center", fontsize=13, color=MUTED)
            ax.set_yticks([])
            for sp in ["top", "right", "left"]:
                ax.spines[sp].set_visible(False)
            ax.tick_params(axis="x", length=0)
        axes[r, 0].set_ylabel(f"{lab.replace(', ', ',\n')}\nn = {n:,} · ignored {ign:.0%}", fontsize=13, fontweight="bold", color=INK,
                              rotation=0, ha="right", va="center", labelpad=12)
    lab, n, ign, hit, els = rows[1]
    fig.suptitle(f"Even with a clean pocket and time, QBs passed on the open man {ign:.0%}\n"
                 f"of the time, and it cost {hit.yds - els.yds:.1f} yards a play.", x=0.03, ha="left", fontsize=18, fontweight="bold", y=0.985)
    fig.text(0.03, 0.875, f"Pressure doesn't explain the ignoring: QBs hit the clearly-open downfield man on {hr_clean:.0%} of clean "
             f"plays and {hr_pres:.0%} of pressured ones.", fontsize=12.5, color="#444")
    fig.text(0.03, 0.018, f"n = {rows[0][1]:,} plays where a route runner 5+ yds downfield was clearly the most open (most open by 2+ yds, "
             f"projected 0.7 s ahead); {n:,} of them clean & on time.\nClean = no rusher within 2 yds of the QB before the throw "
             "(tracking) and no PFF hit, hurry or sack; on time = 2.5 s+ snap to throw.\nWeeks 1–8, 2021. Descriptive, not causal: "
             "the QB's reads, coverage and play design also drive where the ball goes.", fontsize=9.5, color=MUTED)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.8, bottom=0.15, wspace=0.12, hspace=0.18)
    fig.savefig(os.path.join(OUT, "cost_of_ignoring.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
