"""Visual 4: interactive player scouting card (WR/TE/RB with >= 50 routes), with a dropdown to pick the player.

Outputs: scouting_card.html, scouting_card_preview.png (Cooper Kupp, via kaleido)
Clean-pocket ignored rate from pressure/player_clean_ignored.csv (clean_both = tracking + PFF), shown when >= 40 clean routes.
"""
import os as _os
ANALYSIS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repo/analysis
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common import CLOCK_DIR, FONT, INK, MUTED, OUT, POS_COL, clock_available, load_openness, load_player_summary

DEFAULT = "Cooper Kupp"
MIN_ROUTES = {"WR": 75, "TE": 50}                # card population; RBs excluded (see README)
RATING_CSV = _os.path.join(_os.path.dirname(ANALYSIS), "outputs", "receiver_rating.csv")
MIN_CLEAN = 40
CLEAN_CSV = f"{ANALYSIS}/pressure/player_clean_ignored.csv"
BINS = np.arange(0, 13, 1.0)                     # openness bins (yds), last bin = 12+
BIN_LAB = [f"{int(b)}" for b in BINS[:-1]] + ["12+"]
AVG_COL = "#9a9a9a"


def hist(o):
    c = np.histogram(np.clip(o, 0, 12.5), bins=np.r_[BINS, 99])[0]
    return (c / max(c.sum(), 1) * 100).round(1)


def build():
    ro = load_openness()
    ps = load_player_summary()
    ps = ps[ps.pos.map(MIN_ROUTES).le(ps.routes)]          # WR >= 75 routes, TE >= 50; no RBs
    rt = pd.read_csv(RATING_CSV)[["nflId", "clock_rating", "window_rate", "ttoe"]]
    ps = ps.merge(rt, on="nflId", how="left")
    ps["rating_rank"] = ps.groupby("pos").clock_rating.rank(ascending=False, method="min")
    ps["rating_n"] = ps.groupby("pos").clock_rating.transform("count")
    ro = ro[ro.nflId.isin(ps.nflId)]
    mz = ro[ro.pff_passCoverageType.isin(["Man", "Zone"])].groupby(["nflId", "pff_passCoverageType"]).openness.mean().unstack()
    ps = ps.join(mz.rename(columns={"Man": "open_man", "Zone": "open_zone"}), on="nflId")
    has_clock = clock_available()
    if has_clock:
        pc = pd.read_csv(f"{CLOCK_DIR}/player_clock.csv")[["nflId", "km_median_tto", "median_tto", "beat_clock",
                                                          "beat_clock_man", "beat_clock_zone", "open_pct"]]
        ps = ps.merge(pc, on="nflId", how="left")
    hists = {r.nflId: hist(g.openness.values) for r, g in [(r, ro[ro.nflId == r.nflId]) for r in ps.itertuples()]}
    pos_hist = {p: hist(ro[ro.pos == p].openness.values) for p in ["WR", "TE", "RB"]}
    pcl = pd.read_csv(CLEAN_CSV)[["nflId", "n_cleanAB", "ign_dn_cleanAB"]]
    ps = ps.merge(pcl, on="nflId", how="left")
    ps["ign_dn_clean"] = ps.ign_dn_cleanAB.where(ps.n_cleanAB >= MIN_CLEAN)   # NaN below the floor (also drops from pos avg)
    num = ps.select_dtypes("number").columns
    pos_avg = ps.groupby("pos")[num].mean()
    ps = ps.sort_values("displayName").reset_index(drop=True)
    return ps, hists, pos_hist, pos_avg, has_clock


def pct(v):
    return "–" if pd.isna(v) else f"{v * 100:.0f}%"


def num(v, fmt="{:.1f}"):
    return "–" if pd.isna(v) else fmt.format(v)


def player_state(r, hists, pos_hist, pa, has_clock):
    """Per-trace data (x, y, color, text) and layout annotations for one player."""
    p, c = r.pos, POS_COL[r.pos]
    a = pa.loc[p]
    rank = lambda col, asc=False: None
    tr = []
    tr.append(dict(x=BIN_LAB, y=hists[r.nflId], color=c, text=None))                                  # 0 player hist
    tr.append(dict(x=BIN_LAB, y=pos_hist[p], color=AVG_COL, text=None))                                # 1 pos hist
    cats = ["Downfield 5+ yds", "Any depth", "Downfield,<br>clean pocket"]
    pv = [r.ignored_open_dn_rate * 100, r.ignored_open_rate * 100, r.ign_dn_clean * 100]
    av = [a.ignored_open_dn_rate * 100, a.ignored_open_rate * 100, a.ign_dn_clean * 100]
    ptxt = [f"{v:.1f}%" for v in pv[:2]] + ([f"n<{MIN_CLEAN}"] if pd.isna(pv[2]) else [f"{pv[2]:.1f}%"])
    pv[2] = 0 if pd.isna(pv[2]) else pv[2]
    tr.append(dict(x=cats, y=pv, color=c, text=ptxt))                                                   # 2
    tr.append(dict(x=cats, y=av, color=AVG_COL, text=[f"{v:.1f}%" for v in av]))                       # 3
    if has_clock:
        pv = [r.km_median_tto]; av = [a.km_median_tto]
        tr.append(dict(x=["Time to open (s)"], y=pv, color=c, text=[num(v, "{:.1f}s") for v in pv]))   # 4
        tr.append(dict(x=["Time to open (s)"], y=av, color=AVG_COL, text=[num(v, "{:.1f}s") for v in av]))  # 5
        cats = ["Overall", "vs. Man", "vs. Zone"]
        pv = [r.beat_clock, r.beat_clock_man, r.beat_clock_zone]
        av = [a.beat_clock, a.beat_clock_man, a.beat_clock_zone]
        tr.append(dict(x=cats, y=pv, color=c, text=[num(v, "{:.2f}") for v in pv]))                    # 6
        tr.append(dict(x=cats, y=av, color=AVG_COL, text=[num(v, "{:.2f}") for v in av]))              # 7
    else:
        for _ in range(4):
            tr.append(dict(x=[], y=[], color=c, text=[]))
    cats = ["vs. Man", "vs. Zone"]
    pv = [r.open_man, r.open_zone]; av = [a.open_man, a.open_zone]
    tr.append(dict(x=cats, y=pv, color=c, text=[num(v, "{:.1f} yds") for v in pv]))                   # 8
    tr.append(dict(x=cats, y=av, color=AVG_COL, text=[num(v, "{:.1f} yds") for v in av]))             # 9
    card = (f"<b>{int(r.targets)}</b> targets ({pct(r.target_rate)} of routes)   <span style='color:{MUTED}'>{p} avg {a.targets:.0f} ({pct(a.target_rate)})</span><br>"
            f"<b>{pct(r.comp_pct_when_targeted)}</b> completion when targeted   <span style='color:{MUTED}'>{p} avg {pct(a.comp_pct_when_targeted)}</span><br>"
            f"<b>{num(r.yds_per_target)}</b> yds per target   <span style='color:{MUTED}'>{p} avg {a.yds_per_target:.1f}</span><br>"
            f"<b>{num(r.avg_openness)}</b> yds avg openness at throw   <span style='color:{MUTED}'>{p} avg {a.avg_openness:.1f}</span><br>"
            f"<b>{pct(r.most_open_share)}</b> of routes most open on the play   <span style='color:{MUTED}'>{p} avg {pct(a.most_open_share)}</span>")
    m = "ignored_open_rate" if p == "RB" else "ignored_open_dn_rate"
    ratio = getattr(r, m) / a[m] if a[m] else np.nan
    what = "clearly most open" + ("" if p == "RB" else " downfield")
    if ratio >= 1.3:
        take = f"{what} and ignored {ratio:.1f}× as often as the avg {p}"
    elif ratio <= 0.7:
        take = f"rarely ignored when {what} ({ratio:.1f}× the avg {p})"
    else:
        take = f"ignored when {what} about as often as the avg {p}"
    rank = "" if pd.isna(r.rating_rank) else f" · #{int(r.rating_rank)} of {int(r.rating_n)} {p}s on this card"
    ttoe = "–" if pd.isna(r.ttoe) else f"{r.ttoe:+.2f}s"
    rating = (f"<span style='font-size:26px;color:{c}'><b>clock_rating {num(r.clock_rating, '{:.0f}')}</b></span>"
              f"<span style='font-size:15px'>{rank}   ·   window_rate <b>{pct(r.window_rate)}</b>"
              f" <span style='color:{MUTED}'>({p} avg {pct(a.window_rate)})</span>   ·   ttoe <b>{ttoe}</b>"
              f" <span style='color:{MUTED}'>(+ = opens faster than expected)</span></span>")
    title = (f"<b>{r.displayName}</b>  <span style='color:{c}'>{p}</span> · {r.possessionTeam} · {int(r.routes)} routes"
             f"<br>{rating}"
             f"<br><span style='font-size:14px'>{take}</span>"
             f"<br><span style='font-size:12px;color:{MUTED}'>Colored = {r.displayName}; grey bars / black line = average {p} "
             f"(WR ≥75 routes, TE ≥50). clock_rating = 100 × mean of within-position percentiles of window_rate and ttoe."
             f"<br>Openness = yds to nearest defender projected 0.7s after the throw. Weeks 1–8, 2021.</span>")
    return tr, card, title


PANEL_TITLES = ["Openness at the throw: share of his routes (%)", "Ignored while clearly most open (% of routes)",
                "Time to get open after being covered", "Beat-the-Clock score (pocket intact when he opens)",
                "Avg openness at throw: man vs. zone", "When targeted"]


def main():
    ps, hists, pos_hist, pa, has_clock = build()
    fig = make_subplots(rows=2, cols=3, subplot_titles=PANEL_TITLES, horizontal_spacing=0.08, vertical_spacing=0.2,
                        specs=[[{}, {}, {}], [{}, {}, {"type": "domain"}]])
    cells = [(1, 1), (1, 1), (1, 2), (1, 2), (1, 3), (1, 3), (2, 1), (2, 1), (2, 2), (2, 2)]
    d0 = ps[ps.displayName == DEFAULT].iloc[0]
    tr0, card0, title0 = player_state(d0, hists, pos_hist, pa, has_clock)
    for i, ((rr, cc), t) in enumerate(zip(cells, tr0)):
        is_avg = i % 2 == 1
        if i == 1:
            fig.add_trace(go.Scatter(x=t["x"], y=t["y"], mode="lines+markers", name="Position avg", line=dict(color=INK, width=2.5, shape="linear"),
                                     marker=dict(size=5), hovertemplate="%{x} yds: %{y:.1f}%<extra>position avg</extra>"), rr, cc)
        elif i == 0:
            fig.add_trace(go.Bar(x=t["x"], y=t["y"], name="Player", marker_color=t["color"], opacity=0.85,
                                 hovertemplate="%{x} yds: %{y:.1f}%<extra>player</extra>"), rr, cc)
        else:
            fig.add_trace(go.Bar(x=t["x"], y=t["y"], text=t["text"], textposition="outside", cliponaxis=False,
                                 marker_color=t["color"], name="Position avg" if is_avg else "Player", showlegend=False,
                                 hovertemplate="%{x}: %{text}<extra>" + ("position avg" if is_avg else "player") + "</extra>"), rr, cc)
    base_ann = list(fig.layout.annotations)

    def anns(card):
        return [a.to_plotly_json() for a in base_ann] + [
            dict(text=card, x=0.72, y=0.36, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False,
                 align="left", font=dict(size=14, family=FONT, color=INK))] + (
            [] if has_clock else [dict(text="clock outputs not available", x=0.5, y=0.75, xref="paper", yref="paper", showarrow=False)])

    buttons = []
    for r in ps.itertuples():
        tr, card, title = player_state(r, hists, pos_hist, pa, has_clock)
        buttons.append(dict(method="update", label=f"{r.displayName} ({r.pos}, {r.possessionTeam})",
                            args=[{"x": [t["x"] for t in tr], "y": [t["y"] for t in tr],
                                   "marker.color": [t["color"] if i != 1 else INK for i, t in enumerate(tr)],
                                   "text": [t["text"] for t in tr]},
                                  {"title.text": title, "annotations": anns(card)}]))
    fig.update_layout(
        title=dict(text=title0, x=0.01, y=0.975, yanchor="top", font=dict(size=20, family=FONT)),
        annotations=anns(card0), template="simple_white", font=dict(family=FONT, size=12), barmode="group", bargap=0.25,
        height=940, width=1300, margin=dict(t=265, l=60, r=30, b=60), showlegend=False,
        updatemenus=[dict(buttons=buttons, active=int(ps.index[ps.displayName == DEFAULT][0]), x=0.085, xanchor="left",
                          y=1.13, yanchor="top", direction="down", showactive=True, bgcolor="white",
                          font=dict(size=13), pad=dict(r=4, t=4))],
    )
    fig.add_annotation(text="<b>Pick a player ▸</b>", x=0.0, xanchor="left", y=1.122, yanchor="top", xref="paper", yref="paper",
                       showarrow=False, font=dict(size=13, color=MUTED))
    fig.update_xaxes(title_text="Openness at throw (yds)", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", rangemode="tozero", row=1, col=2)
    fig.update_yaxes(title_text="seconds after snap", rangemode="tozero", row=1, col=3)
    fig.update_yaxes(range=[0, 1.1], row=2, col=1)
    fig.update_yaxes(title_text="yds", rangemode="tozero", row=2, col=2)
    # keep the "Pick a player" label when switching players
    for b in buttons:
        b["args"][1]["annotations"].append(fig.layout.annotations[-1].to_plotly_json())
    fig.layout.updatemenus[0].buttons = buttons
    fig.write_html(os.path.join(OUT, "scouting_card.html"), include_plotlyjs="cdn")
    fig._buttons = buttons
    try:
        fig.write_image(os.path.join(OUT, "scouting_card_preview.png"), scale=1.5)
    except Exception as e:  # kaleido/Chrome missing
        print("kaleido export failed:", e)
    print(f"{len(ps)} players; clock={has_clock}")
    return fig


if __name__ == "__main__":
    main()
