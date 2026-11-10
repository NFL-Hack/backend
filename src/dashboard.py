"""Dashboard page + animated play, NFL theme on white.

Usage:
  python src/broadcast.py --tracking data/tracking --play 2021101711,231
  python src/dashboard.py --data data --play 2021101711,231
Writes figures/play.html (animated play) and index.html (open it, or serve with GitHub Pages).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

NAVY, RED, SILVER, YELLOW, INK, SUB = "#013369", "#D50A0A", "#A5ACAF", "#FFD100", "#013369", "#5A6475"
KEY = ["gameId", "playId"]
truthy = lambda s: s.astype(str).str.lower().isin(["true", "1", "1.0"])


# ---- animated play -----------------------------------------------------------------------
def animated_play(a, play, out):
    import plotly.graph_objects as go
    gid, pid = play
    pc = pd.read_csv(os.path.join(a.outputs, "play_clock.csv")).set_index(KEY).loc[play]
    b = pd.read_csv(os.path.join(a.outputs, "buckets.csv"))
    r = b[(b.gameId == gid) & (b.playId == pid)].copy()
    r["is_target"], r["is_clear_open_downfield"] = truthy(r.is_target), truthy(r.is_clear_open_downfield)
    pff = pd.read_csv(os.path.join(a.data, "pffScoutingData.csv"))
    pff = pff[(pff.gameId == gid) & (pff.playId == pid)][["nflId", "pff_role"]]

    t = pd.read_csv(os.path.join(a.data, "tracking", f"tracking_{gid}.csv"))
    g = t[(t.playId == pid) & (t.frameId >= pc.snap_frame)].copy()
    left = g.playDirection == "left"
    g.loc[left, "x"] = 120 - g.loc[left, "x"]
    g.loc[left, "y"] = 53.3 - g.loc[left, "y"]
    g["t"] = ((g.frameId - pc.snap_frame) / 10).round(1)
    g = g.merge(pff, on="nflId", how="left")

    # separation per receiver per frame (nearest Coverage defender), frozen after the throw
    rec, cov = g[g.pff_role == "Pass Route"], g[g.pff_role == "Coverage"]
    m = rec[["nflId", "frameId", "x", "y"]].merge(cov[["frameId", "x", "y"]], on="frameId", suffixes=("", "_d"))
    m["d"] = np.hypot(m.x - m.x_d, m.y - m.y_d)
    sep = m.groupby(["nflId", "frameId"]).d.min().rename("sep").reset_index()
    sep.loc[sep.frameId > pc.end_frame, "sep"] = np.nan
    g = g.merge(sep, on=["nflId", "frameId"], how="left").sort_values(["nflId", "frameId"])
    g["sep"] = g.groupby("nflId").sep.ffill()
    g["num"] = g.jerseyNumber.fillna(0).astype(int).astype(str)

    qb_id = pc.qb_nflId
    off_team = g[g.nflId == qb_id].team.iloc[0]
    recv = set(rec.nflId)
    target = set(r[r.is_target].nflId)
    ignored = set(r[r.is_clear_open_downfield & ~r.is_target].nflId)
    names = r.set_index("nflId").displayName.str.split().str[-1].str.upper().to_dict()
    ttp, t_throw = pc.time_to_pressure, pc.t_end
    los = g[(g.team == "football") & (g.t == 0)].x.iloc[0]
    x0 = max(0, min(g.x.min(), los) - 8)
    x1 = min(120, max(g.x.max() + 6, x0 + 70))
    bar_y, bar_x0, bar_x1 = 58.5, x0 + 1, x1 - 1
    WHITE = "#FFFFFF"
    num_w = dict(size=11, color=WHITE, family="Arial Black, Arial")

    def traces(tt):
        f = g[g.t == tt]
        ball, p = f[f.team == "football"], f[f.team != "football"]
        qb, rc = p[p.nflId == qb_id], p[p.nflId.isin(recv)]
        is_open = (rc.sep >= 3.0) & (tt >= 1.0)
        offn = p[(p.team == off_team) & ~p.nflId.isin(recv) & (p.nflId != qb_id)]
        deff = p[p.team != off_team]
        trail = g[g.nflId.isin(recv) & (g.t <= min(tt, t_throw))]
        tx, ty = [], []
        for _, tr in trail.groupby("nflId"):
            tx += list(tr.x) + [None]
            ty += list(tr.y) + [None]
        hero = rc[rc.nflId.isin(ignored)]
        htxt = [names.get(i, "") + "  OPEN" if s >= 3 and tt >= 1.0 else "" for i, s in zip(hero.nflId, hero.sep)]
        tgt = rc[rc.nflId.isin(target)]
        pressured = pd.notna(ttp) and tt >= ttp
        frac = min(tt, t_throw) / t_throw
        state = "BALL IN THE AIR" if tt > t_throw else ("PRESSURE" if pressured else "CLEAN POCKET")
        return [
            go.Scatter(x=tx, y=ty, mode="lines", name="Route run so far", line=dict(color="rgba(255,255,255,0.6)", width=3)),
            go.Scatter(x=deff.x, y=deff.y, mode="markers+text", name="Defense", text=deff.num, textfont=num_w,
                       marker=dict(size=24, color=RED, line=dict(width=2, color=WHITE))),
            go.Scatter(x=offn.x, y=offn.y, mode="markers+text", name="Offense", text=offn.num, textfont=num_w,
                       marker=dict(size=24, color=NAVY, line=dict(width=2, color=WHITE))),
            go.Scatter(x=rc[~is_open].x, y=rc[~is_open].y, mode="markers+text", name="Receiver, covered", text=rc[~is_open].num,
                       textfont=num_w, marker=dict(size=26, color=NAVY, line=dict(width=3, color=WHITE))),
            go.Scatter(x=rc[is_open].x, y=rc[is_open].y, mode="markers+text", name="Receiver, open (3+ yd)", text=rc[is_open].num,
                       textfont=dict(size=11, color=NAVY, family="Arial Black, Arial"),
                       marker=dict(size=28, color=YELLOW, line=dict(width=3, color=NAVY))),
            go.Scatter(x=qb.x, y=qb.y, mode="markers+text", name="Quarterback", text=qb.num, textfont=num_w,
                       marker=dict(size=26, color=NAVY, symbol="square", line=dict(width=3, color=WHITE))),
            go.Scatter(x=ball.x, y=ball.y, mode="markers", name="Football",
                       marker=dict(size=15, color="#7A3E12", symbol="diamond-wide", line=dict(width=2, color=WHITE))),
            go.Scatter(x=hero.x + 1.8, y=hero.y, mode="text", text=htxt, textposition="middle right", showlegend=False,
                       textfont=dict(size=15, color=WHITE, family="Arial Black, Arial")),
            go.Scatter(x=tgt.x, y=tgt.y + 2.2, mode="text", text=[names.get(i, "") + "  TARGET" for i in tgt.nflId],
                       textposition="top center", showlegend=False, textfont=dict(size=13, color=WHITE, family="Arial Black, Arial")),
            go.Scatter(x=[bar_x0, bar_x0 + (bar_x1 - bar_x0) * frac], y=[bar_y, bar_y], mode="lines",
                       name="Pocket clock (red = pressure)", line=dict(width=16, color=RED if pressured else NAVY)),
            go.Scatter(x=[bar_x0], y=[bar_y + 2.7], mode="text", text=[f"POCKET CLOCK  {min(tt, t_throw):.1f}s   {state}"],
                       textposition="middle right", textfont=dict(size=17, color=NAVY, family="Arial Black, Arial"), showlegend=False),
        ]

    times = sorted(g.t.unique())
    shapes, ann = [], []
    for i, xs in enumerate(range(0, 120, 5)):
        if xs + 5 > x0 and xs < x1:
            col = "#1F5A31" if (xs < 10 or xs >= 110) else ("#2E7D43" if i % 2 else "#297239")
            shapes.append(dict(type="rect", x0=max(xs, x0), x1=min(xs + 5, x1), y0=0, y1=53.3, fillcolor=col, line=dict(width=0), layer="below"))
    for yl in range(10, 111):
        if not (x0 <= yl <= x1):
            continue
        if yl % 5 == 0:
            shapes.append(dict(type="line", x0=yl, x1=yl, y0=0, y1=53.3, line=dict(color="rgba(255,255,255,0.85)", width=2), layer="below"))
            if yl % 10 == 0 and 10 < yl < 110 and yl < x1 - 2:
                n = yl - 10 if yl <= 60 else 110 - yl
                for yy in (7, 46.3):
                    ann.append(dict(x=yl, y=yy, text=f"<b>{n}</b>", showarrow=False, font=dict(size=26, color="rgba(255,255,255,0.8)")))
        else:
            for ya, yb in ((0.4, 1.4), (22.9, 23.9), (29.4, 30.4), (51.9, 52.9)):
                shapes.append(dict(type="line", x0=yl, x1=yl, y0=ya, y1=yb, line=dict(color="rgba(255,255,255,0.7)", width=1), layer="below"))
    shapes += [dict(type="rect", x0=x0, x1=x1, y0=0, y1=53.3, line=dict(color=WHITE, width=3), fillcolor="rgba(0,0,0,0)", layer="below"),
               dict(type="line", x0=los, x1=los, y0=0, y1=53.3, line=dict(color="#4DA3FF", width=4), layer="below"),
               dict(type="line", x0=bar_x0, x1=bar_x1, y0=bar_y, y1=bar_y, line=dict(color="#DDE2EA", width=16), layer="below")]
    if pd.notna(ttp):
        px = bar_x0 + (bar_x1 - bar_x0) * ttp / t_throw
        shapes.append(dict(type="line", x0=px, x1=px, y0=bar_y - 1.6, y1=bar_y + 1.6, line=dict(color=RED, width=4)))
    hero_name = r[r.nflId.isin(ignored)].displayName
    tgt_name = r[r.nflId.isin(target)].displayName
    title = f"{hero_name.iloc[0]} is open. The ball goes to {tgt_name.iloc[0]}." if len(hero_name) and len(tgt_name) else "One play"

    fig = go.Figure(data=traces(times[0]), frames=[go.Frame(data=traces(tt), name=f"{tt:.1f}") for tt in times])
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=24, color=NAVY), x=0.01),
        xaxis=dict(range=[x0 - 0.5, x1 + 0.5], visible=False, fixedrange=True),
        yaxis=dict(range=[-1, 63], visible=False, scaleanchor="x", fixedrange=True),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(color=NAVY), shapes=shapes, height=820,
        annotations=ann,
        legend=dict(orientation="h", y=-0.03, x=0, font=dict(size=14, color=NAVY), itemsizing="constant"),
        margin=dict(l=16, r=16, t=70, b=170),
        updatemenus=[dict(type="buttons", x=0.0, y=-0.14, xanchor="left", direction="left", showactive=False,
                          bgcolor=NAVY, font=dict(color="white", size=14), buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=120, redraw=False), fromcurrent=True, transition=dict(duration=0))]),
            dict(label="❚❚ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(x=0.22, len=0.78, y=-0.12, font=dict(color=NAVY), currentvalue=dict(visible=False), bgcolor="#DDE2EA",
                      steps=[dict(label=f"{tt:.1f}" if int(round(tt * 10)) % 5 == 0 else "", method="animate",
                                  args=[[f"{tt:.1f}"], dict(frame=dict(duration=0, redraw=False), mode="immediate")]) for tt in times])])
    html = fig.to_html(include_plotlyjs=True, full_html=True, auto_play=False, config={"displayModeBar": False, "responsive": True})
    html = html.replace("<html>", '<!doctype html>\n<html lang="en">', 1).replace(
        "<head>", '<head><meta name="viewport" content="width=device-width, initial-scale=1"><title>Animated play</title>'
                  "<style>html,body{margin:0;background:#fff}</style>", 1)
    html = html.replace("\ufffd", "\\uFFFD")   # literal U+FFFD inside a regex in plotly.js -> escape form, same meaning
    with open(out, "w") as f:
        f.write(html)
    print("saved", out)
    return fig


# ---- dashboard page ----------------------------------------------------------------------
def page(a, out):
    b = pd.read_csv(os.path.join(a.outputs, "buckets.csv"))
    an = b[truthy(b.analyzed)]
    tg = an[truthy(an.is_target)]
    ig = an[~an.index.isin(tg.index)]
    share = ig.bucket3.value_counts(normalize=True) * 100
    comp = lambda d: (d.passResult == "C").mean() * 100
    rr = pd.read_csv(os.path.join(a.outputs, "receiver_rating.csv"))
    rr = rr[rr.n_routes >= a.min_routes]
    data = rr[["displayName", "position", "n_routes", "window_rate", "ttoe", "clock_rating", "pct_window", "pct_ttoe"]].round(4).to_dict("records")

    kpis = [
        (f"{len(rr)}", f"receivers ranked ({(rr.position == 'WR').sum()} WR, {(rr.position == 'TE').sum()} TE, {a.min_routes}+ routes)", "navy"),
        (f"{len(ig) / len(an) * 100:.0f}%", f"of {len(an):,} plays with a clearly open receiver downfield, the QB threw elsewhere", "red"),
        (f"{share.get('OPEN_WINDOW', 0):.0f}%", "of those ignored receivers were a real, unused window", "navy"),
        (f"{comp(tg):.0f}% vs {comp(ig):.0f}%", "completion rate when the open receiver got the ball vs. when it went elsewhere", "red"),
    ]
    tiles = "".join(f'<div class="kpi {c}"><div class="num">{n}</div><div class="lab">{l}</div></div>' for n, l, c in kpis)
    fig = lambda n: n if "/" in n else f"figures/{n}"
    img = lambda n, alt: (f'<a class="zoom" href="{fig(n)}" target="_blank" rel="noopener">'
                          f'<img src="{fig(n)}" alt="{alt}" loading="lazy"></a>'
                          f'<div class="open"><a href="{fig(n)}" target="_blank" rel="noopener">Open full size ↗</a></div>')
    hit, els = tg.playResult.mean(), ig.playResult.mean()
    n_open = int((ig.bucket3 == "OPEN_WINDOW").sum())

    html = f"""<!doctype html>
<html lang="en" data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clock Rating</title>
<style>
  :root {{ --bg:#FFFFFF; --card:#FFFFFF; --ink:#013369; --body:#1B2433; --muted:#5A6475; --line:#DDE2EA;
           --navy:#013369; --red:#D50A0A; --silver:#A5ACAF; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0B1626; --card:#13233A; --ink:#FFFFFF; --body:#E6EBF2;
           --muted:#9DB2C4; --line:#24364F; }} }}
  :root[data-theme="dark"] {{ --bg:#0B1626; --card:#13233A; --ink:#FFFFFF; --body:#E6EBF2; --muted:#9DB2C4; --line:#24364F; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--body); line-height:1.45;
         font-family:"Segoe UI",-apple-system,BlinkMacSystemFont,Roboto,Helvetica,Arial,sans-serif; }}
  .top {{ background:var(--bg); border-bottom:6px solid var(--red); }}
  .top .in {{ max-width:1240px; margin:0 auto; padding:22px 16px; }}
  .eyebrow {{ color:var(--red); font-weight:700; letter-spacing:.1em; font-size:13px; text-transform:uppercase; }}
  h1 {{ margin:2px 0 6px; font-size:clamp(28px,4vw,42px); line-height:1.1; color:var(--ink); }}
  .sub {{ margin:0; color:var(--muted); font-size:clamp(15px,1.5vw,19px); max-width:78ch; }}
  .wrap {{ max-width:1240px; margin:0 auto; padding:16px 16px 48px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:14px; margin-bottom:14px; }}
  .kpi {{ background:var(--card); border-radius:6px; padding:16px 18px; border:1px solid var(--line); border-top:6px solid var(--navy); min-width:0; }}
  .kpi.red {{ border-top-color:var(--red); }}
  .kpi .num {{ color:var(--ink); font-size:clamp(28px,3.2vw,40px); font-weight:800; line-height:1.1; overflow-wrap:anywhere; }}
  .kpi .lab {{ color:var(--muted); font-size:14.5px; margin-top:6px; }}
  .card {{ background:var(--card); border-radius:6px; padding:20px; border:1px solid var(--line); margin-bottom:14px; min-width:0; }}
  .card h2 {{ margin:0 0 4px; font-size:20px; color:var(--ink); }}
  .note {{ margin:0 0 14px; color:var(--muted); font-size:15px; }}
  .card img {{ display:block; width:100%; height:auto; border-radius:4px; }}
  .frame {{ position:relative; width:100%; height:860px; overflow:hidden; border-radius:4px; }}
  .frame iframe {{ position:absolute; inset:0; width:100%; height:100%; border:0; }}
  .tabs {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }}
  .tabs button {{ font:inherit; font-weight:700; padding:8px 18px; border-radius:4px; border:1px solid var(--line); background:var(--card); color:var(--ink); cursor:pointer; }}
  .tabs button[aria-pressed="true"] {{ background:var(--navy); color:#fff; border-color:var(--navy); }}
  .tabs button:focus-visible {{ outline:3px solid var(--red); outline-offset:2px; }}
  .tbl {{ overflow-x:auto; }}
  table {{ width:100%; border-collapse:collapse; font-size:15px; }}
  th {{ text-align:left; color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.03em; padding:8px 6px; border-bottom:2px solid var(--line); white-space:nowrap; }}
  td {{ padding:7px 6px; border-bottom:1px solid var(--line); white-space:nowrap; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.rank {{ color:var(--muted); width:32px; }}
  td.name {{ font-weight:700; color:var(--ink); }}
  .bar {{ display:flex; align-items:center; gap:8px; }}
  .bar i {{ display:block; height:14px; border-radius:2px; background:var(--navy); flex:0 0 auto; }}
  .bar b {{ font-variant-numeric:tabular-nums; color:var(--ink); }}
  .how {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; }}
  .how div {{ border-left:4px solid var(--red); padding-left:12px; }}
  .how b {{ display:block; margin-bottom:2px; color:var(--ink); }}
  footer {{ margin-top:20px; color:var(--muted); font-size:14px; }}
  .lead {{ font-size:17px; color:var(--body); max-width:85ch; margin:0 0 16px; }}
  .steps {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; margin:0; padding:0; list-style:none; counter-reset:s; }}
  .steps li {{ counter-increment:s; background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:14px 14px 14px 52px; position:relative; }}
  .steps li::before {{ content:counter(s); position:absolute; left:14px; top:12px; width:28px; height:28px; border-radius:50%;
                      background:var(--navy); color:#fff; font-weight:800; display:flex; align-items:center; justify-content:center; }}
  .steps b {{ display:block; color:var(--ink); margin-bottom:2px; }}
  .terms {{ display:grid; grid-template-columns:max-content 1fr; gap:6px 16px; margin:16px 0 0; font-size:15px; }}
  .terms dt {{ font-weight:700; color:var(--ink); }}
  .terms dd {{ margin:0; color:var(--muted); }}
  .chip {{ display:inline-block; font-size:12px; font-weight:800; letter-spacing:.06em; color:#fff; border-radius:3px; padding:1px 7px; margin-right:6px; }}
  .chip.late {{ background:var(--red); }} .chip.nrw {{ background:var(--silver); }} .chip.open {{ background:var(--navy); }}
  .step-n {{ color:var(--red); font-weight:800; letter-spacing:.08em; font-size:13px; text-transform:uppercase; margin-bottom:2px; }}
  .zoom img {{ border:1px solid var(--line); }}
  .open {{ text-align:right; font-size:13px; margin-top:6px; }}
  .open a, .links a {{ color:var(--navy); font-weight:700; text-decoration:none; }}
  :root[data-theme="dark"] .open a, :root[data-theme="dark"] .links a {{ color:#9DC3FF; }}
  .links {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }}
  .links a {{ display:block; border:1px solid var(--line); border-left:4px solid var(--navy); border-radius:6px; padding:12px 14px; }}
  .links span {{ display:block; color:var(--muted); font-weight:400; font-size:14px; margin-top:2px; }}
  .toc {{ display:flex; flex-wrap:wrap; gap:6px 18px; margin-top:14px; font-size:14px; }}
  .toc a {{ color:var(--navy); font-weight:700; text-decoration:none; }}
  @media (max-width:700px) {{ .frame {{ height:620px; }} }}
</style></head>
<body>
<div class="top"><div class="in">
  <div class="eyebrow">NFL Big Data Bowl &middot; NYU Edition &middot; 2021 Weeks 1-8</div>
  <h1>Open, Seen, In Time: why QBs ignore open receivers</h1>
  <p class="sub">On {len(an):,} plays a receiver 5+ yards downfield was clearly the most open man. QBs threw to him only {len(tg):,} times. We use player tracking to ask, for every one they passed up: did he open too late, could the QB not see him, or did the QB have him and throw elsewhere? Then we rate receivers on getting open in time, where the QB can see them.</p>
  <nav class="toc" aria-label="Sections">
    <a href="#how">How it works</a><a href="#play">One play</a><a href="#why">Why he was ignored</a>
    <a href="#cost">What it costs</a><a href="#rating">Clock Rating</a><a href="#terms">Terms</a><a href="#limits">Limits</a><a href="#more">More visuals</a>
  </nav>
</div></div>
<div class="wrap">
<div class="kpis">{tiles}</div>

<section class="card" id="how"><div class="step-n">How it works</div><h2>Four checks for every open receiver</h2>
  <p class="lead">For each play we track every route runner, the pass rushers and the QB from snap to throw (10 frames per second). A receiver is <b>open</b> when the nearest coverage defender is 3+ yards away.</p>
  <ol class="steps">
    <li><b>Find the clear-open man</b>A receiver 5+ yds past the line of scrimmage who is the most open by 2+ yds (separation projected 0.7 s ahead).</li>
    <li><b>Start the pocket clock</b>Pressure = the first moment any pass rusher is within 2 yds of the QB. Did his window open before that, and at least 0.4 s before the throw?</li>
    <li><b>Check the QB's sight</b>Is the receiver within 90&deg; of the direction the QB's body faces at the moment his window opens?</li>
    <li><b>Check he was still open</b>Was he still 3+ yds from the nearest defender when the ball was thrown?</li>
  </ol>
  <p class="note" style="margin-top:14px">Fail check 2 &rarr; <span class="chip late">TOO LATE</span> Fail check 3 or 4 &rarr; <span class="chip nrw">NOT A REAL WINDOW</span> Pass all &rarr; <span class="chip open">OPEN WINDOW</span>: the QB had him and threw elsewhere.</p></section>

<section class="card" id="play"><div class="step-n">One play</div><h2>Swain is open 1.0 s before the rush arrives. Geno Smith throws a 2-yd pass to Metcalf.</h2>
  <p class="note">An OPEN WINDOW play (SEA at PIT, Week 6). Freddie Swain opens 1.7 s after the snap, inside the QB's view; pressure arrives at 2.7 s; the throw is at 3.0 s, with Swain 10 yds clear. Press Play: a receiver turns yellow when he is 3+ yds open, and the pocket clock turns red when the rush reaches the QB.</p>
  <div class="frame"><iframe src="{fig('play.html')}" title="Animated play: Freddie Swain open, Geno Smith throws to DK Metcalf" loading="lazy" scrolling="no"></iframe></div>
  <h3 style="margin:18px 0 6px;color:var(--ink);font-size:17px">The same play, frozen when Swain opens</h3>
  {img("b1_telestrator.png", "Field diagram at 1.7 s: Swain open inside the shaded QB field of view, before the nearest rusher reaches the QB")}
  <h3 style="margin:18px 0 6px;color:var(--ink);font-size:17px">Every route runner against the clock</h3>
  {img("b2_race_lower_third.png", "Timeline: Swain open from 1.7 s, pressure at 2.7 s, throw at 3.0 s; the other four receivers never open")}</section>

<section class="card" id="why"><div class="step-n">Why he was ignored</div><h2>{len(ig):,} ignored open receivers, sorted by the first check they fail</h2>
  <p class="note"><span class="chip late">TOO LATE</span> {share.get('TOO_LATE', 0):.1f}%: pressure came first, or he opened under 0.4 s before the throw.
    <span class="chip nrw">NOT A REAL WINDOW</span> {share.get('NOT_REAL_WINDOW', 0):.1f}%: outside the QB's view, or not open by the throw.
    <span class="chip open">OPEN WINDOW</span> {share.get('OPEN_WINDOW', 0):.1f}% ({n_open} plays): the QB had him and threw elsewhere.</p>
  {img("b4_bucket_waffle.png", "Waffle chart: 29% open window, 32% too late, 39% not a real window, of 1,610 ignored plays")}
  <p class="note" style="margin-top:14px">Pressure doesn't explain the ignoring: QBs hit the clear-open downfield man on about 28% of plays whether the pocket was clean or pressured.</p></section>

<section class="card" id="cost"><div class="step-n">What it costs</div><h2>Throwing to him: {comp(tg):.0f}% completions and {hit:.1f} yds a play. Elsewhere: {comp(ig):.0f}% and {els:.1f}.</h2>
  <p class="note">With a clean pocket and 2.5 s+ to throw, QBs still passed on him 66% of the time, at a cost of 9.3 yds a play. This chart counts 2,236 plays: it keeps the 7 plays with no snap event that the rest of the page drops.</p>
  {img("analysis/viz/cost_of_ignoring.png", "Bar charts: 14.7 vs 6.7 yards per play and 79% vs 65% completion; clean pocket 16.6 vs 7.3 and 85% vs 66%")}
  <h3 style="margin:18px 0 6px;color:var(--ink);font-size:17px">Why openness matters: completion rate by the target's separation at the throw</h3>
  {img("b5_validation.png", "Completion rate rises from 26% at 0-1 yd of separation to 88% at 10+ yd")}</section>

<section class="card" id="rating"><div class="step-n">Clock Rating</div><h2>Who gets open in time, where the QB can see him</h2>
  <p class="note"><b>Clock Rating</b> (0-100) = the average of two percentiles, each ranked within position (WR vs WR, TE vs TE, among WR/TE with 30+ routes):
    <b>in-time window rate</b>, the share of his routes that pass all four checks, and <b>opens vs. expected</b>, how much sooner he gets open after being covered than the typical receiver at his position, alignment and coverage.
    The table lists receivers with {a.min_routes}+ routes. RBs are excluded: their short checkdowns overstate openness.</p>
  <div class="tabs" role="group" aria-label="Position">
    <button type="button" data-pos="WR" aria-pressed="true">Wide receivers</button>
    <button type="button" data-pos="TE" aria-pressed="false">Tight ends</button>
  </div>
  <div class="tbl"><table>
    <thead><tr><th>#</th><th>Player</th><th>Clock Rating</th><th class="num">In-time windows</th><th class="num">Opens vs. expected</th><th class="num">Routes</th></tr></thead>
    <tbody id="rows"></tbody>
  </table></div>
  <h3 style="margin:18px 0 6px;color:var(--ink);font-size:17px">Top 3 at each position</h3>
  {img("b3_player_cards.png", "Top 3 WR and TE by Clock Rating")}</section>

<section class="card" id="terms"><div class="step-n">Terms</div><h2>What the words mean</h2>
  <dl class="terms">
    <dt>Open</dt><dd>Nearest coverage defender 3+ yds away.</dd>
    <dt>Clear-open downfield</dt><dd>5+ yds past the line of scrimmage and the most open receiver by 2+ yds (separation projected 0.7 s ahead).</dd>
    <dt>Ignored</dt><dd>The QB threw to someone else.</dd>
    <dt>Window start</dt><dd>When his final open stretch before the throw began (and no earlier than 5 yds downfield).</dd>
    <dt>Pressure</dt><dd>First moment a pass rusher is within 2 yds of the QB.</dd>
    <dt>In sight</dt><dd>Within 90&deg; of the QB's body orientation. Tracking has no eye or head direction. Check: 99.4% of targeted receivers are in sight at the throw.</dd>
    <dt>In-time window rate</dt><dd>Share of a receiver's routes that pass all four checks.</dd>
    <dt>Opens vs. expected</dt><dd>Typical time to get open after being covered, for his position &times; alignment &times; coverage, minus his own. Positive = faster.</dd>
  </dl></section>

<section class="card" id="limits"><div class="step-n">Who uses it, and the limits</div><h2>Use it to ask better questions, not to grade QBs</h2>
  <div class="how">
    <div><b>Coaches</b>Separate "no time" (protection, route timing) from "didn't see it" (progression, eyes).</div>
    <div><b>Scouts</b>Clock Rating credits receivers who are open in time to be thrown to, not just open eventually.</div>
    <div><b>Broadcasters</b>A live pocket clock with receivers lighting up as they come open.</div>
    <div><b>Limits</b>Open does not mean the QB should throw: reads, coverage and play design matter. Tracking ends at the throw. Body orientation stands in for vision. Thresholds (3 yd, 2 yd, 90&deg;, 0.4 s) are our choices. Eight weeks of one season. Descriptive, not causal.</div>
  </div></section>

<section class="card" id="more"><div class="step-n">More visuals</div><h2>Go deeper</h2>
  <div class="links">
    <a href="analysis/viz/play_replay.html">Second play replay<span>Jefferson 7.5 yds open, clean pocket; Cousins throws to a covered Thielen.</span></a>
    <a href="analysis/viz/scouting_card.html">Interactive scouting card<span>Pick any WR (75+ routes) or TE (50+): Clock Rating plus openness, man vs zone.</span></a>
    <a href="analysis/viz/the_race.html">The race against the pocket<span>How fast each position gets open after being covered, against the pocket clock.</span></a>
    <a href="analysis/viz/index.html">All analysis visuals<span>Funnel, cost of ignoring, the race, scouting card, replay.</span></a>
    <a href="https://github.com/NFL-Hack/backend#readme">Method, code and data<span>README with every definition, threshold and limitation.</span></a>
  </div></section>
<footer>Data: NFL Next Gen Stats + PFF, 2021 Weeks 1-8 (122 games, pass plays only). 7 plays dropped: no snap event.</footer>
</div>
<script>
const DATA = {json.dumps(data)};
const rows = document.getElementById('rows');
function show(pos) {{
  const d = DATA.filter(r => r.position === pos).sort((a, b) => b.clock_rating - a.clock_rating).slice(0, 15);
  rows.innerHTML = d.map((r, i) => `<tr><td class="rank">${{i + 1}}</td><td class="name">${{r.displayName}}</td>
    <td><div class="bar"><i style="width:${{r.clock_rating * 0.9}}px"></i><b>${{Math.round(r.clock_rating)}}</b></div></td>
    <td class="num">${{(r.window_rate * 100).toFixed(0)}}%</td>
    <td class="num">${{r.ttoe >= 0 ? r.ttoe.toFixed(2) + 's faster' : (-r.ttoe).toFixed(2) + 's slower'}}</td>
    <td class="num">${{r.n_routes}}</td></tr>`).join('');
  document.querySelectorAll('.tabs button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.pos === pos)));
}}
document.querySelectorAll('.tabs button').forEach(b => b.addEventListener('click', () => show(b.dataset.pos)));
show('WR');
</script>
</body></html>"""
    with open(out, "w") as f:
        f.write(html)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--play", default="2021101711,231")
    ap.add_argument("--min-routes", type=int, default=100)
    a = ap.parse_args()
    os.makedirs("figures", exist_ok=True)
    animated_play(a, tuple(int(v) for v in a.play.split(",")), "figures/play.html")
    page(a, "index.html")


if __name__ == "__main__":
    main()
