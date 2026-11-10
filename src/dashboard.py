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
    fig = lambda n: f"figures/{n}"
    img = lambda n, alt: f'<img src="{fig(n)}" alt="{alt}" loading="lazy">'

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
  @media (max-width:700px) {{ .frame {{ height:620px; }} }}
</style></head>
<body>
<div class="top"><div class="in">
  <div class="eyebrow">NFL Big Data Bowl &middot; NYU Edition</div>
  <h1>Clock Rating</h1>
  <p class="sub">A receiver ranking built from player tracking: who gets open early enough, and in the quarterback's sight, to be thrown to.</p>
</div></div>
<div class="wrap">
<div class="kpis">{tiles}</div>

<section class="card"><h2>Receiver ranking</h2>
  <p class="note">Clock Rating is a percentile within position ({a.min_routes}+ routes). It rewards getting open before pressure, in the quarterback's view, and sooner than expected for the route.</p>
  <div class="tabs" role="group" aria-label="Position">
    <button type="button" data-pos="WR" aria-pressed="true">Wide receivers</button>
    <button type="button" data-pos="TE" aria-pressed="false">Tight ends</button>
  </div>
  <div class="tbl"><table>
    <thead><tr><th>#</th><th>Player</th><th>Clock Rating</th><th class="num">In-time windows</th><th class="num">Opens vs. expected</th><th class="num">Routes</th></tr></thead>
    <tbody id="rows"></tbody>
  </table></div></section>

<section class="card"><h2>Top 3 at each position</h2>
  <p class="note">Bars show rank within position.</p>
  {img("b3_player_cards.png", "Player cards")}</section>

<section class="card"><h2>Watch one play</h2>
  <p class="note">Real tracking data. Press Play. A receiver turns yellow when the nearest defender is 3+ yards away; the pocket clock turns red when the rush reaches the quarterback. Blue line: line of scrimmage.</p>
  <div class="frame"><iframe src="{fig('play.html')}" title="Animated play" loading="lazy" scrolling="no"></iframe></div></section>

<section class="card"><h2>Why open receivers go unthrown</h2>
  <p class="note">Plays where the clearly open downfield receiver was not the target, sorted by what the tracking shows.</p>
  {img("b4_bucket_waffle.png", "Bucket waffle")}</section>

<section class="card"><h2>Open means completions</h2>
  <p class="note">Completion rate climbs with every yard of separation, which is why getting open in time matters.</p>
  {img("b5_validation.png", "Completion by separation")}</section>

<section class="card"><h2>How to use it</h2>
  <div class="how">
    <div><b>Coaches</b>Call plays for the receivers who reliably get open before the pocket breaks.</div>
    <div><b>Scouts</b>Clock Rating credits receivers who are open in time to be thrown to, not just open eventually.</div>
    <div><b>Broadcasters</b>A live pocket clock with receivers lighting up as they come open.</div>
    <div><b>Limits</b>Tracking ends at the throw, so catching and yards after the catch are not measured. Eight weeks of data.</div>
  </div></section>
<footer>Data: NFL Next Gen Stats + PFF, 2021 Weeks 1-8 (122 games, pass plays only).</footer>
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
