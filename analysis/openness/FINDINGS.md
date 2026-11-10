# The Open Man They Ignored: findings

Data: 2021 weeks 1–8, 122 games. That gives 7,604 dropbacks with a throw frame, 33,133 WR/TE/RB route-runner snapshots and 6,831 throws with an identified target.

## Method (3 lines)
1. At the throw frame (`pass_forward`, else `autoevent_passforward`), every player is projected **0.7 s ahead at constant velocity** (NGS `dir`: vx = s·sin, vy = s·cos, which was checked against frame-to-frame motion, r = 0.98).
2. **Openness** is the projected distance from a route runner to the nearest defender. A fast-closing defender shrinks it, so it bakes in closing speed (corr with closing speed −0.53).
3. The target is parsed from `playDescription` and matched to route runners. "Clear most open" means the most open route runner, by 2+ yds over the next man. "Ignored" means he was clearly most open and was not thrown to.

## Validation
Completion rate climbs steadily with the target's openness: **31% at 0–1 yd, 46% at 1–2, 65% at 2–3, 73% at 4–5, 81% at 5–7 and 89% at 10+**. Yards per target rise from 3.6 to about 10 (`01_validation_completion_vs_openness.png`).

## Top findings
1. **QBs throw to the most open man only 29% of the time.** On the 3,021 plays with a clear most-open receiver, he got the ball just 31% of the time. The median ignored man was **8.3 yds open, but at the line of scrimmage** (depth −0.2 yds), while the actual target was 2.5 yds open at 6.6 yds depth. Most "ignoring" is a QB choosing a contested downfield throw over a wide-open checkdown.
2. **Ignoring the open man *downfield* is expensive.** On 2,235 plays a route runner 5+ yds past the LOS was clearly most open (median 5.6 yds open, 9.8 yds deep). When the QB hit him the play averaged **79% completion and 14.7 yds**. When he went elsewhere, it averaged **65% and 6.7 yds**, usually to a tighter (3.2 yds open), shorter (4.2 yds deep) target. QBs passed on that downfield open man **72% of the time**.
3. **Stars are not immune: Justin Jefferson is among the most "ignored open" WRs downfield.** Jefferson was clearly the most open downfield man without getting the ball on **12.5% of his 212 routes**, against a WR average of 6.8%. Elijah Moore (NYJ) and Dyami Brown (WAS) were at 13%, and DK Metcalf at 11.3%. Among TEs the leaders were T.J. Hockenson (6.8%), Tonyan and Waller.
   - At RB, Ty'Son Williams, Javonte Williams and James Robinson were the clear most-open man and untargeted on **28–34% of routes** (RB average 17.6%). These are the league's most ignored checkdowns.
   - By team or QB, the clear open man was hit least often by HOU/Davis Mills (20–22%), DAL/Dak Prescott (21–23%) and ARI/Kyler Murray (23%), and most often by NE, NYJ and Josh Allen (~40%).

## Outputs
- `receiver_openness.csv` has one row per route runner per play.
- `player_summary.csv` covers players with ≥ 50 routes.
- `validation_by_openness.csv` and `ignore_by_{possessionTeam,qb}.csv` hold the supporting tables.
- `02_leaderboard_ignored_vs_openness.png`: the WR/TE panels use the downfield definition and the RB panel uses the all-routes definition.
- `03_play_diagram.png`: NYG–LA, Daniel Jones throws incomplete to a covered Dante Pettis (0.8 yds) while Collin Johnson is 21 yds open.

## Assumptions and limitations
- **Target identification:** 85% of non-penalty dropbacks name a receiver, and 99.7% of those match a route runner by first initial and last name (`D.Harris` is ambiguous). Penalty "No Play" snaps are excluded, and so are sacks and scrambles (no throw frame).
- **Openness is geometric only.** It ignores QB sightlines, pressure, progression timing, ball travel time to depth, and whether a backfield checkdown makes football sense. RBs dominate the "most open" counts (42% of RB routes, versus 17% for WR and 18% for TE), which is why the WR/TE leaderboards use the 5+ yd downfield variant.
- The projection uses constant velocity over 0.7 s, with no acceleration or ball-flight time. Thresholds (2-yd margin, 5-yd depth) are judgment calls. Small samples near 50 routes are noisy.
- The outcome gap between hitting and ignoring the open man is descriptive, not causal: QBs may skip open men because of pressure or what they read.
