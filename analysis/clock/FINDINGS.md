# Clock Rating: can the receiver get open before the pocket dies?
Data: 122 games, 8,045 dropbacks, 37,155 WR/TE/RB routes. Run with `python clock.py`, which takes about 6 s.

## Top 3 findings
1. Man coverage stops receivers getting open much more than the pass rush does. Counting only routes where a defender came within 3 yds:
   - WRs vs. man got open on 31% of routes, vs. 66% against zone.
   - TEs: 49% vs. 84%. RBs: 65% vs. 86%.
   - The pocket barely differs: median pocket life is 3.4 s vs. 3.6 s.
   Zone-only receivers (Beat-the-Clock score vs. man / vs. zone):
   - Mike Gesicki (TE): 0.30 / 0.84, on 67 man routes
   - Chester Rogers: 0.15 / 0.76
   - Christian Kirk: 0.35 / 0.78
   - Cooper Kupp: 0.40 / 0.78
   D'Andre Swift holds up against both (0.75 / 0.88).
2. A covered receiver who never gets open is a bad target.
   - Those receivers still drew 1,722 targets, completed 43% of the time for 6.6 yds per target.
   - Targets who got open with the pocket intact (beat_clock 0.9+) were completed 72% of the time; free releases 77%.
   - Logistic regression of completion on beat_clock: coef 1.23, z = 19.7, n = 5,724.
   - When he gets open matters less than whether he does: completion stays at 70-75% until 3.5 s, then drops to 58%, while yards per target rise from 7.5 to 13.6.
3. The blitz takes about 0.6 s off the pocket clock.
   - Kaplan-Meier median pocket life is 3.5 s overall: 3.1 s against the blitz, 3.7 s against 4-man rushes.
   - At 2.5 s the pocket is still intact on 75% of blitzes vs. 86% of 4-man rushes.
   - Pressure or a sack is seen on only 34.5% of dropbacks, so censoring is essential.
   Who wins the clock:
   - WR: James Proche 0.70, Christian Kirk 0.68, Rondale Moore 0.67, Cooper Kupp 0.64. Bottom: Nelson Agholor 0.26, DeVante Parker 0.30.
   - TE: Dallas Goedert 0.80, Mark Andrews 0.77, T.J. Hockenson 0.77. Kyle Pitts is low at 0.59.
   - RB: D'Andre Swift 0.85, Kenneth Gainwell 0.84.
   Slot WRs get open on 62% of covered routes vs. 47% for wide WRs (median time to open 1.4 s vs. 1.7 s). A player's score correlates with his completion % when targeted at r = 0.45.

## Method
- Covered: nearest defender within 3 yds. Open: separation projected 0.7 s ahead at constant velocity is at least 3 yds, for 2 frames in a row.
- time_to_open runs from the snap to the first open frame after he was first covered. It is censored at the throw, or at the sack or end of tracking.
- Pocket intact: no rusher within 2 yds of the QB and no sack, fitted with a Kaplan-Meier curve censored at the throw. Blitz means 5+ rushers.
- beat_clock = S_pocket(time_to_open), using the play's blitz or no-blitz curve. Beat-the-Clock score = mean beat_clock over covered routes, for players with 50+ routes.

## Assumptions and limitations
- Free releases (14%; WR 18%, RB 16%, TE 4%) are left out of the score, but their share is reported.
- Covered but never open scores beat_clock = 0.
- S_pocket is close to 1 until about 2 s, so the score is driven mostly by whether a receiver gets open; a logit on "opened" alone gives about the same coefficient (1.19 vs. 1.23).
- Thresholds are heuristics. Separation is not route-aware. There is no adjustment for route depth, QB or opponent.
- The sample is 8 weeks. Treat the man/zone split as directional when a player has fewer than 30 man routes.
- Targets come from openness.py's playDescription parser. Plays marked "No Play" are dropped.
