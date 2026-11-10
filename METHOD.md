# Method, results and limitations

Details behind the summary in [README.md](README.md).

## What we built

A pipeline on the NFL Big Data Bowl tracking data (2021 season, weeks 1-8, 122 games) that answers one question for every clear-open downfield receiver who was not thrown to: **was it the clock, the QB's line of sight, or the QB?**

- `src/openness.py`: per-route separation, `window_start`, projected separation, clear-open-downfield flag, targets. Writes `routes.csv` and `openness_frames.csv`. See `OPENNESS_NOTES.md`.
- `src/clock.py`: per-play pressure clock (`time_to_pressure`, `t_end`, `end_event`, `n_rushers`) and per-frame QB line of sight to every route runner.
- `src/combine.py`: joins the two, assigns buckets, prints the headline, builds the receiver rating.
- Committed outputs: `outputs/routes.csv`, `outputs/play_clock.csv`, `outputs/buckets.csv`, `outputs/receiver_rating.csv`.
- `analysis/`: supporting work. `openness/` covers openness at the throw and which QBs and receivers it hits; `pressure/` has the clean vs pressured split; `clock/` has the Beat-the-Clock score; `viz/` has the visuals.

**Prior work.** Hypothetical completion probability (Big Data Bowl 2019, Deshpande & Evans) and ESPN's Open Score measure how open a receiver is. What's new here: timing the open window against the pocket clock and the QB's sight cone, and sorting every ignored open man into why-ignored buckets. All of this is on a dataset whose 2023 Big Data Bowl entries studied linemen.

## The insight

On **2,229** plays a receiver was clear open downfield. QBs threw to him on **619** plays (27.8%) and **ignored him on 1,610 (72.2%)**. Why he was ignored:

| bucket | share of ignored | sub-buckets |
|---|---|---|
| TOO_LATE | 32.4% | pressured 8.0%, late 24.3% |
| NOT_REAL_WINDOW | 38.9% | blind 20.7%, closed 18.3% |
| OPEN_WINDOW | 28.7% | missed 28.7% |

When the clear-open receiver was targeted: **79.3%** completion, **14.72** yards/play. When he was ignored: **64.9%** completion, **6.69** yards/play.

On 462 plays (28.7% of the ignored ones), the QB passed up a receiver who opened before pressure, at least 0.4 s before the throw, in his sight, and was still open at the throw.

**Pressure doesn't explain the ignoring** (`analysis/pressure/`). QBs hit the clear-open downfield man on 27.8% of plays with a clean pocket and 27.7% of plays with any pressure. Pressure here means tracking pressure (rusher within 2 yd) or a PFF hit, hurry or sack. With a clean pocket and 2.5 s+ to throw, they still passed on him 66% of the time, at a cost of 9.3 yards a play (16.6 vs 7.3). This split uses the 2,236-play count.

**Receiver rating.** `clock_rating` (in `outputs/receiver_rating.csv`) is the only receiver rating. It is defined under Method below. The Beat-the-Clock score in `analysis/clock/` is supporting detail; its main use is the man vs zone split (e.g. WRs get open on 31% of covered routes against man vs 66% against zone).

## Who uses it

- **Offensive coaches / QB coaches:** separate "QB didn't see it" (fix: progression, eye discipline) from "QB had no time" (fix: protection, route timing).
- **Pro personnel / scouts:** `clock_rating` credits receivers who get open early enough, in the QB's sight, to be throwable, not just open eventually.
- **Defensive coordinators:** which coverages give up windows that QBs fail to use.

## Method

**Filters.**
- Plays whose `playDescription` contains "No Play" are dropped.
- Only WR, TE and RB route runners are kept; FB counts as RB.
- Sacks and scrambles (`passResult` S / R) are left out of the ignored-play analysis and the targeted vs ignored comparison.

**Time.** Tracking is 10 Hz. Every time is seconds after the snap: `t = (frameId - snap_frame) / 10`. Snap frame = first `ball_snap` (fallback `autoevent_ballsnap`).

**End of the decision window.**
- `t_end` comes from `routes.csv`: the first pass, sack or run event after the snap.
- `clock.py` also records its own `end_frame` and `end_event`: the first tagged `pass_forward`, `qb_sack`, `qb_strip_sack`, `run`, `handoff`, `lateral` or `fumble`, else `autoevent_passforward`, else the last frame. The two ends agree on 99.6% of routes.

**Pressure (`time_to_pressure`).**
- QB = PFF role `Pass`.
- Pressure = the first frame, strictly before the end frame, where any PFF `Pass Rush` player is within 2.0 yd of the QB. Blank if that never happens.
- Check: 99% of sack plays register pressure. Plays where PFF credits a hit, hurry or sack register it 63% of the time, against 8% for plays without one.

**Line of sight (`angle_off_qb`, `in_sight`).**
- Angle convention: tracking `o` is a compass bearing (0° = +y, clockwise, 90° = +x). The bearing from QB to receiver uses the same convention, `atan2(dx, dy)`.
- `angle_off_qb` is the smallest difference between the two, 0-180°. `in_sight = angle_off_qb <= 90`.
- Check: 99.4% of targeted receivers are in sight at the throw.

**Opening time = `window_start`.**
- From `routes.csv`: the later of (a) the start of the final open stretch (sep ≥ 3 yd) that runs to the end frame and (b) the first time he is 5+ yd past the LOS.
- If it is missing (he is not open at the end frame), the fallback is `max(1.0, start of the final continuous stretch of sep ≥ 3 at or before t_end)`, from `openness_frames.csv`.
- `in_sight` is taken at the frame where `t == window_start`.
- If `window_start` falls after the clock's end frame (69 routes), he opened after the QB's window closed, and the route counts as `late`.

**Closed check.** `sep_at_throw < 3.0`: he wasn't actually open at the throw. This catches the fallback windows that had closed by the throw.

**Buckets** (clear-open downfield receivers; first match wins):

| bucket | bucket3 | rule |
|---|---|---|
| `targeted` | n/a | QB threw to him |
| `pressured` | TOO_LATE | `window_start` at or after the first pressure |
| `late` | TOO_LATE | `window_start` < 0.4 s before `t_end`, or after the clock end frame |
| `blind` | NOT_REAL_WINDOW | outside the QB's 90° sight cone at `window_start` |
| `closed` | NOT_REAL_WINDOW | `sep_at_throw < 3.0` |
| `missed` | OPEN_WINDOW | passed every check |

Completion % is over pass attempts (C / I / IN). Yards/play is the mean `playResult`.

**Receiver rating** (`receiver_rating.csv`, WR and TE with ≥ 30 routes: 152 WR, 76 TE):
- `window_rate`: share of his routes that pass every window check above: `window_start` exists, before pressure, ≥ 0.4 s before `t_end`, in sight at `window_start`, not closed.
- `ttoe` (time-to-open over expected) uses `time_to_open_after_covered`: the first time he reaches 3+ yd of separation after a defender has been within 3 yd. The raw `time_to_open` is 0 on 82% of routes, because of the cushion at the snap.
  - Expected = the league median for his position × alignment (`pff_positionLinedUp`) × coverage type (`pff_passCoverageType`) cell.
  - `ttoe = median(expected over his routes) - median(his time_to_open_after_covered)`. Positive = he opens faster than expected.
- `clock_rating = 100 × mean(percentile(window_rate), percentile(ttoe))`. **The two components are weighted equally.** Both percentiles are taken within position: WR against WR, TE against TE.

## Assumptions

- Thresholds are assumptions, not measured truths:
  - 2.0 yd pressure radius
  - 90° sight half-angle
  - 0.4 s minimum lead
  - 3.0 yd open
  - 3.0 yd at the throw to count as still open
  - 1.0 s window floor
  - 30-route minimum

  They live in `src/common.py` and at the top of `src/combine.py`.
- Body orientation `o` stands in for where the QB is looking. Head and eye direction are not tracked.
- A receiver with no sight frame at `window_start` counts as not in sight (0 routes on the current data).

## Limitations

- **Tracking ends at the throw.** "Still open" is measured at the throw, not at the catch point, so a window that closes during the ball's flight still counts as open.
- **Open does not mean the QB should throw.** Down and distance, a post-snap coverage read, a better option elsewhere, or the designed progression can all make passing on an open man correct. `missed` means "unexplained by clock or sight", not "QB error".
- **RBs excluded:** mostly short checkdowns where coverage-only separation overstates openness. RBs are still in the buckets and the headline; they are only left out of the rating.
- **`ttoe` is coarse.** Times are in 0.1 s steps, so many receivers tie on `ttoe`, and small samples (30-40 routes) can reach the top of the list.
- Pressure is a proximity proxy. It misses pressure that moves the QB off his spot from 3+ yd away, and it counts rushers engaged with a blocker inside 2 yd.
- Small position × alignment × coverage cells make `expected` noisy for rare alignments.
- **Two pocket clocks.** The pocket-survival curve in `analysis/clock/` (and "The Race" visual) counts pressure at or before the throw, plus sacks. `src/clock.py` counts it strictly before the end frame. Its survival times are not comparable with `time_to_pressure` here.
- Openness is geometry only: no route concept, no leverage, no ball placement.
- The results describe what happened, not what caused it.
- 8 weeks of one season.

## Run it

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/openness.py     # routes.csv + openness_frames.csv
.venv/bin/python src/clock.py        # play_clock.csv + qb_sight_frames.csv
.venv/bin/python src/combine.py      # buckets.csv + receiver_rating.csv + headline
cd analysis/viz && ../../.venv/bin/python make_viz.py   # all visuals
```

Raw data is expected at `../nfl-big-data-bowl-regional-event-data-main/data` (override with `NFL_DATA_DIR`). It is not committed.
