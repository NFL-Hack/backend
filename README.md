# Open, Seen, In Time: why clear-open receivers go unthrown

> Skeleton. Numbers marked TBD get filled in once `combine.py` runs on the real `routes.csv`.

## What we built

A pipeline on the NFL Big Data Bowl tracking data (2021 season, weeks 1-8, 122 games, 8,533 dropbacks) that answers one question for every clear-open receiver who was not thrown to: **was it the clock, the QB's line of sight, or the QB?**

- `src/clock.py`: per-play pressure clock (`time_to_pressure`, `t_end`, `end_event`, `n_rushers`) and per-frame QB line of sight to every route runner.
- `src/combine.py`: joins routes, the clock, and sight; assigns each clear-open receiver a bucket; prints the headline; builds a receiver rating.
- Outputs: `play_clock.csv`, `qb_sight_frames.csv`, `buckets.csv`, `receiver_rating.csv` in `outputs/`.

## The insight

TBD: "On N plays a receiver was clear open and not targeted. X% of the time he opened after pressure arrived, Y% too late, Z% out of the QB's sight, W% the window closed, and V% the QB had him and passed."

TBD: completion % and yards/play when the clear-open receiver was targeted vs ignored.

## Who uses it

- **Offensive coaches / QB coaches:** separate "QB didn't see it" (fix: progression, eye discipline) from "QB had no time" (fix: protection, route timing).
- **Pro personnel / scouts:** `clock_rating` credits receivers who get open early enough, in the QB's sight, to be throwable, not just open eventually.
- **Defensive coordinators:** which coverages give up windows that QBs fail to use.

## Method

**Time.** Tracking is 10 Hz. Every time is seconds after the snap: `t = (frameId - snap_frame) / 10`. Snap frame = first `ball_snap` (fallback `autoevent_ballsnap`).

**End of the decision window (`t_end`, `end_event`).** First manually tagged `pass_forward`, `qb_sack`, `qb_strip_sack`, `run`, `handoff`, `lateral`, or `fumble` after the snap; fallback `autoevent_passforward`; fallback the last tracked frame (`tracking_end`).

**Pressure (`time_to_pressure`).** QB = PFF role `Pass`. Pressure = first frame, strictly before the end frame, where any PFF `Pass Rush` player is within 2.0 yd of the QB. Blank if it never happens. Check: 99% of sack plays register pressure; plays PFF credits with a hit/hurry/sack register it 63% of the time vs 8% for plays without one.

**Line of sight (`angle_off_qb`, `in_sight`).** Angle convention: tracking `o` is a compass bearing (0° = +y, clockwise, 90° = +x). The bearing from QB to receiver uses the same convention, `atan2(dx, dy)`. `angle_off_qb` is the smallest difference between the two, 0-180°. `in_sight = angle_off_qb <= 90`. Computed for every route runner on every frame from snap to end frame.

**Join.** Each route gets the sight row at `frame = snap_frame + round(10 * time_to_open)`.

**Window.** A route is *in window* when all hold:
1. he opened before pressure (or there was no pressure),
2. he opened at least 0.4 s before `t_end`,
3. he was in sight when he opened,
4. he was still open when the ball would have arrived (`open_on_arrival`).

**Buckets** (clear-open receivers only; first match wins):

| bucket | rule |
|---|---|
| `targeted` | QB threw to him |
| `pressured` | opened at or after the first pressure |
| `late` | opened < 0.4 s before `t_end` |
| `blind` | outside the QB's 90° sight cone when he opened |
| `closed` | no longer open when the ball would have arrived |
| `missed` | passed every check; the QB had him and didn't throw |

Headline unit is the play: a play is "ignored" if it had at least one clear-open receiver and none of them was targeted. Its bucket comes from the earliest-opening clear-open receiver. Completion % is over pass attempts (C / I / IN); yards/play is `playResult` over all plays in the group, sacks and scrambles included.

**Receiver rating** (`receiver_rating.csv`, receivers with ≥ 30 routes):
- `window_rate`: share of his routes that were in window (routes where he never opened count as not in window).
- `ttoe` (time-to-open over expected): expected time to open is the league median `time_to_open` for his position × alignment (`pff_positionLinedUp`) × coverage type (`pff_passCoverageType`) cell. `ttoe = median(expected over his routes) - median(his time_to_open)`. Positive = he opens faster than expected.
- `clock_rating = 100 × mean(percentile(window_rate), percentile(ttoe))`. **The two components are weighted equally.** Percentiles are among qualifying receivers.

## Assumptions

- Thresholds are assumptions, not measured truths: 2.0 yd pressure radius, 90° sight half-angle, 0.4 s minimum lead time, 30-route minimum. All live in `src/common.py`.
- Body orientation `o` stands in for where the QB is looking. Head and eye direction are not tracked.
- `time_to_open`, `open_on_arrival`, `clear_open`, and `targeted` come from `routes.csv` (built separately); this pipeline takes them as given.
- A receiver with no sight frame at his open time counts as not in sight.

## Limitations

- **Tracking ends at the throw.** We see nothing after the pass, so `open_on_arrival` is a projection, not an observation.
- **Open does not mean the QB should throw.** Down and distance, coverage rotation the QB read post-snap, a better option elsewhere, or the designed progression can all make passing on an open man correct. `missed` means "unexplained by clock or sight", not "QB error".
- Pressure is a proximity proxy. It misses pressure that forces a QB off his spot from 3+ yd and counts rushers who are engaged with a blocker inside 2 yd.
- Small cells in position × alignment × coverage make `expected_tto` noisy for rare alignments.
- 8 weeks of one season.

## Run it

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/clock.py                               # a few seconds, all 122 games
.venv/bin/python src/make_mock_routes.py                    # only until real routes.csv lands
.venv/bin/python src/combine.py --routes mock/routes.csv    # swap to outputs/routes.csv
```

Raw data is expected at `../nfl-big-data-bowl-regional-event-data-main/data` (override with `NFL_DATA_DIR`). It is not committed.

### `routes.csv` contract

One row per route runner per play:

| column | type | meaning |
|---|---|---|
| `gameId`, `playId`, `nflId` | int | keys |
| `time_to_open` | float, blank if never | seconds after snap he first became open |
| `open_on_arrival` | 0/1 | still open when a ball thrown at `time_to_open` would arrive |
| `clear_open` | 0/1 | this was a clear-open opportunity |
| `targeted` | 0/1 | he was the intended target |
