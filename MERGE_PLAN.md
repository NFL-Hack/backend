# Merge plan: "Open, Seen, In Time" + "The Open Man They Ignored"

Branch: `analysis-merge`. The deadline is **2:00 PM**, so do the steps in priority order and stop when time runs out.

## What's here

| Path | What it is |
|---|---|
| `src/clock.py`, `src/combine.py` (Krish, on main) | Pressure clock, QB sight cone, ignored-receiver buckets (pressured / late / blind / closed / missed), `clock_rating` |
| `src/openness.py` + `outputs/routes.csv` (Yatharth, on main) | Shared-spec routes file, v2 (see `OPENNESS_NOTES.md`) |
| `analysis/openness/` | Openness at the throw: separation projected 0.7 s ahead (accounts for defenders closing in), target parsed from playDescription (99.7% match), "clearly open downfield" flag (2,236 plays) |
| `analysis/clock/` | Kaplan–Meier pocket survival curve (handles plays where the QB threw before any pressure), time to open after being covered, Beat-the-Clock score, man vs zone splits |
| `analysis/pressure/` | Clean vs pressured split (tracking + PFF) of the "ignored open man" result |
| `analysis/viz/` | 4 visuals + `index.html`: play replay (HTML/GIF), The Race, Cost of Ignoring, interactive scouting card. Regenerate with `python analysis/viz/make_viz.py` |

The raw data path comes from `NFL_DATA_DIR`, the same convention as `src/common.py`. `.pkl` caches aren't committed; the scripts rebuild them.

## The combined story (one README, one landing page)

1. **Hook: the play replay.** A clean pocket, a clearly open downfield receiver, and the QB throws to the covered man.
2. **It's common and it's costly** (`cost_of_ignoring.png`). On 2,236 plays a receiver 5+ yds downfield was clearly the most open, and QBs looked elsewhere 72% of the time.
   - Hit him: 79% completions, 14.7 yds/play. Elsewhere: 65%, 6.7 yds/play.
   - Clean pocket with 2.5 s+: still ignored 66% of the time; 85% vs 66% completions, +9.3 yds/play.
   - QBs hit him 28% of the time whether the pocket was clean or pressured.
3. **Why it happens: Krish's buckets** (NEW chart, see task 2). A funnel from all ignored open men → pressured → late → out of sight → closed → **"QB had him and passed"**. This is the payoff, and it agrees with the pressure split.
4. **The race against the pocket** (`the_race.png`). The pocket survival curve against time to open by position: TEs and RBs about 1.5–1.8 s, the typical WR 2.8 s, half of pockets broken by 3.5 s (3.1 s against a blitz).
5. **The tool: scouting card** (`scouting_card.html`). Use **Krish's `clock_rating`** as the single receiver rating (it compares like with like: position × alignment × coverage). Keep Beat-the-Clock as supporting detail only. Don't present two different "clock ratings".

## Tasks for Krish's bot, in priority order

### 1. Make `combine.py` run on the real `outputs/routes.csv`. BLOCKER
`load_routes` expects the contract columns `clear_open`, `targeted`, `time_to_open`, `open_on_arrival`. The real file uses different names. Map them in `load_routes` and leave `routes.csv` as it is:

| combine.py expects | take from routes.csv | why |
|---|---|---|
| `clear_open` | `is_clear_open_downfield` | team decision 3 (2,236 definition; 2,229 after shared filters, since 7 plays have no snap event) |
| `targeted` | `is_target` | |
| `time_to_open` for **buckets** (when the clear-open man opened) | `window_start` | team decision 2; blank if he wasn't open at the throw |
| `time_to_open` for **rating** (`ttoe`, `window_rate`) | `time_to_open_after_covered` | team decision 1; the raw `time_to_open` is 0 on 82% of routes |
| `open_on_arrival` / `closed` bucket | drop it from the buckets | team decision 4; the 7 yd/s + 1.5 yd radius marks about 70% of targets "not open" |

- Optional: use `proj_sep_0_7 >= 3` as a softer stand-in for `closed`.
- Then run `python src/combine.py --routes outputs/routes.csv` and paste the headline numbers into the README TBDs.

### 2. Bucket funnel chart, `analysis/viz/why_ignored.png`
- A horizontal funnel or stacked bar of the ignored clear-open plays by bucket, ending at `missed` ("in sight, in time, no pressure: QB passed him up").
- Title states the takeaway, e.g. "X% of ignored open men were in the QB's sight, before pressure, with time to throw."
- Add it to `analysis/viz/index.html` between Cost of Ignoring and The Race, and add the script to `make_viz.py`.

### 3. Swap the hero play to a `missed`-bucket play, and draw the sight cone
- Pick from `buckets.csv`: bucket == `missed`, the clear-open man 5+ yds downfield and not near the sideline (10 < y < 43), the target tightly covered, recognizable players.
- Set `GAME_ID`/`PLAY_ID` in `analysis/viz/play_replay.py` (it currently uses PHI–CAR 2021101001/2250, Hurts → Ertz with Quez Watkins open).
- Add Krish's QB sight cone: a ±90° wedge from the QB's `o` (from `qb_sight_frames.csv`), shaded, on every frame, so viewers can see the open man inside the QB's view.

### 4. Scouting card: show `clock_rating`
- In `analysis/viz/scouting_card.py`, add `clock_rating`, `window_rate` and `ttoe` from `outputs/receiver_rating.csv` (join on nflId) as the headline number of each card. Leave the existing panels as they are.

### 5. README, the final version for submission (3–5 sentence summary at the top)
- Use the story above. One play count everywhere: 2,236 (or 2,229 with a footnote).
- Name prior work in one line: hypothetical completion probability (BDB 2019, Deshpande & Evans) and ESPN's Open Score. **What's new:** timing openness against a censored pocket clock, the QB sight cone, and why-ignored buckets, on a dataset whose 2023 entries all studied linemen.
- Limitations:
  - Tracking ends about 0.5 s after the throw.
  - Openness is geometry only.
  - `o` stands in for where the QB is looking.
  - The results describe what happened, not what caused it.
  - 8 weeks of one season.
- Link `analysis/viz/index.html` (GitHub Pages) as the single hosted URL.

## Don't
- Don't rename or regenerate `outputs/routes.csv` columns; other code depends on them.
- Don't commit `.pkl` caches or `openness_frames.csv` (42 MB).
