# v2 changes (team decisions)
1. **Rating uses `time_to_open_after_covered`.** The column is unchanged.
2. **New `window_start` (s), per route** = the LATER of:
   - (a) the start t of the final open stretch: the last contiguous run of frames with sep ≥ 3.0 that ends at the end frame. It is NaN if the receiver is not open at the end frame.
   - (b) the first t with normalized x − LOS ≥ 5.

   **Fallback:** if (b) never happens (he never gets 5+ yds downfield between the snap and the end frame), window_start = max(a, 1.0). A missing sep (no Coverage defender in a frame) counts as not open.
3. **`is_clear_open_downfield` is back to the original definition**, joined from `openness/receiver_openness.csv`: `clear_open_dn & play_has_target`.
   - That definition: the most open by projected 0.7 s separation among receivers 5+ yds downfield, by 2+ yds over the next downfield receiver.
   - The original has 2,236 such plays; 2,229 of them are in routes.csv. The other 7 have no ball_snap event, so the shared spec drops them.
   - Targeted: **27.8%**.
4. **New `proj_sep_0_7`:** the original `openness`, the separation projected 0.7 s ahead at the original throw frame.
   - It is NaN where the original has no row (sacks/runs and plays the original skipped). It covers 88.9% of routes and 100% of pass-play routes.
   - Correlation with sep_at_throw: 0.79.
   - `open_on_arrival` is kept in the file, but the team is dropping it from the buckets logic.
5. **Filters:**
   - "No Play" plays dropped (488 plays, 2,208 routes).
   - FB → RB.
   - QB/C/T route-runner rows dropped (35).
   - The same filters apply to openness_frames.csv.

**v2 sanity (full data)**
- **routes:** 39,398 → **37,155** rows; 8,533 → 8,045 plays.
- **frames:** 1,256,629 → **1,176,600**.
- **Positions:** WR 21,651, TE 8,542, RB 6,962.
- **Clear-open-dn plays:** 2,229 before the filters and 2,229 after.
  - With a matched target: 2,229 before and 2,228 after. The lost play is a Taysom Hill (QB) target, so the QB row filter removes its target row.
  - Targeted: 27.8%.

| window_start | non-null | median | fallback used (% of non-null) |
|---|---|---|---|
| WR | 41.0% | 1.9 s | 29% |
| TE | 43.6% | 1.7 s | 54% |
| RB | 75.0% | 1.0 s | 93% |

The column order is the same as before, with `time_to_open_after_covered`, `window_start` and `proj_sep_0_7` at the end.

> **Superseded by v2:** the "541 plays" clear-open number and the No Play item below now match v2 (2,229 plays; No Play is filtered out).

---

# Openness pipeline (shared spec): outputs and notes for the team

## Files
- `routes.csv`: 39,398 routes on 8,533 plays. The contract columns are in order, plus `time_to_open_after_covered` at the end.
- `openness_frames.csv`: 1.26M rows (gameId, playId, nflId, t, sep).
- `routes_2games.csv` and `openness_frames_2games.csv`: small versions for quick testing.
- `src/openness.py`: the code. Run `python src/openness.py [--games 2]`; the full run takes about 5 s.
- Checks: the target receiver matched on 99.7% of plays, and the normalization and separation checks passed.

## Read before building on it
1. **The spec's `time_to_open` doesn't work.** It is exactly 0 on **82%** of routes (99% of RBs, 83% of TEs and 77% of WRs count as "open" at the snap).
   - Use `time_to_open_after_covered` instead: the first moment the receiver reaches 3+ yards of separation *after* a defender has been within 3 yards of him.
   - Medians: WR 2.2 s, TE 2.1 s, RB 2.9 s.
2. **The spec's "clearly open downfield" is much stricter than the earlier analysis.** It finds only **541 plays** (7.1% of throws), against 2,236 earlier. That leaves only about 350 ignored plays to split into three buckets.
   - The spec compares raw separation now; the earlier analysis projected 0.7 s ahead.
   - The spec makes the downfield receiver beat receivers at *any* depth by 2 yards; the earlier analysis only compared him with other downfield receivers.
   - Pick one definition, so the README headline numbers (2,236 plays, 72% ignored, +9.3 yards) match `routes.csv`.
3. **"Open on arrival" is harsh.** Only 30% of targets count as open on arrival, so most ignored receivers will land in NOT_REAL_WINDOW. It does separate outcomes (72% vs 59% completions). Consider lowering the 7 yd/s closing speed or the 1.5 yd buffer.

## Smaller items
- **Penalty plays are included:** 488 plays wiped out by a penalty ("No Play" in playDescription). Filter them out.
- **Sacks and runs:** `open_on_arrival` is blank for them, not False.
- **Defenders:** `sep` counts Coverage defenders only, as the spec says, so pass rushers are ignored. That makes RBs and TEs near the line look more open than they are.
