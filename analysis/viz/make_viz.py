"""Regenerate every visual in viz/ end to end:  python make_viz.py

Inputs: ../openness/receiver_openness.csv, player_summary.csv; ../clock/routes_clock.csv, player_clock.csv,
cache_clock.pkl (pocket KM curves are rebuilt from it); raw tracking for the replay play.
"""
import os
import sys
import time

# Build the clock cache (pocket times per play) if missing, e.g. on a fresh clone. ~10 s.
_CLOCK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "clock")
if not os.path.exists(os.path.join(_CLOCK, "cache_clock.pkl")):
    print("== building clock cache (first run)")
    sys.path.insert(0, _CLOCK)
    import clock
    clock.load()

import cost_of_ignoring
import play_replay
import scouting_card
import the_race
import why_ignored  # reads ../../outputs/buckets.csv (src/combine.py)

if __name__ == "__main__":
    for name, mod in [("play replay", play_replay), ("the race", the_race),
                      ("cost of ignoring", cost_of_ignoring), ("why ignored", why_ignored),
                      ("scouting card", scouting_card)]:
        t = time.time()
        print(f"== {name}")
        mod.main()
        print(f"   done in {time.time() - t:.1f}s")
    print("open index.html")
