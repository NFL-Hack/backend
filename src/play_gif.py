"""Render the landing-page animated play (figures/play.html) as a GIF for the README.

Usage:
  python src/play_gif.py --data ../nfl-big-data-bowl-regional-event-data-main/data --play 2021101711,231
Writes figures/play.gif (and refreshes figures/play.html). Needs kaleido and ffmpeg.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
from dashboard import animated_play  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--play", default="2021101711,231")
    ap.add_argument("--out", default="figures/play.gif")
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=10)       # tracking is 10 Hz, so 10 fps = real time
    ap.add_argument("--hold", type=float, default=2.0)   # seconds to hold the last frame
    a = ap.parse_args()

    fig = animated_play(a, tuple(int(v) for v in a.play.split(",")), "figures/play.html")
    # No buttons or slider in a GIF; reclaim the space they used.
    fig.layout.updatemenus = None
    fig.layout.sliders = None
    fig.update_layout(margin=dict(l=16, r=16, t=70, b=90), height=700, legend=dict(y=-0.02))

    tmp = tempfile.mkdtemp()
    try:
        frames = list(fig.frames)
        for i, fr in enumerate(frames):
            f = go.Figure(data=fr.data, layout=fig.layout)
            f.write_image(os.path.join(tmp, f"f{i:04d}.png"), width=a.width, height=700, scale=1)
        last = os.path.join(tmp, f"f{len(frames) - 1:04d}.png")
        for k in range(int(a.hold * a.fps)):
            shutil.copy(last, os.path.join(tmp, f"f{len(frames) + k:04d}.png"))
        vf = (f"fps={a.fps},split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];"
              f"[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
                        "-i", os.path.join(tmp, "f%04d.png"), "-vf", vf, "-loop", "0", a.out], check=True)
    finally:
        shutil.rmtree(tmp)
    print("saved", a.out, f"({len(frames)} frames, {os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
