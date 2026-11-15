"""Re-run the stage-1 tracking (detector + tracking rules of stage1_unified_v2.ipynb) on one video and cache (t, xy).

The notebook cells are executed as-is (magics removed) so the paper uses exactly the notebook's algorithm.
Usage: python paper/track_stage1.py 2026-11-01_12-04-11
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
name = sys.argv[1]
nb = json.load(open(os.path.join(ROOT, "stage1_unified_v2.ipynb")))
code = lambda i: "\n".join(l for l in "".join(nb["cells"][i]["source"]).splitlines() if not l.lstrip().startswith("%"))
ns = {"__name__": "nb"}
import matplotlib; matplotlib.use("Agg")
os.chdir(ROOT)
for i in (1, 3, 7):                      # settings, circle detector, tracking rules
    exec(code(i), ns)
vid = os.path.join(ROOT, "stage1_videos", name + ".mp4")
segments = ns["track_video"](vid)
out = {}
for k, (t, xy) in enumerate(segments, 1):
    out[f"t{k}"], out[f"xy{k}"] = t, xy
os.makedirs(os.path.join(ROOT, "paper", "cache"), exist_ok=True)
np.savez(os.path.join(ROOT, "paper", "cache", name + ".npz"), **out)
print("saved", len(segments), "segment(s)")
