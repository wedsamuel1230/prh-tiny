"""Kaggle kernel: train the tiny two-modality JEPA at a scale a laptop will not do, export the C.

Pushed by `make kaggle` (see kaggle/pack.sh). Reads the code from the `prh-tiny-code` dataset and
writes `runs/tiny.pt` plus the generated `firmware/` files into /kaggle/working.
"""
import json
import os
import subprocess
import sys

SRC = "/kaggle/input/prh-tiny-code"
if not os.path.isdir(SRC):
    raise SystemExit(f"dataset not mounted at {SRC}; attach prh-tiny-code to this kernel")

sys.path.insert(0, SRC)
os.chdir("/kaggle/working")

STEPS = int(os.environ.get("PRH_STEPS", 6000))
N = int(os.environ.get("PRH_N", 12000))
BATCH = int(os.environ.get("PRH_BATCH", 128))

print(json.dumps({"steps": STEPS, "n": N, "batch": BATCH}), flush=True)

subprocess.run([sys.executable, f"{SRC}/train.py", "--steps", str(STEPS), "--n", str(N),
                "--batch", str(BATCH), "--out", "runs/tiny.pt", "--device", "auto"], check=True)
subprocess.run([sys.executable, f"{SRC}/export_c.py", "runs/tiny.pt"], check=True)

print(json.dumps({"files": sorted(p for p in os.listdir("/kaggle/working"))}), flush=True)
