import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
from orbax.checkpoint import PyTreeCheckpointer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)          # folder with config.yaml + policies/
    ap.add_argument("--step", type=int, required=True)
    args = ap.parse_args()

    # Orbax OCDBT checkpoints are under: policies/<step>/default/
    ckpt_dir = os.path.join(args.path, "policies", str(args.step), "default")
    print("Checkpoint dir:", ckpt_dir)

    ckpt = PyTreeCheckpointer().restore(ckpt_dir)

    print("\nTop-level type:", type(ckpt))
    if isinstance(ckpt, dict):
        print("Top-level keys:", list(ckpt.keys())[:50])
        # If it looks like a TrainState dict, show param tree root keys too
        if "params" in ckpt and isinstance(ckpt["params"], dict):
            print("\nparams keys:", list(ckpt["params"].keys())[:50])
    else:
        # Sometimes it restores a TrainState-like object
        print("Has attrs:", [a for a in ["params", "opt_state", "step"] if hasattr(ckpt, a)])
        if hasattr(ckpt, "params") and isinstance(ckpt.params, dict):
            print("\nparams keys:", list(ckpt.params.keys())[:50])

if __name__ == "__main__":
    main()
