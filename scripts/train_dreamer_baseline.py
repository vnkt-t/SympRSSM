"""
Train vanilla DreamerV3 baseline on a DMC proprioceptive task.

This script is the Phase 1 reproduction check: confirm DreamerV3 runs and
converges on DMC before replacing its RSSM with SympRSSM in Phase 2.

Target tasks (matching published baselines.yaml scores):
  - dmc_walker_walk     (~950 after 1.1M steps, dmc_proprio config)
  - dmc_cartpole_swingup (~870 after 1.1M steps, dmc_proprio config)

Requirements (install inside dreamerv3 submodule):
  pip install dm_control mujoco
  pip install -r agents/dreamerv3/requirements.txt

Usage
-----
# Smoke test — CPU, dummy task, tiny model (no dm_control needed):
  python -m scripts.train_dreamer_baseline --configs debug

# Full run — walker_walk (Kaggle T4 / Modal A10G / Vast.ai A100):
  python -m scripts.train_dreamer_baseline --configs dmc_proprio

# Full run — cartpole_swingup (faster, ~3x fewer steps to threshold):
  python -m scripts.train_dreamer_baseline \\
      --configs dmc_proprio --task dmc_cartpole_swingup

# Override logdir:
  python -m scripts.train_dreamer_baseline \\
      --configs dmc_proprio --logdir ~/logdir/baseline/{timestamp}
"""

import sys
import pathlib

# Make dreamerv3 submodule importable
_root = pathlib.Path(__file__).parent.parent
_dreamerv3_root = _root / "agents" / "dreamerv3"
sys.path.insert(0, str(_dreamerv3_root))

import dreamerv3.main as dreamer_main  # noqa: E402


def main():
    argv = list(sys.argv[1:])

    # Default logdir: project logs/dreamerv3/
    if not any("logdir" in a for a in argv):
        logdir = str(_root / "logs" / "dreamerv3" / "{timestamp}")
        argv = ["--logdir", logdir] + argv

    dreamer_main.main(argv)


if __name__ == "__main__":
    main()
