#!/usr/bin/env bash
# Setup script for DreamerV3 reproduction environment (Kaggle / Modal / Vast.ai).
#
# DreamerV3 requires JAX 0.4.33 (pinned in requirements.txt).
# This is separate from the SympRSSM JAX environment to avoid conflicts.
#
# Usage on Kaggle (run in a notebook cell with !):
#   !bash scripts/setup_dreamerv3_env.sh
#
# Usage on Modal / Vast.ai:
#   bash scripts/setup_dreamerv3_env.sh && python -m scripts.train_dreamer_baseline \
#       --configs dmc_proprio --task dmc_walker_walk

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMERV3_DIR="$SCRIPT_DIR/../agents/dreamerv3"

echo "=== Installing DreamerV3 dependencies (JAX 0.4.33 + CUDA) ==="

# Core dreamerv3 deps (pinned versions from their requirements.txt)
pip install \
    "jax[cuda12]==0.4.33" \
    "elements>=3.19.1" \
    "ninjax>=3.5.1" \
    "portal>=3.5.0" \
    "granular>=0.20.3" \
    "scope>=0.4.0" \
    "ruamel.yaml" \
    "einops" \
    "chex" \
    "optax" \
    "numpy<2" \
    "dm_control" \
    "mujoco" \
    -q

echo "=== DreamerV3 environment ready ==="
echo "Run baseline:"
echo "  python -m scripts.train_dreamer_baseline --configs dmc_proprio"
echo "  python -m scripts.train_dreamer_baseline --configs dmc_proprio --task dmc_cartpole_swingup"
