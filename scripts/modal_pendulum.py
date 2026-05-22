"""
Modal deployment for Phase 2 pendulum sanity checks.

Runs both simple and double pendulum experiments on an A10G GPU.
Uploads results CSV and figures back to local machine.

Usage:
  modal run scripts/modal_pendulum.py                        # both systems
  modal run scripts/modal_pendulum.py --system simple        # quick test
  modal run scripts/modal_pendulum.py --steps 10000          # longer training
"""

from __future__ import annotations

import pathlib
import sys

import modal

# ---------------------------------------------------------------------------
# Modal app + image
# ---------------------------------------------------------------------------

app = modal.App("symp-dreamer-pendulum")

_project_root = pathlib.Path(__file__).parent.parent

# Include project source in the image (Modal 1.x API: add_local_dir on Image)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "jax[cuda12]",
        "equinox>=0.11.0",
        "optax",
        "diffrax",
        "numpy",
        "matplotlib",
        "gymnasium",
    )
    .add_local_dir(
        str(_project_root / "envs"),
        remote_path="/root/SympRSSM/envs",
    )
    .add_local_dir(
        str(_project_root / "models"),
        remote_path="/root/SympRSSM/models",
    )
    .add_local_dir(
        str(_project_root / "integrators"),
        remote_path="/root/SympRSSM/integrators",
    )
    .add_local_dir(
        str(_project_root / "scripts"),
        remote_path="/root/SympRSSM/scripts",
    )
)


# ---------------------------------------------------------------------------
# Remote function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    timeout=3600,
    image=image,
)
def run_pendulum_sanity(
    system: str = "both",
    steps: int = 10000,
    rollout: int = 2000,
    n_traj: int = 80,
    hidden: int = 64,
    seed: int = 42,
) -> dict:
    """Train and evaluate SympRSSM vs RK4-RSSM on pendulum systems."""
    import sys
    import os
    import io
    import base64

    sys.path.insert(0, "/root/SympRSSM")
    os.chdir("/root/SympRSSM")

    # Run training
    from scripts.train_pendulum_sanity import main
    results = main([
        "--system", system,
        "--steps", str(steps),
        "--rollout", str(rollout),
        "--n_traj", str(n_traj),
        "--hidden", str(hidden),
        "--seed", str(seed),
    ])

    # Read back figures as base64 for local display
    figures = {}
    for system_name in (["simple", "double"] if system == "both" else [system]):
        fig_path = f"figures/pendulum_{system_name}_sanity.png"
        if os.path.exists(fig_path):
            with open(fig_path, "rb") as f:
                figures[system_name] = base64.b64encode(f.read()).decode()

    results["_figures_b64"] = figures
    return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    system: str = "both",
    steps: int = 10000,
    rollout: int = 2000,
    n_traj: int = 80,
    hidden: int = 64,
    seed: int = 42,
):
    import base64
    import os

    print(f"Launching Modal job: system={system}, steps={steps}, rollout={rollout}")

    results = run_pendulum_sanity.remote(
        system=system,
        steps=steps,
        rollout=rollout,
        n_traj=n_traj,
        hidden=hidden,
        seed=seed,
    )

    # Save figures locally
    os.makedirs("figures", exist_ok=True)
    figures = results.pop("_figures_b64", {})
    for sys_name, b64 in figures.items():
        fig_path = f"figures/pendulum_{sys_name}_sanity.png"
        with open(fig_path, "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"Saved figure: {fig_path}")

    # Print summary
    print("\n=== MODAL RESULTS ===")
    for sys_name, r in results.items():
        if isinstance(r, dict):
            print(f"  {sys_name}:")
            print(f"    SympRSSM max |ΔH|:          {r['symp_max_drift']:.3e}")
            print(f"    RK4-RSSM max |ΔH|:          {r['rk4_max_drift']:.3e}")
            print(f"    Model ratio (RK4/Symp):      {r['ratio']:.1f}x")
            print(f"    Cross-ratio (shadow H test): {r.get('gate_ratio', r['ratio']):.1f}x")
            if sys_name == "simple":
                status = "PASS ✓" if r["symp_max_drift"] < 1e-4 else "FAIL ✗"
                print(f"    Phase 2 target (< 1e-4): {status}")
            else:
                gate_ratio = r.get("gate_ratio", r["ratio"])
                status = "PASS ✓" if gate_ratio >= 10.0 else "FAIL ✗"
                print(f"    Gate 1 (≥10x ratio): {status}")
