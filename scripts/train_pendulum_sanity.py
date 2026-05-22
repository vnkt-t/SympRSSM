"""
Phase 2 pendulum sanity checks.

Tests the core claim: symplectic integration (Yoshida-4) of a learned
Hamiltonian preserves the learned energy, while RK4 integration drifts.

Experiments:
  1. Simple pendulum  — energy drift < 1e-4 over 1000 steps (SympRSSM target)
  2. Double pendulum  — SympRSSM shows ≥10x lower drift than RK4-RSSM (Gate 1)

Usage
-----
  # Local CPU test (simple pendulum only, fast):
  python -m scripts.train_pendulum_sanity --system simple --steps 2000

  # Full Gate 1 check (both systems):
  python -m scripts.train_pendulum_sanity --system double --steps 5000

  # Via Modal (GPU, both):
  modal run scripts/modal_pendulum.py
"""

from __future__ import annotations

import argparse
import os
import time
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from models.hamiltonian_net import HamiltonianNet
from integrators.yoshida4 import yoshida4_step
from models.rk4_rssm import rk4_step


# ---------------------------------------------------------------------------
# Integration wrappers
# ---------------------------------------------------------------------------

def symp_step(
    ham: HamiltonianNet,
    q: jnp.ndarray,
    p: jnp.ndarray,
    action: jnp.ndarray,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One Yoshida-4 step using the learned Hamiltonian."""
    # For separable H = T(p) + V(q) + Phi(q,a):
    #   grad_V only depends on q (not p)
    #   grad_T only depends on p (not q)
    grad_V = lambda q_: ham.grad_q(q_, p, action)
    grad_T = lambda p_: ham.grad_p(q, p_, action)
    return yoshida4_step(q, p, grad_V, grad_T, dt)


def rk4_step_learned(
    ham: HamiltonianNet,
    q: jnp.ndarray,
    p: jnp.ndarray,
    action: jnp.ndarray,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One RK4 step using the learned Hamiltonian (non-symplectic ablation)."""
    return rk4_step(q, p, ham, action, dt)


# ---------------------------------------------------------------------------
# Rollout and energy drift
# ---------------------------------------------------------------------------

def rollout_n_steps(
    ham: HamiltonianNet,
    q0: jnp.ndarray,
    p0: jnp.ndarray,
    action: jnp.ndarray,
    dt: float,
    n_steps: int,
    step_fn: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Open-loop rollout for n_steps. Returns (q_traj, p_traj) of shape (n_steps, dim)."""
    def scan_fn(carry, _):
        q, p = carry
        q_new, p_new = step_fn(ham, q, p, action, dt)
        return (q_new, p_new), (q_new, p_new)

    _, (q_traj, p_traj) = jax.lax.scan(scan_fn, (q0, p0), None, length=n_steps)
    return q_traj, p_traj


def energy_drift_curve(
    ham: HamiltonianNet,
    q_traj: jnp.ndarray,
    p_traj: jnp.ndarray,
    action: jnp.ndarray,
) -> jnp.ndarray:
    """Compute |H_theta(q_t, p_t) - H_theta(q_0, p_0)| along a rollout trajectory.

    Measures conservation of the *learned* Hamiltonian (not the true one).
    For Yoshida-4: bounded by shadow Hamiltonian theorem.
    For RK4: monotonically drifting.
    """
    H0 = ham(q_traj[0], p_traj[0], action)
    H_vals = jax.vmap(lambda q, p: ham(q, p, action))(q_traj, p_traj)
    return jnp.abs(H_vals - H0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def k_step_loss(
    ham: HamiltonianNet,
    q_window: jnp.ndarray,
    p_window: jnp.ndarray,
    action: jnp.ndarray,
    dt: float,
    k: int,
    step_fn: Callable,
) -> jnp.ndarray:
    """k-step rollout MSE loss on a window of (k+1) consecutive states.

    Args:
        q_window: (k+1, q_dim) — states from t to t+k
        p_window: (k+1, p_dim)
    """
    q, p = q_window[0], p_window[0]
    loss = jnp.zeros(())

    def scan_fn(carry, i):
        q, p, loss = carry
        q_new, p_new = step_fn(ham, q, p, action, dt)
        step_loss = jnp.mean((q_new - q_window[i + 1]) ** 2)
        step_loss += jnp.mean((p_new - p_window[i + 1]) ** 2)
        return (q_new, p_new, loss + step_loss), None

    (_, _, total_loss), _ = jax.lax.scan(scan_fn, (q, p, loss), jnp.arange(k))
    return total_loss / k


@eqx.filter_jit
def train_step(
    ham: HamiltonianNet,
    opt_state,
    q_batch: jnp.ndarray,
    p_batch: jnp.ndarray,
    action: jnp.ndarray,
    dt: float,
    k: int,
    step_fn: Callable,
    optimizer,
) -> tuple[HamiltonianNet, object, jnp.ndarray]:
    """Single gradient step on a batch of trajectory windows."""

    def loss_fn(ham):
        per_sample = jax.vmap(
            lambda qw, pw: k_step_loss(ham, qw, pw, action, dt, k, step_fn)
        )(q_batch, p_batch)
        return jnp.mean(per_sample)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(ham)
    updates, opt_state_new = optimizer.update(
        grads, opt_state, eqx.filter(ham, eqx.is_array)
    )
    ham_new = eqx.apply_updates(ham, updates)
    return ham_new, opt_state_new, loss


def make_windows(
    q_data: np.ndarray,
    p_data: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice trajectory data into overlapping windows of length k+1."""
    n_traj, traj_len_plus1, dim = q_data.shape
    windows_q, windows_p = [], []
    for t in range(n_traj):
        for start in range(traj_len_plus1 - k):
            windows_q.append(q_data[t, start : start + k + 1])
            windows_p.append(p_data[t, start : start + k + 1])
    return np.stack(windows_q), np.stack(windows_p)


def train(
    ham: HamiltonianNet,
    q_train: jnp.ndarray,
    p_train: jnp.ndarray,
    action: jnp.ndarray,
    dt: float,
    k: int,
    n_steps: int,
    lr: float,
    batch_size: int,
    step_fn: Callable,
    key: jax.random.PRNGKey,
    log_every: int = 200,
) -> tuple[HamiltonianNet, list]:
    """Train HamiltonianNet to predict k-step rollouts."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )
    opt_state = optimizer.init(eqx.filter(ham, eqx.is_array))

    windows_q, windows_p = make_windows(np.array(q_train), np.array(p_train), k)
    n_windows = len(windows_q)
    losses = []

    for step in range(n_steps):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (batch_size,), 0, n_windows)
        q_batch = jnp.array(windows_q[np.array(idx)])
        p_batch = jnp.array(windows_p[np.array(idx)])

        ham, opt_state, loss = train_step(
            ham, opt_state, q_batch, p_batch, action, dt, k, step_fn, optimizer
        )
        losses.append(float(loss))

        if (step + 1) % log_every == 0:
            print(f"  step {step+1:5d}/{n_steps}  loss={np.mean(losses[-log_every:]):.4e}")

    return ham, losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_energy_drift(
    ham: HamiltonianNet,
    q_test: jnp.ndarray,
    p_test: jnp.ndarray,
    action: jnp.ndarray,
    dt: float,
    rollout_steps: int,
    step_fn: Callable,
) -> jnp.ndarray:
    """Mean energy drift curve over test trajectories (rollout_steps long)."""
    n_test = q_test.shape[0]
    drift_curves = []

    for i in range(n_test):
        q0, p0 = q_test[i, 0], p_test[i, 0]
        q_traj, p_traj = rollout_n_steps(ham, q0, p0, action, dt, rollout_steps, step_fn)
        # Prepend IC for energy_drift_curve
        q_full = jnp.concatenate([q0[None], q_traj], axis=0)
        p_full = jnp.concatenate([p0[None], p_traj], axis=0)
        drift_curves.append(energy_drift_curve(ham, q_full, p_full, action))

    return jnp.stack(drift_curves).mean(axis=0)  # (rollout_steps+1,)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(config: dict) -> dict:
    """Run both SympRSSM and RK4-RSSM on a pendulum system, return metrics."""
    system = config["system"]
    q_dim = config["q_dim"]
    p_dim = config["p_dim"]
    a_dim = 1  # always 1 (action=zeros for passive dynamics)
    dt = config["dt"]
    n_traj = config["n_traj"]
    traj_len = config["traj_len"]
    n_steps = config["n_steps"]
    rollout_steps = config["rollout_steps"]
    hidden = config["hidden"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    k = config["k"]
    key = jax.random.PRNGKey(config.get("seed", 42))

    action = jnp.zeros(a_dim)

    # --- Generate data ---
    print(f"\n=== {system.upper()} PENDULUM (q_dim={q_dim}) ===")
    print(f"Generating {n_traj} trajectories × {traj_len} steps...")
    t0 = time.time()

    key, dkey = jax.random.split(key)
    if system == "simple":
        from envs.pendulum import generate_simple_pendulum_data
        q_data, p_data = generate_simple_pendulum_data(n_traj, traj_len, dt, dkey)
    else:
        from envs.pendulum import generate_double_pendulum_data
        q_data, p_data = generate_double_pendulum_data(n_traj, traj_len, dt, dkey)

    print(f"  Data: {q_data.shape} in {time.time()-t0:.1f}s")

    # Train/test split
    n_train = int(0.8 * n_traj)
    q_train, p_train = q_data[:n_train], p_data[:n_train]
    q_test, p_test = q_data[n_train:], p_data[n_train:]

    results = {}

    for model_name, step_fn in [("symp", symp_step), ("rk4", rk4_step_learned)]:
        print(f"\n--- Training {model_name.upper()}-RSSM ---")
        key, mkey = jax.random.split(key)
        ham = HamiltonianNet(q_dim=q_dim, p_dim=p_dim, a_dim=a_dim, hidden=hidden, key=mkey)

        t0 = time.time()
        ham, losses = train(
            ham, q_train, p_train, action, dt,
            k=k, n_steps=n_steps, lr=lr, batch_size=batch_size,
            step_fn=step_fn, key=key, log_every=max(1, n_steps // 5),
        )
        train_time = time.time() - t0

        drift_curve = evaluate_energy_drift(
            ham, q_test, p_test, action, dt, rollout_steps, step_fn
        )
        max_drift = float(jnp.max(drift_curve))
        final_drift = float(drift_curve[-1])
        mean_drift = float(jnp.mean(drift_curve[rollout_steps // 2:]))  # latter half

        print(f"  Train time: {train_time:.1f}s")
        print(f"  Max |ΔH|:   {max_drift:.3e}")
        print(f"  Final |ΔH|: {final_drift:.3e}")
        print(f"  Mean |ΔH| (latter 500 steps): {mean_drift:.3e}")

        results[model_name] = {
            "drift_curve": np.array(drift_curve),
            "max_drift": max_drift,
            "final_drift": final_drift,
            "mean_drift": mean_drift,
            "losses": losses,
        }

    # --- Gate evaluation ---
    ratio = results["rk4"]["mean_drift"] / max(results["symp"]["mean_drift"], 1e-20)
    print(f"\n=== GATE 1 RESULT ({system.upper()}) ===")
    print(f"  SympRSSM mean |ΔH|: {results['symp']['mean_drift']:.3e}")
    print(f"  RK4-RSSM mean |ΔH|: {results['rk4']['mean_drift']:.3e}")
    print(f"  Ratio (RK4/Symp):   {ratio:.1f}x")

    if system == "simple":
        target = results["symp"]["max_drift"] < 1e-4
        print(f"  Phase 2 target (max |ΔH| < 1e-4): {'PASS ✓' if target else 'FAIL ✗'}")
    else:
        gate = ratio >= 10.0
        print(f"  Gate 1 (≥10x lower drift for Symp): {'PASS ✓' if gate else 'FAIL ✗ (debug integrator)'}")

    results["ratio"] = ratio
    results["system"] = system
    return results


def save_results(results: dict, outdir: str = "results"):
    """Save drift curves to CSV and plot to figures/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    system = results["system"]

    # CSV
    symp_curve = results["symp"]["drift_curve"]
    rk4_curve = results["rk4"]["drift_curve"]
    n = min(len(symp_curve), len(rk4_curve))
    data = np.column_stack([np.arange(n), symp_curve[:n], rk4_curve[:n]])
    np.savetxt(
        f"{outdir}/pendulum_{system}_energy_drift.csv", data,
        header="step,symp_drift,rk4_drift", delimiter=",", comments=""
    )
    print(f"  Saved: {outdir}/pendulum_{system}_energy_drift.csv")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.semilogy(symp_curve[:n], label=f"SympRSSM (Yoshida-4)", color="blue")
    ax.semilogy(rk4_curve[:n], label=f"RK4-RSSM (ablation)", color="red", linestyle="--")
    if system == "simple":
        ax.axhline(1e-4, color="k", linestyle=":", alpha=0.5, label="target 1e-4")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("|H_θ(t) - H_θ(0)|")
    ax.set_title(f"{system.title()} Pendulum — Learned Energy Drift")
    ax.legend()

    ax = axes[1]
    ax.semilogy(results["symp"]["losses"], label="SympRSSM", color="blue", alpha=0.7)
    ax.semilogy(results["rk4"]["losses"], label="RK4-RSSM", color="red", alpha=0.7, linestyle="--")
    ax.set_xlabel("training step")
    ax.set_ylabel("k-step MSE loss")
    ax.set_title("Training Loss")
    ax.legend()

    plt.tight_layout()
    fig_path = f"figures/pendulum_{system}_sanity.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


def main(args=None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["simple", "double", "both"], default="both")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--rollout", type=int, default=1000)
    parser.add_argument("--n_traj", type=int, default=80)
    parser.add_argument("--traj_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    cfg = parser.parse_args(args)

    systems = ["simple", "double"] if cfg.system == "both" else [cfg.system]
    all_results = {}

    for system in systems:
        q_dim = 1 if system == "simple" else 2
        config = {
            "system": system,
            "q_dim": q_dim,
            "p_dim": q_dim,
            "dt": 0.1 if system == "simple" else 0.05,
            "n_traj": cfg.n_traj,
            "traj_len": cfg.traj_len,
            "n_steps": cfg.steps,
            "rollout_steps": cfg.rollout,
            "hidden": cfg.hidden,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "k": 4,
            "seed": cfg.seed,
        }
        results = run_experiment(config)
        save_results(results)
        all_results[system] = {
            "symp_max_drift": results["symp"]["max_drift"],
            "rk4_max_drift": results["rk4"]["max_drift"],
            "ratio": results["ratio"],
        }

    print("\n=== SUMMARY ===")
    for sys_name, r in all_results.items():
        print(f"  {sys_name}: symp={r['symp_max_drift']:.2e}, "
              f"rk4={r['rk4_max_drift']:.2e}, ratio={r['ratio']:.1f}x")

    return all_results


if __name__ == "__main__":
    main()
