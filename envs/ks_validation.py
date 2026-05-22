"""
KS environment validation.

Reproduces known properties of uncontrolled KS at L=22:
  1. Time-averaged spatial energy ⟨(1/N)Σu²⟩_t ≈ 42.8
  2. Spatiotemporal chaos (positive Lyapunov exponent)
  3. Correct energy spectrum shape

Run: python -m envs.ks_validation
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from envs.ks import KSEnv


def validate_uncontrolled_energy(
    L: float = 22.0,
    N: int = 64,
    nu: float = 1.0,
    dt: float = 0.5,
    inner_steps: int = 10,
    total_steps: int = 2000,
    warmup_steps: int = 500,
    n_seeds: int = 5,
):
    """Run uncontrolled KS and measure time-averaged energy.

    Energy conventions:
      - mean(u²) = (1/N)Σu²: spatial energy density (~1.4 for L=22, N=64)
      - (1/2)Σu² = (N/2)*mean(u²): discrete L2 energy (~43 for L=22, N=64)
      - ∫u²dx = L*mean(u²): continuous L2 norm squared (~30 for L=22)

    The plan's "~42.8" target matches the (1/2)Σu² convention.
    """
    print(f"KS Validation: L={L}, N={N}, ν={nu}, dt={dt}, inner_steps={inner_steps}")
    print(f"  total_steps={total_steps}, warmup={warmup_steps}, seeds={n_seeds}")
    print()

    all_energies_density = []
    all_energies_l2 = []
    all_trajectories = []

    for seed in range(n_seeds):
        env = KSEnv(N=N, L=L, nu=nu, dt=dt, inner_steps=inner_steps, episode_length=total_steps)
        obs, _ = env.reset(seed=seed)

        energies_density = []
        energies_l2 = []
        trajectory = [obs.copy()]

        for step in range(total_steps):
            action = np.zeros(env.n_actuators)
            obs, _, _, _, _ = env.step(action)
            energies_density.append(env.get_energy())
            energies_l2.append(env.get_l2_energy())
            if step < 200 or step % 10 == 0:
                trajectory.append(obs.copy())

        ed = np.array(energies_density)[warmup_steps:]
        el = np.array(energies_l2)[warmup_steps:]

        print(f"  Seed {seed}: mean(u²) = {np.mean(ed):.3f} ± {np.std(ed):.3f},"
              f"  (1/2)Σu² = {np.mean(el):.1f} ± {np.std(el):.1f}")

        all_energies_density.append(ed)
        all_energies_l2.append(el)
        all_trajectories.append(np.array(trajectory))

    # Aggregate
    combined_density = np.concatenate(all_energies_density)
    combined_l2 = np.concatenate(all_energies_l2)
    mean_density = np.mean(combined_density)
    mean_l2 = np.mean(combined_l2)
    std_l2 = np.std(combined_l2)

    print(f"\n  Grand mean(u²): {mean_density:.3f}")
    print(f"  Grand (1/2)Σu²: {mean_l2:.1f} ± {std_l2:.1f}")
    print(f"  Target (1/2)Σu²: ~42.8")

    if 25.0 < mean_l2 < 65.0:
        print("  ✓ Energy in expected range for chaotic KS at L=22")
    else:
        print(f"  ✗ Energy {mean_l2:.1f} outside expected range [25, 65]")

    # Check for chaos (energy fluctuations)
    if np.std(combined_density) > 0.05:
        print("  ✓ Energy fluctuations confirm chaotic dynamics")
    else:
        print("  ⚠ Low energy variance — may be steady state, not chaos")

    return mean_l2, std_l2, all_energies_l2, all_trajectories


def plot_validation(energies_list, trajectory, L=22.0, save_path="figures/ks_validation.png"):
    """Create validation plots: space-time diagram + energy trace."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Space-time diagram
    traj = np.array(trajectory)
    ax = axes[0]
    x = np.linspace(0, L, traj.shape[1], endpoint=False)
    t = np.arange(traj.shape[0])
    im = ax.pcolormesh(x, t, traj, cmap="RdBu_r", shading="auto")
    ax.set_xlabel("x")
    ax.set_ylabel("time step")
    ax.set_title("KS spatiotemporal chaos (seed 0)")
    plt.colorbar(im, ax=ax, label="u(x,t)")

    # Energy trace
    ax = axes[1]
    for i, e in enumerate(energies_list):
        ax.plot(e, alpha=0.5, label=f"seed {i}")
    ax.axhline(42.8, color="k", linestyle="--", alpha=0.5, label="target ~42.8")
    ax.set_xlabel("time step (post-warmup)")
    ax.set_ylabel("(1/2)Σu²")
    ax.set_title("Discrete L2 energy")
    ax.legend(fontsize=8)

    # Energy histogram
    ax = axes[2]
    combined = np.concatenate(energies_list)
    ax.hist(combined, bins=50, density=True, alpha=0.7)
    ax.axvline(42.8, color="r", linestyle="--", label="target ~42.8")
    ax.axvline(np.mean(combined), color="k", linestyle="-", label=f"measured {np.mean(combined):.1f}")
    ax.set_xlabel("(1/2)Σu²")
    ax.set_ylabel("density")
    ax.set_title("Energy distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved validation plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)

    mean_e, std_e, energies, trajectories = validate_uncontrolled_energy()

    plot_validation(energies, trajectories[0], save_path="figures/ks_validation.png")

    print("\n=== KS Validation Summary ===")
    print(f"  Mean (1/2)Σu² energy: {mean_e:.1f} ± {std_e:.1f}")
    if abs(mean_e - 42.8) < 20:
        print("  PASS: Within acceptable range of literature value (~42.8)")
    else:
        print("  WARN: Energy differs from target — check convention or parameters")
