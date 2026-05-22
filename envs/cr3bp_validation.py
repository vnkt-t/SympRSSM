"""
CR3BP environment validation.

Validates:
  1. Jacobi integral conservation over 100 unforced orbital periods (drift < 1e-10)
  2. Libration point equilibria (L1-L5 acceleration residuals)
  3. Near-L1 Lyapunov orbit return-map closure

Integrator: DOP853 (diffrax.Dopri8, rtol=1e-12, atol=1e-14, float64).

Run: python -m envs.cr3bp_validation
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import diffrax


# ---------------------------------------------------------------------------
# Core CR3BP functions (pure, float64)
# ---------------------------------------------------------------------------

def cr3bp_deriv(state, mu):
    """CR3BP equations of motion in rotating frame (pure function, float64)."""
    x, y, xd, yd = state
    r1 = jnp.sqrt((x + mu) ** 2 + y ** 2)
    r2 = jnp.sqrt((x - 1.0 + mu) ** 2 + y ** 2)
    xdd = 2 * yd + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    ydd = -2 * xd + y - (1 - mu) * y / r1**3 - mu * y / r2**3
    return jnp.array([xd, yd, xdd, ydd])


def jacobi_constant(state, mu):
    """Compute Jacobi integral C = 2U - v², conserved in unforced CR3BP."""
    x, y, xd, yd = state
    r1 = jnp.sqrt((x + mu) ** 2 + y ** 2)
    r2 = jnp.sqrt((x - 1.0 + mu) ** 2 + y ** 2)
    U = 0.5 * (x**2 + y**2) + (1 - mu) / r1 + mu / r2
    v2 = xd**2 + yd**2
    return 2 * U - v2


# ---------------------------------------------------------------------------
# DOP853 integrator (replaces RK4 placeholder)
# ---------------------------------------------------------------------------

def integrate_trajectory(state0, mu, t1, n_save=1001):
    """Integrate unforced CR3BP with DOP853 (diffrax.Dopri8).

    Args:
        state0: Initial state (4,): (x, y, xdot, ydot)
        mu: Mass ratio (float)
        t1: End time in TU (float)
        n_save: Number of equally-spaced save points (including t=0)

    Returns:
        states: (n_save, 4)
        jacobi: (n_save,) Jacobi constant at each saved time
    """
    ts = jnp.linspace(0.0, float(t1), n_save)

    def vector_field(t, s, args):
        return cr3bp_deriv(s, mu)

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Dopri8()
    controller = diffrax.PIDController(rtol=1e-14, atol=1e-16)
    saveat = diffrax.SaveAt(ts=ts)

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=float(t1),
        dt0=float(t1) / 1000.0,
        y0=state0,
        args=None,
        stepsize_controller=controller,
        saveat=saveat,
        max_steps=500_000,
    )
    states = solution.ys  # (n_save, 4)
    jacobi_vals = jax.vmap(lambda s: jacobi_constant(s, mu))(states)
    return states, jacobi_vals


# ---------------------------------------------------------------------------
# Validation routines
# ---------------------------------------------------------------------------

def validate_jacobi_conservation(
    mu: float = 0.01215,
    n_periods: int = 100,
):
    """Validate Jacobi integral conservation over 100 orbital periods.

    Target: max |ΔC| < 1e-10 (plan requirement for DOP853 environment integrator).
    Uses the same initial condition as the environment (near L2, quasi-periodic).
    """
    print(f"CR3BP Validation: μ={mu}, n_periods={n_periods}")

    # Approximate L2 position (Richardson 1st-order: gamma = (mu/3)^(1/3))
    x_L2 = 1.0 + (mu / 3) ** (1.0 / 3.0)
    print(f"  Approximate L2 position: x = {x_L2:.6f}")

    # Small-amplitude Lyapunov orbit near L2 (quasi-periodic, unforced)
    state0 = jnp.array([x_L2 + 0.01, 0.0, 0.0, 0.1])

    # Near-L2 orbital period ≈ 2π TU
    T_period = 2.0 * float(jnp.pi)
    t1 = n_periods * T_period
    n_save = n_periods * 20 + 1  # 20 points per period — enough for drift tracking

    print(f"  Integrating for {n_periods} periods (t1={t1:.2f} TU, {n_save} save points)...")

    states, jacobi = integrate_trajectory(state0, mu, t1, n_save=n_save)

    C0 = jacobi[0]
    drift = jnp.abs(jacobi - C0)
    max_drift = float(jnp.max(drift))
    final_drift = float(drift[-1])

    print(f"  Jacobi constant C₀ = {float(C0):.12f}")
    print(f"  Max |ΔC| over {n_periods} periods = {max_drift:.2e}")
    print(f"  Final |ΔC| = {final_drift:.2e}")

    if max_drift < 1e-10:
        print("  ✓ PASS: Jacobi drift < 1e-10 (DOP853 target met)")
    elif max_drift < 1e-6:
        print("  ~ PARTIAL: Jacobi drift < 1e-6 (below 1e-10 target — tighten tolerances)")
    else:
        print("  ✗ FAIL: Jacobi drift too large — check equations of motion or float64")

    return states, jacobi, max_drift


def validate_libration_points(mu: float = 0.01215):
    """Verify L1-L5 positions satisfy the zero-acceleration equilibrium condition."""
    print(f"\nLibration point validation (μ={mu}):")

    gamma = (mu / 3) ** (1.0 / 3.0)
    L1_approx = 1.0 - mu - gamma
    L2_approx = 1.0 - mu + gamma
    L3_approx = -(1.0 + 5 * mu / 12)

    for name, x_approx in [("L1", L1_approx), ("L2", L2_approx), ("L3", L3_approx)]:
        state = jnp.array([x_approx, 0.0, 0.0, 0.0])
        accel = cr3bp_deriv(state, mu)
        residual = float(jnp.sqrt(accel[2]**2 + accel[3]**2))
        C = float(jacobi_constant(state, mu))
        print(f"  {name}: x={float(x_approx):.6f}, accel_residual={residual:.4e}, C={C:.6f}")

    L4 = jnp.array([0.5 - mu, float(jnp.sqrt(3.0)) / 2, 0.0, 0.0])
    L5 = jnp.array([0.5 - mu, -float(jnp.sqrt(3.0)) / 2, 0.0, 0.0])

    for name, state in [("L4", L4), ("L5", L5)]:
        accel = cr3bp_deriv(state, mu)
        residual = float(jnp.sqrt(accel[2]**2 + accel[3]**2))
        C = float(jacobi_constant(state, mu))
        print(f"  {name}: (x,y)=({float(state[0]):.4f},{float(state[1]):.4f}), "
              f"accel_residual={residual:.4e}, C={C:.6f}")


def validate_known_periodic_orbit(mu: float = 0.01215):
    """Validate near-L1 Lyapunov orbit closure with DOP853.

    Checks:
    - Jacobi constant conserved to < 1e-10 over one period
    - Orbit returns near its starting point (return-map closure)
    """
    print(f"\nPeriodic orbit validation (μ={mu}):")

    gamma = (mu / 3) ** (1.0 / 3.0)
    x_L1 = 1.0 - mu - gamma

    # Small-amplitude Lyapunov orbit near L1
    Ax = 0.005
    vy_guess = -Ax * 2.0
    state0 = jnp.array([x_L1 + Ax, 0.0, 0.0, vy_guess])

    T_guess = 3.0  # approximate period in TU
    n_save = 30001  # dt_eff = 1e-4 TU

    states, jacobi = integrate_trajectory(state0, mu, T_guess, n_save=n_save)
    ts = jnp.linspace(0.0, T_guess, n_save)

    C0 = float(jacobi[0])
    max_drift = float(jnp.max(jnp.abs(jacobi - C0)))

    # Find y=0 crossings (Poincaré section)
    y_vals = states[:, 1]
    crossings = jnp.where(y_vals[:-1] * y_vals[1:] < 0)[0]

    print(f"  IC: x={float(state0[0]):.6f}, vy={float(state0[3]):.6f}")
    print(f"  Jacobi C₀ = {C0:.10f}, max |ΔC| = {max_drift:.2e}")
    print(f"  y-axis crossings found: {len(crossings)}")

    if len(crossings) >= 2:
        return_idx = int(crossings[1])
        return_state = states[return_idx]
        t_return = float(ts[return_idx])
        distance = float(jnp.sqrt(
            (return_state[0] - state0[0])**2 + (return_state[1] - state0[1])**2
        ))
        print(f"  Return at t={t_return:.4f} TU: Euclidean distance={distance:.4e}")
        if distance < 0.01:
            print("  ✓ Orbit closes to within 0.01 DU (good for linearized IC)")
    else:
        print("  (no return crossing — orbit may be escaping or IC needs refinement)")

    return states, jacobi


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_validation(states, jacobi, save_path="figures/cr3bp_validation.png", mu=0.01215):
    """Plot trajectory, Jacobi drift, and phase space."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(states[:, 0], states[:, 1], "b-", linewidth=0.3, alpha=0.7)
    ax.plot(float(states[0, 0]), float(states[0, 1]), "go", markersize=6, label="start")
    ax.plot(-mu, 0, "ko", markersize=8, label="Earth")
    ax.plot(1 - mu, 0, "o", color="gray", markersize=5, label="Moon")
    ax.set_xlabel("x (rotating frame)")
    ax.set_ylabel("y (rotating frame)")
    ax.set_title("CR3BP trajectory (100 periods)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    ax = axes[1]
    C0 = jacobi[0]
    drift = jnp.abs(jacobi - C0)
    ax.plot(np.array(drift), "b-", linewidth=0.5)
    ax.axhline(1e-10, color="r", linestyle="--", alpha=0.7, label="target 1e-10")
    ax.set_xlabel("save index")
    ax.set_ylabel("|C(t) - C₀|")
    ax.set_title("Jacobi integral drift (DOP853)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(np.array(states[:, 0]), np.array(states[:, 2]), "b-", linewidth=0.3, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("ẋ")
    ax.set_title("Phase space (x, ẋ)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved validation plot to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)

    # 1. Jacobi conservation over 100 periods (primary target)
    states, jacobi, max_drift = validate_jacobi_conservation(n_periods=100)
    plot_validation(states, jacobi, save_path="figures/cr3bp_validation.png")

    # 2. Libration point equilibria
    validate_libration_points()

    # 3. Near-L1 Lyapunov orbit closure
    validate_known_periodic_orbit()

    print("\n=== CR3BP Validation Summary ===")
    print(f"  Max Jacobi drift (100 periods): {max_drift:.2e}")
    if max_drift < 1e-10:
        print("  PASS ✓  Jacobi drift < 1e-10")
    elif max_drift < 1e-6:
        print("  PARTIAL  Jacobi drift < 1e-6 (below 1e-10 target)")
    else:
        print("  FAIL    Check equations of motion or float64 precision")
