"""
Pendulum environments for Phase 2 sanity checks.

SimplePendulum:   H(q, p) = p²/2 + (1 - cos(q))   [m=l=g=1, integrable]
DoublePendulum:   H(q, p) = T(q,p) + V(q)          [m=l=1, g=1, chaotic]

Data is generated with high-accuracy integrators (leapfrog for simple,
DOP853 for double) at float64 precision. The learned world models train
and evaluate in float32.

Phase 2 targets:
  - Simple pendulum: energy drift < 1e-4 over 1000 steps (SympRSSM)
  - Double pendulum: ≥10x lower energy drift for SympRSSM vs RK4-RSSM (Gate 1)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import diffrax

from integrators.leapfrog import leapfrog_integrate


# ---------------------------------------------------------------------------
# Simple Pendulum
# ---------------------------------------------------------------------------

def simple_pendulum_H(q: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """H(q, p) = p²/2 + (1 - cos(q)), non-dim (m=l=g=1)."""
    return 0.5 * p[0] ** 2 + (1.0 - jnp.cos(q[0]))


def generate_simple_pendulum_data(
    n_traj: int,
    traj_len: int,
    dt: float,
    key: jax.random.PRNGKey,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate simple pendulum trajectories via leapfrog (exact Hamiltonian).

    Args:
        n_traj: Number of trajectories.
        traj_len: Steps per trajectory (not counting IC).
        dt: Time step.
        key: JAX PRNG key.

    Returns:
        q_data: (n_traj, traj_len+1, 1) angles
        p_data: (n_traj, traj_len+1, 1) momenta
    """
    grad_V = lambda q: jnp.array([jnp.sin(q[0])])   # dV/dq = sin(q)
    grad_T = lambda p: p                              # dT/dp = p

    keys = jax.random.split(key, n_traj)
    qs, ps = [], []

    for k in keys:
        k1, k2 = jax.random.split(k)
        q0 = jax.random.uniform(k1, shape=(1,), minval=-2.5, maxval=2.5)
        p0 = jax.random.uniform(k2, shape=(1,), minval=-2.0, maxval=2.0)
        q_traj, p_traj = leapfrog_integrate(q0, p0, grad_V, grad_T, dt, traj_len)
        # Prepend IC so shape is (traj_len+1, 1)
        qs.append(jnp.concatenate([q0[None], q_traj], axis=0))
        ps.append(jnp.concatenate([p0[None], p_traj], axis=0))

    return jnp.stack(qs), jnp.stack(ps)


# ---------------------------------------------------------------------------
# Double Pendulum (chaotic, non-separable kinetic energy)
# ---------------------------------------------------------------------------

def double_pendulum_H(q: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """H for equal-mass equal-length double pendulum (m=l=1, g=1).

    Non-separable: T(q, p) depends on both q and p.
    The world model learns a separable *approximation*; the test is whether
    Yoshida4 integration of H_theta preserves H_theta vs RK4 drift.
    """
    q1, q2 = q[0], q[1]
    p1, p2 = p[0], p[1]
    delta = q1 - q2
    denom = 2.0 - jnp.cos(delta) ** 2
    T = (p1 ** 2 + 2 * p2 ** 2 - 2 * p1 * p2 * jnp.cos(delta)) / (2.0 * denom)
    V = -2.0 * jnp.cos(q1) - jnp.cos(q2)
    return T + V


def generate_double_pendulum_data(
    n_traj: int,
    traj_len: int,
    dt: float,
    key: jax.random.PRNGKey,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate double pendulum trajectories via DOP853 (diffrax.Dopri8).

    Args:
        n_traj: Number of trajectories.
        traj_len: Steps per trajectory (not counting IC).
        dt: Time step (env dt, not internal integrator step).
        key: JAX PRNG key.

    Returns:
        q_data: (n_traj, traj_len+1, 2)
        p_data: (n_traj, traj_len+1, 2)
    """
    t1 = float(traj_len * dt)
    ts = jnp.linspace(0.0, t1, traj_len + 1)

    def deriv(t, state, args):
        q, p = state[:2], state[2:]
        dq = jax.grad(double_pendulum_H, argnums=1)(q, p)
        dp = -jax.grad(double_pendulum_H, argnums=0)(q, p)
        return jnp.concatenate([dq, dp])

    term = diffrax.ODETerm(deriv)
    solver = diffrax.Dopri8()
    controller = diffrax.PIDController(rtol=1e-8, atol=1e-10)
    saveat = diffrax.SaveAt(ts=ts)

    keys = jax.random.split(key, n_traj)
    qs, ps = [], []

    for k in keys:
        k1, k2 = jax.random.split(k)
        # Small-amplitude ICs to avoid near-singular configurations
        q0 = jax.random.uniform(k1, shape=(2,), minval=-0.8, maxval=0.8)
        p0 = jax.random.uniform(k2, shape=(2,), minval=-0.8, maxval=0.8)
        state0 = jnp.concatenate([q0, p0])

        sol = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=t1,
            dt0=dt / 10.0,
            y0=state0, args=None,
            stepsize_controller=controller,
            saveat=saveat,
            max_steps=200_000,
        )
        traj = sol.ys  # (traj_len+1, 4)
        qs.append(traj[:, :2])
        ps.append(traj[:, 2:])

    return jnp.stack(qs), jnp.stack(ps)
