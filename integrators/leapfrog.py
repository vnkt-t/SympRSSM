"""
2nd-order Leapfrog (Stormer-Verlet) symplectic integrator in JAX.

Differentiable. For separable Hamiltonians H = T(p) + V(q).
"""

import jax
import jax.numpy as jnp
from typing import Callable


def leapfrog_step(
    q: jnp.ndarray,
    p: jnp.ndarray,
    grad_V: Callable[[jnp.ndarray], jnp.ndarray],
    grad_T: Callable[[jnp.ndarray], jnp.ndarray],
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single leapfrog step for separable Hamiltonian H = T(p) + V(q).

    Args:
        q: Position (generalized coordinates).
        p: Momentum (conjugate momenta).
        grad_V: Gradient of potential energy w.r.t. q (i.e. dV/dq).
        grad_T: Gradient of kinetic energy w.r.t. p (i.e. dT/dp).
        dt: Time step size.

    Returns:
        (q_new, p_new) after one leapfrog step.
    """
    # Half-step momentum
    p_half = p - 0.5 * dt * grad_V(q)

    # Full-step position
    q_new = q + dt * grad_T(p_half)

    # Half-step momentum
    p_new = p_half - 0.5 * dt * grad_V(q_new)

    return q_new, p_new


def leapfrog_integrate(
    q0: jnp.ndarray,
    p0: jnp.ndarray,
    grad_V: Callable[[jnp.ndarray], jnp.ndarray],
    grad_T: Callable[[jnp.ndarray], jnp.ndarray],
    dt: float,
    n_steps: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Multi-step leapfrog integration via jax.lax.scan."""

    def scan_fn(carry, _):
        q, p = carry
        q, p = leapfrog_step(q, p, grad_V, grad_T, dt)
        return (q, p), (q, p)

    (q_final, p_final), (q_traj, p_traj) = jax.lax.scan(
        scan_fn, (q0, p0), None, length=n_steps
    )
    return q_traj, p_traj
