"""
4th-order Yoshida symplectic integrator in JAX.

Composition of leapfrog steps with Yoshida (1990) coefficients.
3 force evaluations per step, preserves symplectic 2-form exactly.
Differentiable for end-to-end training.
"""

import jax
import jax.numpy as jnp
from typing import Callable


# Yoshida (1990) 4th-order coefficients
_CBRT2 = 2.0 ** (1.0 / 3.0)
_W1 = 1.0 / (2.0 - _CBRT2)
_W0 = -_CBRT2 / (2.0 - _CBRT2)

# Position coefficients c_i
C1 = _W1 / 2.0
C2 = (_W0 + _W1) / 2.0
C3 = C2
C4 = C1

# Momentum coefficients d_i
D1 = _W1
D2 = _W0
D3 = _W1


def yoshida4_step(
    q: jnp.ndarray,
    p: jnp.ndarray,
    grad_V: Callable[[jnp.ndarray], jnp.ndarray],
    grad_T: Callable[[jnp.ndarray], jnp.ndarray],
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single Yoshida-4 step for separable Hamiltonian H = T(p) + V(q).

    This is a 3-stage composition of leapfrog sub-steps with coefficients
    chosen to cancel the O(h²) and O(h³) error terms, yielding O(h⁴) accuracy
    while preserving the symplectic structure exactly.

    Args:
        q: Position (generalized coordinates).
        p: Momentum (conjugate momenta).
        grad_V: Gradient of potential energy w.r.t. q (i.e. dV/dq).
                 For SympRSSM this includes the action-coupling potential.
        grad_T: Gradient of kinetic energy w.r.t. p (i.e. dT/dp).
        dt: Time step size.

    Returns:
        (q_new, p_new) after one Yoshida-4 step.
    """
    # Sub-step 1
    q = q + C1 * dt * grad_T(p)
    p = p - D1 * dt * grad_V(q)

    # Sub-step 2
    q = q + C2 * dt * grad_T(p)
    p = p - D2 * dt * grad_V(q)

    # Sub-step 3
    q = q + C3 * dt * grad_T(p)
    p = p - D3 * dt * grad_V(q)

    # Final position update
    q = q + C4 * dt * grad_T(p)

    return q, p


def yoshida4_integrate(
    q0: jnp.ndarray,
    p0: jnp.ndarray,
    grad_V: Callable[[jnp.ndarray], jnp.ndarray],
    grad_T: Callable[[jnp.ndarray], jnp.ndarray],
    dt: float,
    n_steps: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Multi-step Yoshida-4 integration via jax.lax.scan."""

    def scan_fn(carry, _):
        q, p = carry
        q, p = yoshida4_step(q, p, grad_V, grad_T, dt)
        return (q, p), (q, p)

    (q_final, p_final), (q_traj, p_traj) = jax.lax.scan(
        scan_fn, (q0, p0), None, length=n_steps
    )
    return q_traj, p_traj
