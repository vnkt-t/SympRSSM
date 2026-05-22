"""
RK4-RSSM ablation variant.

Same learned Hamiltonian ODE as SympRSSM, but integrated with RK4
instead of Yoshida-4. Tests whether symplecticity specifically matters
vs just having ODE structure.

This is the "Dreaming Falcon" pattern.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from models.hamiltonian_net import HamiltonianNet


def rk4_step(
    q: jnp.ndarray,
    p: jnp.ndarray,
    hamiltonian: HamiltonianNet,
    action: jnp.ndarray,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single RK4 step for Hamilton's equations.

    dq/dt =  dH/dp
    dp/dt = -dH/dq
    """

    def deriv(q_, p_):
        dq = hamiltonian.grad_p(q_, p_, action)
        dp = -hamiltonian.grad_q(q_, p_, action)
        return dq, dp

    dq1, dp1 = deriv(q, p)
    dq2, dp2 = deriv(q + 0.5 * dt * dq1, p + 0.5 * dt * dp1)
    dq3, dp3 = deriv(q + 0.5 * dt * dq2, p + 0.5 * dt * dp2)
    dq4, dp4 = deriv(q + dt * dq3, p + dt * dp3)

    q_new = q + (dt / 6.0) * (dq1 + 2 * dq2 + 2 * dq3 + dq4)
    p_new = p + (dt / 6.0) * (dp1 + 2 * dp2 + 2 * dp3 + dp4)

    return q_new, p_new
