"""
Free-form ODE ablation variant.

No Hamiltonian decomposition — just a learned ODE f_theta(s, a) integrated
by Yoshida-4. Tests whether the Hamiltonian structure matters beyond
the integrator.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx


class FreeformODE(eqx.Module):
    """Learned ODE dynamics without Hamiltonian structure."""
    net: eqx.nn.MLP

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        *,
        key: jax.random.PRNGKey,
    ):
        self.net = eqx.nn.MLP(
            in_size=state_dim + action_dim,
            out_size=state_dim,
            width_size=hidden,
            depth=3,
            key=key,
        )

    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Compute ds/dt = f_theta(s, a)."""
        x = jnp.concatenate([state, action], axis=-1)
        return self.net(x)


def freeform_rk4_step(
    state: jnp.ndarray,
    action: jnp.ndarray,
    ode: FreeformODE,
    dt: float,
) -> jnp.ndarray:
    """RK4 integration of free-form ODE."""
    k1 = ode(state, action)
    k2 = ode(state + 0.5 * dt * k1, action)
    k3 = ode(state + 0.5 * dt * k2, action)
    k4 = ode(state + dt * k3, action)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
