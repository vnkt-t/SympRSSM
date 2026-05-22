"""
SympRSSM: Symplectic Recurrent State-Space Model.

Drop-in replacement for DreamerV3's RSSM. Replaces the GRU deterministic
transition with Yoshida-4 symplectic integration of a learned Hamiltonian.
Keeps the stochastic categorical latent z_t from DreamerV3.

Full transition:
    (q_{t+1}, p_{t+1}) = Yoshida4(H_theta, q_t, p_t, a_t)   # deterministic
    z_{t+1} ~ p_theta(z | q_{t+1}, p_{t+1})                   # stochastic
    s_{t+1} = (q_{t+1}, p_{t+1}, z_{t+1})                     # full latent
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial

from models.hamiltonian_net import HamiltonianNet
from integrators.yoshida4 import yoshida4_step


class SympRSSM(eqx.Module):
    """Symplectic RSSM replacing DreamerV3's GRU-based deterministic path."""

    hamiltonian: HamiltonianNet
    q_dim: int = eqx.field(static=True)
    p_dim: int = eqx.field(static=True)
    a_dim: int = eqx.field(static=True)
    stoch_dim: int = eqx.field(static=True)
    stoch_classes: int = eqx.field(static=True)
    integrator_dt: float = eqx.field(static=True)

    # Prior and posterior networks (categorical distribution over z)
    prior_net: eqx.nn.MLP
    posterior_net: eqx.nn.MLP

    def __init__(
        self,
        q_dim: int = 128,
        p_dim: int = 128,
        a_dim: int = 4,
        stoch_dim: int = 32,
        stoch_classes: int = 32,
        hidden: int = 256,
        integrator_dt: float = 1.0,
        *,
        key: jax.random.PRNGKey,
    ):
        k1, k2, k3 = jax.random.split(key, 3)

        self.q_dim = q_dim
        self.p_dim = p_dim
        self.a_dim = a_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.integrator_dt = integrator_dt

        self.hamiltonian = HamiltonianNet(
            q_dim=q_dim, p_dim=p_dim, a_dim=a_dim, hidden=hidden, key=k1
        )

        # Prior: p(z | q, p)
        self.prior_net = eqx.nn.MLP(
            in_size=q_dim + p_dim,
            out_size=stoch_dim * stoch_classes,
            width_size=hidden,
            depth=2,
            key=k2,
        )

        # Posterior: q(z | q, p, embed) — embed comes from encoder
        self.posterior_net = eqx.nn.MLP(
            in_size=q_dim + p_dim + hidden,  # +hidden for observation embedding
            out_size=stoch_dim * stoch_classes,
            width_size=hidden,
            depth=2,
            key=k3,
        )

    def deterministic_transition(
        self, q: jnp.ndarray, p: jnp.ndarray, action: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Symplectic deterministic transition via Yoshida-4."""
        grad_V = lambda q_: self.hamiltonian.grad_q(q_, p, action)
        grad_T = lambda p_: self.hamiltonian.grad_p(q, p_, action)
        return yoshida4_step(q, p, grad_V, grad_T, self.integrator_dt)

    def prior(self, q: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        """Prior logits for categorical stochastic latent."""
        x = jnp.concatenate([q, p], axis=-1)
        logits = self.prior_net(x)
        return logits.reshape(self.stoch_dim, self.stoch_classes)

    def posterior(
        self, q: jnp.ndarray, p: jnp.ndarray, obs_embed: jnp.ndarray
    ) -> jnp.ndarray:
        """Posterior logits for categorical stochastic latent."""
        x = jnp.concatenate([q, p, obs_embed], axis=-1)
        logits = self.posterior_net(x)
        return logits.reshape(self.stoch_dim, self.stoch_classes)

    def initial_state(self, key: jax.random.PRNGKey) -> dict:
        """Initial latent state (q, p, z) all zeros."""
        return {
            "q": jnp.zeros(self.q_dim),
            "p": jnp.zeros(self.p_dim),
            "z": jnp.zeros((self.stoch_dim, self.stoch_classes)),
        }

    def get_latent(self, state: dict) -> jnp.ndarray:
        """Flatten (q, p, z) into a single vector for decoders."""
        return jnp.concatenate([
            state["q"],
            state["p"],
            state["z"].reshape(-1),
        ])
