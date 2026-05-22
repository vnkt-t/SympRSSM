"""
Learned separable Hamiltonian network.

H_theta(q, p, a) = T_theta(p) + V_theta(q) + Phi_theta(q, a)

- T_theta(p): Kinetic energy with learned Cholesky mass matrix (positive-definite).
- V_theta(q): Potential energy (spectral-normed MLP).
- Phi_theta(q, a): Action-coupling potential (spectral-normed MLP).

Built with Equinox for JAX compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional


class SpectralNormedLinear(eqx.Module):
    """Linear layer with spectral normalization for Lipschitz control."""
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_features: int, out_features: int, *, key: jax.random.PRNGKey):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_features, in_features)) * 0.01
        self.bias = jnp.zeros(out_features)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Spectral normalization: W / sigma(W)
        sigma = jnp.linalg.svd(self.weight, compute_uv=False)[0]
        W_normalized = self.weight / jnp.maximum(sigma, 1e-12)
        return W_normalized @ x + self.bias


class KineticEnergy(eqx.Module):
    """T_theta(p) = 0.5 * p^T M_theta p, with M = L L^T (Cholesky, positive-definite)."""
    L_flat: jnp.ndarray  # Lower-triangular Cholesky factor (flattened)
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int, *, key: jax.random.PRNGKey):
        self.dim = dim
        # Initialize near identity: L = I + small noise
        n_params = dim * (dim + 1) // 2
        self.L_flat = jnp.zeros(n_params)  # will reconstruct as I + params

    def _get_mass_matrix(self) -> jnp.ndarray:
        """Reconstruct positive-definite mass matrix M = L L^T."""
        L = jnp.zeros((self.dim, self.dim))
        idx = jnp.tril_indices(self.dim)
        L = L.at[idx].set(self.L_flat)
        # Ensure positive diagonal
        diag_idx = jnp.diag_indices(self.dim)
        L = L.at[diag_idx].set(jax.nn.softplus(L[diag_idx]) + 0.1)
        return L @ L.T

    def __call__(self, p: jnp.ndarray) -> jnp.ndarray:
        """Compute T(p) = 0.5 * p^T M p."""
        M = self._get_mass_matrix()
        return 0.5 * p @ M @ p


class PotentialEnergy(eqx.Module):
    """V_theta(q): Learned potential energy via spectral-normed MLP."""
    layers: list

    def __init__(self, dim: int, hidden: int = 64, n_layers: int = 2, *, key: jax.random.PRNGKey):
        keys = jax.random.split(key, n_layers + 1)
        self.layers = []
        in_d = dim
        for i in range(n_layers):
            self.layers.append(SpectralNormedLinear(in_d, hidden, key=keys[i]))
            in_d = hidden
        self.layers.append(SpectralNormedLinear(in_d, 1, key=keys[-1]))

    def __call__(self, q: jnp.ndarray) -> jnp.ndarray:
        x = q
        for layer in self.layers[:-1]:
            x = jax.nn.softplus(layer(x))
        return self.layers[-1](x).squeeze(-1)


class ActionCouplingPotential(eqx.Module):
    """Phi_theta(q, a): How external actions inject energy into the system."""
    layers: list

    def __init__(
        self, q_dim: int, a_dim: int, hidden: int = 64, n_layers: int = 2, *, key: jax.random.PRNGKey
    ):
        keys = jax.random.split(key, n_layers + 1)
        in_d = q_dim + a_dim
        self.layers = []
        for i in range(n_layers):
            self.layers.append(SpectralNormedLinear(in_d, hidden, key=keys[i]))
            in_d = hidden
        self.layers.append(SpectralNormedLinear(in_d, 1, key=keys[-1]))

    def __call__(self, q: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([q, a], axis=-1)
        for layer in self.layers[:-1]:
            x = jax.nn.softplus(layer(x))
        return self.layers[-1](x).squeeze(-1)


class HamiltonianNet(eqx.Module):
    """Full separable Hamiltonian: H(q, p, a) = T(p) + V(q) + Phi(q, a)."""
    kinetic: KineticEnergy
    potential: PotentialEnergy
    coupling: ActionCouplingPotential
    q_dim: int = eqx.field(static=True)
    p_dim: int = eqx.field(static=True)
    a_dim: int = eqx.field(static=True)

    def __init__(
        self,
        q_dim: int,
        p_dim: int,
        a_dim: int,
        hidden: int = 64,
        *,
        key: jax.random.PRNGKey,
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.q_dim = q_dim
        self.p_dim = p_dim
        self.a_dim = a_dim
        self.kinetic = KineticEnergy(p_dim, key=k1)
        self.potential = PotentialEnergy(q_dim, hidden=hidden, key=k2)
        self.coupling = ActionCouplingPotential(q_dim, a_dim, hidden=hidden, key=k3)

    def __call__(self, q: jnp.ndarray, p: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        return self.kinetic(p) + self.potential(q) + self.coupling(q, a)

    def grad_q(self, q: jnp.ndarray, p: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """dH/dq = dV/dq + dPhi/dq — the 'force' for momentum updates."""
        def h_of_q(q_):
            return self.potential(q_) + self.coupling(q_, a)
        return jax.grad(h_of_q)(q)

    def grad_p(self, q: jnp.ndarray, p: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """dH/dp = dT/dp — the 'velocity' for position updates."""
        return jax.grad(self.kinetic)(p)
