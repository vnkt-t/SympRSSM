"""
Tests for the learned Hamiltonian network.

- Positive-definite mass matrix (T > 0 for p != 0)
- Gradients are finite and correct shape
- HamiltonianNet forward pass produces scalar
- grad_q and grad_p produce correct-shape vectors
"""

import jax
import jax.numpy as jnp
import pytest

from models.hamiltonian_net import HamiltonianNet, KineticEnergy


class TestKineticEnergy:
    def test_positive_definite(self):
        key = jax.random.PRNGKey(0)
        T = KineticEnergy(dim=8, key=key)
        p = jax.random.normal(key, (8,))
        energy = T(p)
        assert energy > 0, "Kinetic energy should be positive for non-zero p"

    def test_zero_at_zero(self):
        key = jax.random.PRNGKey(0)
        T = KineticEnergy(dim=8, key=key)
        assert T(jnp.zeros(8)) == 0.0


class TestHamiltonianNet:
    def setup_method(self):
        self.key = jax.random.PRNGKey(42)
        self.H = HamiltonianNet(q_dim=8, p_dim=8, a_dim=4, hidden=32, key=self.key)
        self.q = jax.random.normal(self.key, (8,))
        self.p = jax.random.normal(self.key, (8,))
        self.a = jax.random.normal(self.key, (4,))

    def test_forward_scalar(self):
        val = self.H(self.q, self.p, self.a)
        assert val.shape == (), "Hamiltonian should return scalar"

    def test_grad_q_shape(self):
        g = self.H.grad_q(self.q, self.p, self.a)
        assert g.shape == (8,)

    def test_grad_p_shape(self):
        g = self.H.grad_p(self.q, self.p, self.a)
        assert g.shape == (8,)

    def test_gradients_finite(self):
        gq = self.H.grad_q(self.q, self.p, self.a)
        gp = self.H.grad_p(self.q, self.p, self.a)
        assert jnp.all(jnp.isfinite(gq))
        assert jnp.all(jnp.isfinite(gp))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
