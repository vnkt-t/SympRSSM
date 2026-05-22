"""
Tests for Noether projection layer.

- Linear projection: residual < 1e-12 for linear constraints
- Newton projection: residual < 1e-6 for quadratic constraints (1 step)
- Straight-through gradients are finite
"""

import jax
import jax.numpy as jnp
import pytest

from models.noether import linear_projection, newton_projection, straight_through_project


class TestLinearProjection:
    def test_mean_zero_constraint(self):
        """KS momentum conservation: mean(q) = 0."""
        dim = 16
        q = jnp.ones(dim) * 0.5  # not mean-zero
        p = jax.random.normal(jax.random.PRNGKey(0), (dim,))

        # Constraint: sum(q) / dim = 0, i.e., [1/dim, ..., 1/dim, 0, ..., 0] @ [q,p] = 0
        A = jnp.zeros((1, 2 * dim))
        A = A.at[0, :dim].set(1.0 / dim)
        target = jnp.array([0.0])

        q_proj, p_proj = linear_projection(q, p, A, target)
        assert abs(jnp.mean(q_proj)) < 1e-12

    def test_preserves_momentum(self):
        """p should be unchanged by a q-only constraint."""
        dim = 8
        q = jnp.ones(dim)
        p = jax.random.normal(jax.random.PRNGKey(1), (dim,))

        A = jnp.zeros((1, 2 * dim))
        A = A.at[0, :dim].set(1.0 / dim)
        target = jnp.array([0.0])

        _, p_proj = linear_projection(q, p, A, target)
        assert jnp.allclose(p, p_proj, atol=1e-12)


class TestNewtonProjection:
    def test_jacobi_constraint(self):
        """Simple quadratic constraint: C(q,p) = |q|² + |p|² = C_0."""
        q = jnp.array([1.1, 0.0])
        p = jnp.array([0.0, 0.9])
        target = 2.0  # want |q|² + |p|² = 2

        def constraint(q, p):
            return jnp.sum(q**2) + jnp.sum(p**2)

        q_proj, p_proj = newton_projection(q, p, constraint, target, n_steps=3)
        residual = abs(constraint(q_proj, p_proj) - target)
        assert residual < 1e-6, f"Newton projection residual: {residual}"


class TestStraightThrough:
    def test_gradients_flow(self):
        """Gradients should pass through projection unchanged."""
        def loss(q):
            p = jnp.zeros_like(q)
            A = jnp.zeros((1, 2 * q.shape[0]))
            A = A.at[0, :q.shape[0]].set(1.0 / q.shape[0])
            target = jnp.array([0.0])
            q_proj, _ = straight_through_project(
                q, p, linear_projection, A, target
            )
            return jnp.sum(q_proj**2)

        q = jnp.array([1.0, 2.0, 3.0])
        grad = jax.grad(loss)(q)
        assert jnp.all(jnp.isfinite(grad))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
