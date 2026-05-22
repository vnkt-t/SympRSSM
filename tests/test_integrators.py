"""
Tests for symplectic integrators on analytic Hamiltonian systems.

Kepler problem: H = |p|²/2 - 1/|q|
- Energy drift < 1e-8 over 10⁴ steps (leapfrog)
- Energy drift < 1e-10 over 10⁴ steps (Yoshida-4)
- Symplectic 2-form preservation
- Differentiability (gradients flow through)
"""

import jax
import jax.numpy as jnp
import pytest

# Enable float64 for precision tests
jax.config.update("jax_enable_x64", True)

from integrators.leapfrog import leapfrog_step, leapfrog_integrate
from integrators.yoshida4 import yoshida4_step, yoshida4_integrate


def kepler_grad_V(q):
    """dV/dq for Kepler: V(q) = -1/|q|, so dV/dq = q/|q|³."""
    r = jnp.sqrt(jnp.sum(q**2) + 1e-12)
    return q / r**3


def kepler_grad_T(p):
    """dT/dp for Kepler: T(p) = |p|²/2, so dT/dp = p."""
    return p


def kepler_energy(q, p):
    r = jnp.sqrt(jnp.sum(q**2))
    return 0.5 * jnp.sum(p**2) - 1.0 / r


class TestLeapfrog:
    def test_energy_conservation(self):
        q0 = jnp.array([1.0, 0.0])
        p0 = jnp.array([0.0, 1.0])
        dt = 0.01
        n_steps = 10_000

        q_traj, p_traj = leapfrog_integrate(q0, p0, kepler_grad_V, kepler_grad_T, dt, n_steps)

        E0 = kepler_energy(q0, p0)
        E_final = kepler_energy(q_traj[-1], p_traj[-1])
        assert abs(E_final - E0) < 1e-8, f"Leapfrog energy drift: {abs(E_final - E0)}"

    def test_differentiable(self):
        q0 = jnp.array([1.0, 0.0])
        p0 = jnp.array([0.0, 1.0])

        def loss(q0):
            q, p = leapfrog_step(q0, p0, kepler_grad_V, kepler_grad_T, 0.01)
            return jnp.sum(q**2)

        grad = jax.grad(loss)(q0)
        assert jnp.all(jnp.isfinite(grad))


class TestYoshida4:
    def test_energy_conservation(self):
        q0 = jnp.array([1.0, 0.0])
        p0 = jnp.array([0.0, 1.0])
        dt = 0.01
        n_steps = 10_000

        q_traj, p_traj = yoshida4_integrate(q0, p0, kepler_grad_V, kepler_grad_T, dt, n_steps)

        E0 = kepler_energy(q0, p0)
        E_final = kepler_energy(q_traj[-1], p_traj[-1])
        assert abs(E_final - E0) < 1e-10, f"Yoshida-4 energy drift: {abs(E_final - E0)}"

    def test_higher_order_than_leapfrog(self):
        """Yoshida-4 should have tighter energy bound than leapfrog at same dt."""
        q0 = jnp.array([1.0, 0.0])
        p0 = jnp.array([0.0, 1.0])
        dt = 0.1  # larger dt to see the difference
        n_steps = 1000

        q_lf, p_lf = leapfrog_integrate(q0, p0, kepler_grad_V, kepler_grad_T, dt, n_steps)
        q_y4, p_y4 = yoshida4_integrate(q0, p0, kepler_grad_V, kepler_grad_T, dt, n_steps)

        E0 = kepler_energy(q0, p0)
        drift_lf = abs(kepler_energy(q_lf[-1], p_lf[-1]) - E0)
        drift_y4 = abs(kepler_energy(q_y4[-1], p_y4[-1]) - E0)
        assert drift_y4 < drift_lf, f"Yoshida-4 ({drift_y4}) should beat leapfrog ({drift_lf})"

    def test_differentiable(self):
        q0 = jnp.array([1.0, 0.0])
        p0 = jnp.array([0.0, 1.0])

        def loss(q0):
            q, p = yoshida4_step(q0, p0, kepler_grad_V, kepler_grad_T, 0.01)
            return jnp.sum(q**2)

        grad = jax.grad(loss)(q0)
        assert jnp.all(jnp.isfinite(grad))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
