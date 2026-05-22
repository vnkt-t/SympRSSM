"""
Noether projection layer.

After each symplectic integration step, project the latent (q, p) onto the
constraint manifold defined by known conservation laws:

    (q', p') = Yoshida4(H_theta, q, p, a)
    (q'', p'') = Pi_C(q', p')    # project onto {(q,p) : C(q,p) = C(q_0, p_0)}

Linear projection for linear constraints (momentum conservation).
One Newton step for quadratic constraints (Jacobi integral).

Straight-through gradient estimator for backprop.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Callable


def linear_projection(
    q: jnp.ndarray,
    p: jnp.ndarray,
    constraint_matrix: jnp.ndarray,
    target_value: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project (q, p) onto linear constraint: A @ [q, p] = b.

    Used for momentum conservation (translational symmetry).
    For KS: mean-zero constraint on spatial field.

    Args:
        q: Position vector.
        p: Momentum vector.
        constraint_matrix: A matrix defining the linear constraint.
        target_value: b vector (conserved quantity values).

    Returns:
        Projected (q, p).
    """
    state = jnp.concatenate([q, p])
    residual = constraint_matrix @ state - target_value
    # Minimum-norm correction: state - A^T (A A^T)^{-1} (A state - b)
    AAT = constraint_matrix @ constraint_matrix.T
    correction = constraint_matrix.T @ jnp.linalg.solve(AAT, residual)
    state_proj = state - correction
    dim = q.shape[0]
    return state_proj[:dim], state_proj[dim:]


def newton_projection(
    q: jnp.ndarray,
    p: jnp.ndarray,
    constraint_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    target_value: float,
    n_steps: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project (q, p) onto nonlinear constraint: C(q, p) = C_0.

    Used for Jacobi integral conservation (CR3BP).
    One Newton step per call (sufficient for near-constraint states).

    Args:
        q: Position vector.
        p: Momentum vector.
        constraint_fn: Scalar function C(q, p).
        target_value: C_0 (initial conserved value).
        n_steps: Number of Newton iterations.

    Returns:
        Projected (q, p).
    """
    def _newton_step(carry, _):
        q, p = carry
        c_val = constraint_fn(q, p)
        residual = c_val - target_value

        # Gradient of constraint w.r.t. (q, p)
        grad_c_q = jax.grad(constraint_fn, argnums=0)(q, p)
        grad_c_p = jax.grad(constraint_fn, argnums=1)(q, p)
        grad_c = jnp.concatenate([grad_c_q, grad_c_p])

        # Newton correction along gradient direction
        grad_norm_sq = jnp.sum(grad_c**2) + 1e-12
        alpha = residual / grad_norm_sq

        q = q - alpha * grad_c_q
        p = p - alpha * grad_c_p
        return (q, p), None

    (q, p), _ = jax.lax.scan(_newton_step, (q, p), None, length=n_steps)
    return q, p


def straight_through_project(
    q: jnp.ndarray,
    p: jnp.ndarray,
    project_fn: Callable,
    *args,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply projection with straight-through gradient estimator.

    Forward pass: projected values.
    Backward pass: gradients pass through as if no projection happened.
    """
    q_proj, p_proj = project_fn(q, p, *args)

    # Straight-through: stop gradient on the correction
    q_out = q + jax.lax.stop_gradient(q_proj - q)
    p_out = p + jax.lax.stop_gradient(p_proj - p)
    return q_out, p_out
