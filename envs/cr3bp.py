"""
Circular Restricted Three-Body Problem (CR3BP) environment.

Spacecraft station-keeping near Earth-Moon L2 halo orbit.
JAX-based, Gymnasium-compatible, float64.

State: (x, y, x_dot, y_dot, fuel) in rotating frame.
Action: continuous 2D thrust vector bounded by T_max.
Reward: -||r - r_halo||² - lambda * fuel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from functools import partial
import diffrax


class CR3BPEnv(gym.Env):
    """CR3BP station-keeping environment in the rotating frame.

    Uses non-dimensional units (TU, DU) standard for CR3BP.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        mu: float = 0.01215,  # Earth-Moon mass ratio
        dt: float = 0.01,     # Time step in TU
        T_max: float = 0.01,  # Max thrust (non-dim)
        fuel_penalty: float = 0.1,
        episode_length: int = 500,
        seed: int | None = None,
    ):
        super().__init__()
        self.mu = mu
        self.dt = dt
        self.T_max = T_max
        self.fuel_penalty = fuel_penalty
        self.episode_length = episode_length

        # L2 halo orbit reference (approximate, to be refined in validation)
        self.r_halo = jnp.array([1.1557, 0.0])  # Earth-Moon L2 (non-dim, Richardson approx)

        # Gym spaces: state = (x, y, xdot, ydot, fuel)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float64
        )

        self._step_count = 0
        self._state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        key = jax.random.PRNGKey(seed if seed is not None else 0)
        # Start near L2 with small perturbation
        perturbation = 1e-3 * jax.random.normal(key, shape=(4,))
        x0 = jnp.array([self.r_halo[0], self.r_halo[1], 0.0, 0.0]) + perturbation
        fuel0 = jnp.array([1.0])  # full fuel
        self._state = jnp.concatenate([x0, fuel0])
        self._step_count = 0
        return np.array(self._state), {}

    def step(self, action):
        action = jnp.clip(jnp.array(action), -1.0, 1.0) * self.T_max
        self._state = self._integrate_step(self._state, action)
        self._step_count += 1

        pos = self._state[:2]
        fuel = self._state[4]
        pos_error = jnp.sum((pos - self.r_halo) ** 2)
        fuel_cost = self.fuel_penalty * jnp.sum(action**2)
        reward = -float(pos_error) - float(fuel_cost)

        terminated = fuel <= 0.0
        truncated = self._step_count >= self.episode_length

        return np.array(self._state), reward, bool(terminated), bool(truncated), {}

    @partial(jax.jit, static_argnums=(0,))
    def _integrate_step(
        self, state: jnp.ndarray, thrust: jnp.ndarray
    ) -> jnp.ndarray:
        """DOP853 integration of CR3BP equations of motion (diffrax.Dopri8).

        Uses adaptive sub-stepping within each environment dt for orbital-mechanics
        accuracy. rtol=1e-10, atol=1e-12 ensures Jacobi drift < 1e-10 per step.
        """
        s = state[:4]
        fuel = state[4]
        mu = self.mu
        dt = self.dt

        def vector_field(t, y, args):
            x, y_, xd, yd = y
            thrust_ = args
            r1 = jnp.sqrt((x + mu) ** 2 + y_ ** 2)
            r2 = jnp.sqrt((x - 1.0 + mu) ** 2 + y_ ** 2)
            xdd = (
                2 * yd + x
                - (1 - mu) * (x + mu) / r1 ** 3
                - mu * (x - 1.0 + mu) / r2 ** 3
                + thrust_[0]
            )
            ydd = (
                -2 * xd + y_
                - (1 - mu) * y_ / r1 ** 3
                - mu * y_ / r2 ** 3
                + thrust_[1]
            )
            return jnp.array([xd, yd, xdd, ydd])

        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Dopri8()
        controller = diffrax.PIDController(rtol=1e-10, atol=1e-12)
        saveat = diffrax.SaveAt(t1=True)

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=dt,
            dt0=dt / 10.0,
            y0=s,
            args=thrust,
            stepsize_controller=controller,
            saveat=saveat,
            max_steps=1024,
        )
        s_new = solution.ys[0]  # (4,) — state at t=dt

        fuel_new = jnp.clip(fuel - dt * jnp.sum(jnp.abs(thrust)), 0.0, 1.0)
        return jnp.concatenate([s_new, jnp.array([fuel_new])])

    def jacobi_constant(self, state: jnp.ndarray | None = None) -> float:
        """Compute Jacobi integral C(x, y, xdot, ydot)."""
        if state is None:
            state = self._state
        x, y, xd, yd = state[:4]
        mu = self.mu

        r1 = jnp.sqrt((x + mu) ** 2 + y**2)
        r2 = jnp.sqrt((x - 1.0 + mu) ** 2 + y**2)

        U = 0.5 * (x**2 + y**2) + (1 - mu) / r1 + mu / r2
        v2 = xd**2 + yd**2
        return float(2 * U - v2)
