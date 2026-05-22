"""
1D Kuramoto-Sivashinsky environment for RL control.

PDE: du/dt + u du/dx + d²u/dx² + ν d⁴u/dx⁴ = f(x, t)

Pseudo-spectral solver with ETDRK4 (Kassam & Trefethen 2005) time-stepping.
Periodic BCs, N=64 grid points, domain L=22 (chaotic regime).
Gymnasium-compatible. float64 for numerical accuracy.

Reference setup from Botteghi & Fasel (2023/2025).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from functools import partial


def _precompute_etdrk4_coefficients(L: jnp.ndarray, dt: float) -> dict:
    """Precompute ETDRK4 coefficients using contour integral method.

    Kassam & Trefethen (2005): evaluate coefficient functions via the mean
    of M points on a circle in the complex plane to avoid 0/0 near L=0.

    Args:
        L: Linear operator in Fourier space (1D array, one per wavenumber).
        dt: Time step.

    Returns:
        Dictionary of precomputed arrays: E, E2, f1, f2, f3, Q.
    """
    M = 64  # contour points (KT uses 16-32; 64 for safety)
    # Contour: M points on the unit circle (KT convention)
    theta = jnp.pi * (jnp.arange(1, M + 1) - 0.5) / M
    r = jnp.exp(1j * theta)  # (M,)

    hL = dt * L  # (N_modes,)
    LR = hL[:, None] + r[None, :]  # (N_modes, M) — contour-shifted

    E = jnp.exp(hL)
    E2 = jnp.exp(hL / 2)

    # Substep coefficient: L^{-1}(E2 - I) via contour
    Q = dt * jnp.real(jnp.mean((jnp.exp(LR / 2) - 1.0) / LR, axis=1))

    # Final-step coefficients (Kassam & Trefethen eq. 26)
    f1 = dt * jnp.real(jnp.mean(
        (-4 - LR + jnp.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1,
    ))
    f2 = dt * jnp.real(jnp.mean(
        (2 + LR + jnp.exp(LR) * (-2 + LR)) / LR**3, axis=1,
    ))
    f3 = dt * jnp.real(jnp.mean(
        (-4 - 3 * LR - LR**2 + jnp.exp(LR) * (4 - LR)) / LR**3, axis=1,
    ))

    return {"E": E, "E2": E2, "Q": Q, "f1": f1, "f2": f2, "f3": f3}


class KSEnv(gym.Env):
    """Kuramoto-Sivashinsky chaotic PDE control environment.

    State: u(x, t) on N grid points with periodic BCs.
    Action: boundary forcing f(x, t) at M actuator locations.
    Reward: -||u||² (stabilize to zero).
    Episode: 200 steps at dt=0.5.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        N: int = 64,
        L: float = 22.0,
        nu: float = 1.0,
        dt: float = 0.5,
        inner_steps: int = 10,
        n_actuators: int = 4,
        episode_length: int = 200,
        seed: int | None = None,
    ):
        super().__init__()
        self.N = N
        self.L = L
        self.nu = nu
        self.dt = dt
        self.inner_steps = inner_steps  # sub-steps per env step for stability
        self.inner_dt = dt / inner_steps
        self.n_actuators = n_actuators
        self.episode_length = episode_length

        # Spatial grid
        self.dx = L / N
        self.x = jnp.linspace(0, L, N, endpoint=False)

        # Wavenumbers for pseudo-spectral method
        # rfftfreq(N, d=dx) gives f_n = n/(N*dx) = n/L
        # wavenumber k_n = 2π f_n = 2πn/L
        self.k = 2 * jnp.pi * jnp.fft.rfftfreq(N, d=L / N)

        # Linear operator for KS in Fourier space:
        # du/dt = Lu + N(u) + f
        # L_k = k² - ν k⁴  (destabilizing at low k, stabilizing at high k)
        self.lin_op = self.k**2 - self.nu * self.k**4

        # Precompute ETDRK4 coefficients for the inner time step
        self._etdrk4 = _precompute_etdrk4_coefficients(self.lin_op, self.inner_dt)

        # Dealiasing mask (2/3 rule)
        self.dealias = jnp.where(jnp.abs(self.k) < (2.0 / 3.0) * jnp.max(self.k), 1.0, 0.0)

        # Actuator positions (evenly spaced)
        self.actuator_idx = jnp.linspace(0, N, n_actuators, endpoint=False).astype(int)

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(N,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_actuators,), dtype=np.float64
        )

        self._step_count = 0
        self._state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        key = jax.random.PRNGKey(seed if seed is not None else 0)
        # Random initial condition: small perturbation
        self._state = 0.1 * jax.random.normal(key, shape=(self.N,))
        self._step_count = 0
        return np.array(self._state), {}

    def step(self, action):
        action = jnp.array(action)
        forcing = self._build_forcing(action)
        self._state = self._integrate_step(self._state, forcing)
        self._step_count += 1

        reward = -float(jnp.mean(self._state**2))
        terminated = False
        truncated = self._step_count >= self.episode_length

        return np.array(self._state), reward, terminated, truncated, {}

    def _build_forcing(self, action: jnp.ndarray) -> jnp.ndarray:
        """Convert discrete actuator actions to spatial forcing field."""
        forcing = jnp.zeros(self.N)
        sigma = self.dx * 3  # Gaussian width
        for i in range(self.n_actuators):
            center = self.x[self.actuator_idx[i]]
            # Periodic Gaussian
            dist = jnp.abs(self.x - center)
            dist = jnp.minimum(dist, self.L - dist)
            gaussian = jnp.exp(-0.5 * (dist / sigma) ** 2)
            gaussian = gaussian / (jnp.sum(gaussian) * self.dx)  # normalize
            forcing = forcing + action[i] * gaussian
        return forcing

    def _nonlinear(self, u_hat: jnp.ndarray) -> jnp.ndarray:
        """Compute nonlinear term N̂ = -0.5*ik*FFT(u²) in Fourier space.

        Uses the conserved form: -u du/dx = -d(u²/2)/dx → -0.5*ik*FFT(u²).
        This is equivalent to the product form but better for dealiasing.
        Follows Kassam & Trefethen (2005).
        """
        u = jnp.fft.irfft(self.dealias * u_hat, n=self.N)
        return -0.5j * self.k * jnp.fft.rfft(u**2)

    @partial(jax.jit, static_argnums=(0,))
    def _integrate_step(self, u: jnp.ndarray, forcing: jnp.ndarray) -> jnp.ndarray:
        """ETDRK4 integration for one environment dt (multiple inner steps).

        Kassam & Trefethen (2005) exponential time-differencing RK4.
        Linear part handled exactly via matrix exponential.
        Nonlinear part advanced with 4th-order accuracy.
        """
        u_hat = jnp.fft.rfft(u)
        f_hat = jnp.fft.rfft(forcing)

        E = self._etdrk4["E"]
        E2 = self._etdrk4["E2"]
        Q = self._etdrk4["Q"]
        f1 = self._etdrk4["f1"]
        f2 = self._etdrk4["f2"]
        f3 = self._etdrk4["f3"]

        def _one_etdrk4_step(u_hat, _):
            # Nonlinear term at current state (includes forcing)
            Nu = self._nonlinear(u_hat) + f_hat

            # Stage a: half-step
            a = E2 * u_hat + Q * Nu
            Na = self._nonlinear(a) + f_hat

            # Stage b: half-step with Na
            b = E2 * u_hat + Q * Na
            Nb = self._nonlinear(b) + f_hat

            # Stage c: full step from a
            c = E2 * a + Q * (2 * Nb - Nu)
            Nc = self._nonlinear(c) + f_hat

            # Combine: full step with 4th-order accuracy
            u_hat_new = E * u_hat + Nu * f1 + 2 * (Na + Nb) * f2 + Nc * f3
            return u_hat_new, None

        u_hat_final, _ = jax.lax.scan(_one_etdrk4_step, u_hat, None, length=self.inner_steps)
        return jnp.fft.irfft(u_hat_final, n=self.N)

    def get_energy(self) -> float:
        """Compute spatial energy density: mean(u²) = (1/N) * Σu².

        For L=22 N=64 uncontrolled KS, time-averaged mean(u²) ≈ 1.4.
        Equivalently, (1/2)Σu² ≈ 43 (discrete L2 convention used in some refs).
        """
        return float(jnp.mean(self._state**2))

    def get_l2_energy(self) -> float:
        """Compute (1/2) * Σ u² — discrete L2 energy (matches ~42.8 convention)."""
        return float(0.5 * jnp.sum(self._state**2))
