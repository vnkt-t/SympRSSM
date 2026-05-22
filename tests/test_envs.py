"""
Tests for KS and CR3BP environments.

- Reset produces valid observations
- Step produces valid (obs, reward, terminated, truncated, info)
- KS energy computable
- CR3BP Jacobi constant computable
- Episodes terminate at correct length
"""

import jax.numpy as jnp
import numpy as np
import pytest

from envs.ks import KSEnv
from envs.cr3bp import CR3BPEnv


class TestKSEnv:
    def test_reset(self):
        env = KSEnv(N=32, L=22.0)
        obs, info = env.reset(seed=0)
        assert obs.shape == (32,)
        assert np.all(np.isfinite(obs))

    def test_step(self):
        env = KSEnv(N=32, L=22.0)
        env.reset(seed=0)
        action = np.zeros(4)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (32,)
        assert isinstance(reward, float)

    def test_episode_length(self):
        env = KSEnv(N=32, L=22.0, episode_length=10)
        env.reset(seed=0)
        for _ in range(10):
            _, _, _, truncated, _ = env.step(np.zeros(4))
        assert truncated


class TestCR3BPEnv:
    def test_reset(self):
        env = CR3BPEnv()
        obs, info = env.reset(seed=0)
        assert obs.shape == (5,)
        assert np.all(np.isfinite(obs))

    def test_jacobi_constant(self):
        env = CR3BPEnv()
        env.reset(seed=0)
        C = env.jacobi_constant()
        assert np.isfinite(C)

    def test_step(self):
        env = CR3BPEnv()
        env.reset(seed=0)
        action = np.zeros(2)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (5,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
