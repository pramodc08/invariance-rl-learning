from gymnasium import ObservationWrapper, spaces
import numpy as np
import math


class ExplicitStaticLunarLanderObsWrapper(ObservationWrapper):
    """Explicit, observation-derived, static auxiliary features."""
    def __init__(self, env, scale=5.0, noise_level=0.0):
        super().__init__(env)
        self.scale = scale
        self.noise_level = noise_level
        self.n_aux = 3

        low = np.concatenate([env.observation_space.low, -np.inf * np.ones(self.n_aux)])
        high = np.concatenate([env.observation_space.high, np.inf * np.ones(self.n_aux)])
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs):
        aux = np.array([obs[0], obs[1], obs[4]], dtype=np.float64) * self.scale
        if self.noise_level > 0:
            obs = obs + np.random.normal(0.0, self.noise_level, size=obs.shape)
        return np.concatenate([obs, aux]).astype(np.float64)


class ImplicitShiftedDynamicsLunarLanderObsWrapper(ObservationWrapper):
    """
    Implicit, env-parameter-derived, episode-shifted auxiliary features.
    Affects state and action for env
    """
    def __init__(self, env, scale=2.0, noise_level=0.0):
        super().__init__(env)
        self.scale = scale
        self.noise_level = noise_level
        self.n_aux = 2

        low = np.concatenate([env.observation_space.low, -np.inf * np.ones(self.n_aux)])
        high = np.concatenate([env.observation_space.high, np.inf * np.ones(self.n_aux)])
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

    def reset(self, **kwargs):
        self.env.MAIN_ENGINE_POWER = np.random.uniform(11.0, 15.0)
        self.env.SIDE_ENGINE_POWER = np.random.uniform(0.4, 0.8)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs):
        aux = np.array(
            [
                self.env.MAIN_ENGINE_POWER / 15.0,
                self.env.SIDE_ENGINE_POWER * 2.0,
            ],
            dtype=np.float64,
        ) * self.scale
        if self.noise_level > 0:
            obs = obs + np.random.normal(0.0, self.noise_level, size=obs.shape)
        return np.concatenate([obs, aux]).astype(np.float64)


class ImplicitStaticDynamicsLunarLanderObsWrapper(ObservationWrapper):
    """
    Implicit, env-parameter-derived, non-shifted auxiliary features.
    Affects only state
    """
    def __init__(self, env, scale=5.0, noise_level=0.0):
        super().__init__(env)
        self.scale = scale
        self.noise_level = noise_level
        self.n_aux = 2

        low = np.concatenate([env.observation_space.low, -np.inf * np.ones(self.n_aux)])
        high = np.concatenate([env.observation_space.high, np.inf * np.ones(self.n_aux)])
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env.wind_power = np.random.uniform(12.5, 17.5)
        self.env.turbulence_power = np.random.uniform(1.0, 2.0)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs):
        wind = math.tanh(
            math.sin(0.02 * self.env.wind_idx)
            + math.sin(math.pi * 0.01 * self.env.wind_idx)
        ) * self.env.wind_power

        torque = math.tanh(
            math.sin(0.02 * self.env.torque_idx)
            + math.sin(math.pi * 0.01 * self.env.torque_idx)
        ) * self.env.turbulence_power

        aux = np.array([wind, torque], dtype=np.float64) * self.scale
        if self.noise_level > 0:
            obs = obs + np.random.normal(0.0, self.noise_level, size=obs.shape)
        return np.concatenate([obs, aux]).astype(np.float64)


class NonCausalInternalLunarLanderObsWrapper(ObservationWrapper):
    """Non-causal internal signals appended to observation."""

    def __init__(self, env, scale=2.0, noise_level=0.0):
        super().__init__(env)
        self.scale = scale
        self.noise_level = noise_level
        self.n_aux = 11

        low = np.concatenate([env.observation_space.low, -np.inf * np.ones(self.n_aux)])
        high = np.concatenate([env.observation_space.high, np.inf * np.ones(self.n_aux)])
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs):
        aux = np.array(self.env.smooth_y, dtype=np.float64) * self.scale
        if self.noise_level > 0:
            obs = obs + np.random.normal(0.0, self.noise_level, size=obs.shape)
        return np.concatenate([obs, aux]).astype(np.float64)
