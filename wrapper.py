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


def _validate_range(name, value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a list/tuple with 2 values")
    low = float(value[0])
    high = float(value[1])
    if low > high:
        raise ValueError(f"{name} min must be <= max (got {low} > {high})")
    return (low, high)


class ImplicitShiftedDynamicsLunarLanderObsWrapper(ObservationWrapper):
    """
    Implicit, env-parameter-derived, episode-shifted auxiliary features.
    Affects state and action for env
    """
    def __init__(
        self,
        env,
        scale=2.0,
        noise_level=0.0,
        main_engine_power_range=(11.0, 15.0),
        side_engine_power_range=(0.4, 0.8),
    ):
        super().__init__(env)
        self.scale = scale
        self.noise_level = noise_level
        self.n_aux = 2
        self.main_engine_power_range = _validate_range(
            "main_engine_power_range", main_engine_power_range
        )
        self.side_engine_power_range = _validate_range(
            "side_engine_power_range", side_engine_power_range
        )

        low = np.concatenate([env.observation_space.low, -np.inf * np.ones(self.n_aux)])
        high = np.concatenate([env.observation_space.high, np.inf * np.ones(self.n_aux)])
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

    def reset(self, **kwargs):
        self.env.MAIN_ENGINE_POWER = np.random.uniform(*self.main_engine_power_range)
        self.env.SIDE_ENGINE_POWER = np.random.uniform(*self.side_engine_power_range)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs):
        main_max = self.main_engine_power_range[1] or 1.0
        side_max = self.side_engine_power_range[1] or 1.0
        aux = np.array(
            [
                self.env.MAIN_ENGINE_POWER / main_max,
                self.env.SIDE_ENGINE_POWER / side_max,
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
    def __init__(
        self,
        env,
        scale=5.0,
        noise_level=0.0,
        wind_power_range=(12.5, 17.5),
        turbulence_power_range=(1.0, 2.0),
    ):
        super().__init__(env)
        self.scale = scale
        self.noise_level = noise_level
        self.n_aux = 2
        self.wind_power_range = _validate_range("wind_power_range", wind_power_range)
        self.turbulence_power_range = _validate_range(
            "turbulence_power_range", turbulence_power_range
        )

        low = np.concatenate([env.observation_space.low, -np.inf * np.ones(self.n_aux)])
        high = np.concatenate([env.observation_space.high, np.inf * np.ones(self.n_aux)])
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env.wind_power = np.random.uniform(*self.wind_power_range)
        self.env.turbulence_power = np.random.uniform(*self.turbulence_power_range)
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
