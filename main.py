import toml
import math
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from inspect import signature
from copy import deepcopy

from lunar_lander import LunarLander
from wrapper import (
    ExplicitStaticLunarLanderObsWrapper,
    ImplicitShiftedDynamicsLunarLanderObsWrapper,
    ImplicitStaticDynamicsLunarLanderObsWrapper,
    NonCausalInternalLunarLanderObsWrapper,
)
from agent import Agent
from utils import argparse_generator, set_seed


WRAPPER_REGISTRY = {
    "none": None,
    "explicit_static": ExplicitStaticLunarLanderObsWrapper,
    "implicit_shifted": ImplicitShiftedDynamicsLunarLanderObsWrapper,
    "implicit_static": ImplicitStaticDynamicsLunarLanderObsWrapper,
    "non_causal_internal": NonCausalInternalLunarLanderObsWrapper,
}


def make_env_from_config(config: dict, seed: int):
    # Only pass parameters that LunarLander(...) accepts
    ll_params = set(signature(LunarLander).parameters.keys())
    ctor_kwargs = {k: v for k, v in config.items() if k in ll_params}

    # --- Wrapper config ---
    wrapper_name = config.get("wrapper_class", "none")
    wrapper_cls = WRAPPER_REGISTRY.get(wrapper_name)
    scales = config.get("wrapper_scale", [0.0])
    noises = config.get("wrapper_noise_level", [0.0])
    scales_test = config.get("wrapper_scale_test", [0.0])
    noises_test = config.get("wrapper_noise_level_test", [0.0])

    assert len(scales) == len(noises), (
        f"wrapper_scale and wrapper_noise_level must have same length "
        f"(got {len(scales)} vs {len(noises)})"
    )
    assert len(scales_test) == len(noises_test), (
        f"wrapper_scale_test and wrapper_noise_level_test must have same length "
        f"(got {len(scales_test)} vs {len(noises_test)})"
    )

    envs = []
    for i, _ in enumerate(scales):
        env = LunarLander(**ctor_kwargs)
        if wrapper_cls is not None:
            env = wrapper_cls(
                env,
                scale=scales[i],
                noise_level=noises[i],
            )
        env.reset(seed=seed)
        envs.append(env)

    test_envs = []
    for i, _ in enumerate(scales_test):
        env = LunarLander(**ctor_kwargs)
        if wrapper_cls is not None:
            env = wrapper_cls(
                env,
                scale=scales_test[i],
                noise_level=noises_test[i],
            )
        env.reset(seed=seed)
        test_envs.append(env)

    return envs, test_envs


def run(config: dict):
    run_config = config.get("RUN")
    agent_config = config.get("AGENT")
    environment_config = config.get("ENVIRONMENT")

    seed = run_config.get("seed")
    set_seed(seed=seed)
    device = torch.device(agent_config["device"])
    print(f"Using device: {device}")

    envs, test_envs = make_env_from_config(environment_config, seed=seed)
    state_shape = envs[0].observation_space.shape[0]
    action_shape = envs[0].action_space.n

    agent = Agent(
        state_size=state_shape,
        action_size=action_shape,
        device=device,
        network_hidden_layers=agent_config["network_hidden_layers"],
        recurrent=agent_config["recurrent"],
        lr=agent_config["lr"],
        lr_decay=agent_config["lr_decay"],
        buffer_size=agent_config["replay_buffer_size"],
        batch_size=agent_config["batch_size"],
        gamma=agent_config["gamma"],
        steps_before_learning=agent_config["steps_before_learning"],
        clip_grad=agent_config["clip_grad"],
        tau=agent_config["tau"],
        update_every=agent_config["network_update_freq"],
        n_epochs=agent_config["n_epochs"],
    )

    n_envs = len(envs)
    n_test_envs = len(test_envs)
    train_returns = [[] for _ in range(n_envs)]
    scores_window = [deque(maxlen=100) for _ in range(n_envs)]
    train_mean_returns = [[] for _ in range(n_envs)]  # avg value during training rollout
    env_eval_returns  = [[] for _ in range(n_envs)]
    test_eval_returns = [[] for _ in range(n_test_envs)]
    
    epsilon_initial = agent_config["epsilon_initial"]
    epsilon_final = agent_config["epsilon_final"]
    epsilon_decay = agent_config["epsilon_decay"]
    n_episodes = run_config["episodes"]
    max_steps = run_config["max_steps"]
    evaluation_episodes = run_config["evaluation_episodes"]
    eps = epsilon_initial
    print("Training started...")
    for i_episode in range(1, n_episodes + 1):
        states = []
        hiddens = []
        dones = [False] * n_envs
        episode_scores = np.zeros(n_envs, dtype=np.float32)
        for env in envs:
            s, _ = env.reset()
            states.append(s)
            hiddens.append(agent.qnetwork_local.init_hidden(1))
        # --- rollout ---
        for t in range(max_steps):
            for i, env in enumerate(envs):
                if dones[i]:
                    continue
                action, next_hidden = agent.act(states[i], eps, hiddens[i])
                next_state, reward, done, truncated, _ = env.step(action)
                terminal = done or truncated
                agent.step(
                    states[i], action, reward,
                    next_state, terminal,
                    hiddens[i], next_hidden
                )
                states[i] = next_state
                hiddens[i] = next_hidden
                episode_scores[i] += reward
                dones[i] = terminal
            if all(dones):
                break

        for i in range(n_envs):
            ep_ret = float(episode_scores[i])
            train_returns[i].append(ep_ret)
            scores_window[i].append(ep_ret)
            train_mean_returns[i].append(np.mean(scores_window[i]))

        eps = max(epsilon_final, epsilon_decay * eps)
        env_stats = " | ".join(
            f"Env{i}: {np.mean(scores_window[i]):.2f}"
            for i in range(n_envs)
        )
        print(
            f"\rEpisode {i_episode}"
            f"\t{env_stats}"
            f"\tEps: {eps:.3f}",
            end=""
        )

        if i_episode % evaluation_episodes == 0:
            for i, env in enumerate(envs):
                eval_list = []
                for _ in range(5):
                    state, _ = env.reset()
                    hidden = agent.qnetwork_local.init_hidden(1)
                    total = 0.0

                    for t in range(max_steps):
                        action, next_hidden = agent.act(state, eps=0.0, hidden=hidden)
                        next_state, reward, done, truncated, _ = env.step(action)
                        state = next_state
                        hidden = next_hidden
                        total += reward
                        if done or truncated:
                            break
                    eval_list.append(total)
                env_eval_returns[i].append(float(np.mean(eval_list)))
                
            for i, env in enumerate(test_envs):
                eval_list = []
                for _ in range(5):
                    state, _ = env.reset()
                    hidden = agent.qnetwork_local.init_hidden(1)
                    total = 0.0
                    for t in range(max_steps):
                        action, next_hidden = agent.act(
                            state, eps=0.0, hidden=hidden
                        )
                        next_state, reward, done, truncated, _ = env.step(action)
                        state = next_state
                        hidden = next_hidden
                        total += reward
                        if done or truncated:
                            break
                    eval_list.append(total)
                test_eval_returns[i].append(np.mean(eval_list))
        # --- clean newline every 100 eps ---
        if i_episode % 100 == 0:
            print("\n" + "-" * 90)
            print(f"Episode {i_episode}")
            for i in range(n_envs):
                line = f"  Env{i}: Train Avg(100)={np.mean(scores_window[i]):.2f}"
                if env_eval_returns[i]:
                    line += f" | Eval={env_eval_returns[i][-1]:.2f}"
                print(line)

            for i in range(n_test_envs):
                print(f"  TestEnv{i}: Test Eval={test_eval_returns[i][-1]:.2f}")


if __name__ == "__main__":
    from torch.multiprocessing import Process, set_start_method
    from functools import partial

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    
    # Create a config file and pass here
    config = argparse_generator("config/default-config.toml")
    # print(config)
    run(config=config)