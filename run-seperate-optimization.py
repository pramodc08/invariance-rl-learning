import os
import toml
import math
from datetime import datetime
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
from buffers import HiddenStateReplayBuffer, ReplayBuffer
from agent import Agent, calculate_q_loss, calculate_grl_loss, calculate_dro_loss, calculate_irm_loss, calculate_vrex_loss
from utils import argparse_generator, set_seed
from torch.utils.tensorboard import SummaryWriter


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
    main_engine_power_range = config.get("wrapper_main_engine_power_range")
    side_engine_power_range = config.get("wrapper_side_engine_power_range")
    wind_power_range = config.get("wrapper_wind_power_range")
    turbulence_power_range = config.get("wrapper_turbulence_power_range")
    main_engine_power_range_test = config.get("wrapper_main_engine_power_range_test")
    side_engine_power_range_test = config.get("wrapper_side_engine_power_range_test")
    wind_power_range_test = config.get("wrapper_wind_power_range_test")
    turbulence_power_range_test = config.get("wrapper_turbulence_power_range_test")

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
            wrapper_kwargs = {
                "scale": scales[i],
                "noise_level": noises[i],
            }
            if wrapper_name == "implicit_shifted":
                if main_engine_power_range is not None:
                    wrapper_kwargs["main_engine_power_range"] = main_engine_power_range[i]
                if side_engine_power_range is not None:
                    wrapper_kwargs["side_engine_power_range"] = side_engine_power_range[i]
            elif wrapper_name == "implicit_static":
                if wind_power_range is not None:
                    wrapper_kwargs["wind_power_range"] = wind_power_range[i]
                if turbulence_power_range is not None:
                    wrapper_kwargs["turbulence_power_range"] = turbulence_power_range[i]
            env = wrapper_cls(env, **wrapper_kwargs)
        env.reset(seed=seed)
        envs.append(env)

    test_envs = []
    for i, _ in enumerate(scales_test):
        env = LunarLander(**ctor_kwargs)
        if wrapper_cls is not None:
            wrapper_kwargs = {
                "scale": scales_test[i],
                "noise_level": noises_test[i],
            }
            if wrapper_name == "implicit_shifted":
                if main_engine_power_range is not None:
                    wrapper_kwargs["main_engine_power_range"] = main_engine_power_range_test[i]
                if side_engine_power_range is not None:
                    wrapper_kwargs["side_engine_power_range"] = side_engine_power_range_test[i]
            elif wrapper_name == "implicit_static":
                if wind_power_range is not None:
                    wrapper_kwargs["wind_power_range"] = wind_power_range_test[i]
                if turbulence_power_range is not None:
                    wrapper_kwargs["turbulence_power_range"] = turbulence_power_range_test[i]
            env = wrapper_cls(env, **wrapper_kwargs)
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
        n_envs=len(envs), 
        n_reward_steps=agent_config["n_reward_steps"],
    )

    buffers = [
        HiddenStateReplayBuffer(
            action_size=action_shape,
            buffer_size=agent_config["replay_buffer_size"],
            batch_size=agent_config["batch_size"],
            device=device,
            n_reward_steps=agent_config["n_reward_steps"]
        ) if agent_config["recurrent"] else ReplayBuffer(
            action_size=action_shape,
            buffer_size=agent_config["replay_buffer_size"],
            batch_size=agent_config["batch_size"],
            device=device,
            n_reward_steps=agent_config["n_reward_steps"]
        )
        for _ in envs
    ]

    run_name = f"{run_config.get('title', 'run')}_{seed}"
    log_dir = os.path.join(run_config.get("log_dir", "logs"), run_name)
    writer = SummaryWriter(log_dir=log_dir)
    save_dir = os.path.join(run_config.get("save_dir", "model"), run_name)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        toml.dump(config, f)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "checkpoint.pt")
    best_path = os.path.join(save_dir, "best.pt")
    save_freq = int(run_config.get("save_freq", 100))
    save_best = bool(run_config.get("save_best", True))
    restart = bool(run_config.get("restart", False))
    best_metric = -float("inf")

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
    start_episode = 1
    if not restart and os.path.exists(checkpoint_path):
        metadata = agent.load_checkpoint(checkpoint_path)
        eps = float(metadata.get("epsilon", eps))
        start_episode = int(metadata.get("episode", 0)) + 1
        best_metric = float(metadata.get("best_metric", best_metric))
        print(f"Loaded checkpoint from {checkpoint_path} (episode {start_episode - 1}).")
    t_step = 0
    print("Training started...")
    for i_episode in range(start_episode, n_episodes + 1):
        states = []
        hiddens = []
        dones = [False] * n_envs
        episode_scores = np.zeros(n_envs, dtype=np.float32)
        for env in envs:
            s, _ = env.reset()
            states.append(s)
            hiddens.append(agent.qnetwork_local.init_hidden(1))
        # --- rollout ---
        for _ in range(max_steps):
            for i, env in enumerate(envs):
                if dones[i]:
                    continue
                action, next_hidden = agent.act(states[i], eps, hiddens[i])
                next_state, reward, done, truncated, _ = env.step(action)
                terminal = done or truncated
                env_id = i

                if agent.recurrent:
                    buffers[i].add(states[i], action, reward, next_state, terminal, env_id, hiddens[i], next_hidden)
                else:
                    buffers[i].add(states[i], action, reward, next_state, terminal, env_id)

                states[i] = next_state
                hiddens[i] = next_hidden
                episode_scores[i] += reward
                dones[i] = terminal

            # Train step
            t_step += 1
            if all([len(i) > agent.batch_size for i in buffers]):
                t_step = (t_step + 1) % agent.update_every
                if t_step == 0:
                    for _ in range(len(envs)):
                        loss_values = {
                            "q_loss": [],
                            "reward_prediction_loss": [],

                            "grl_loss": [],
                            "grl_loss_weigthed": [],
                            "grl_loss_total_loss": [],
                            "grl_loss_total_loss_weigthed": [],

                            "dro_loss": [],
                            "dro_loss_weigthed": [],
                            "dro_loss_total_loss": [],
                            "dro_loss_total_loss_weigthed": [],

                            "irm_loss": [],
                            "irm_weigthed": [],
                            "irm_total_loss": [],
                            "irm_total_loss_weigthed": [],

                            "vrex_loss": [],
                            "vrex_weigthed": [],
                            "vrex_total_loss": [],
                            "vrex_total_loss_weigthed": [],

                            "aux_loss": []
                        }
                        
                        for _ in range(agent.n_epochs):
                            # --- Phase 1: Q-Learning ---
                            sampled_experiences = []
                            q_losses = []
                            
                            for i, env in enumerate(envs):
                                experience = buffers[i].sample()
                                sampled_experiences.append(experience)
                                loss_i, _ = calculate_q_loss(agent, experience)
                                q_losses.append(loss_i)

                            q_loss = torch.stack(q_losses).mean()
                            agent.optimizer.zero_grad()
                            q_loss.backward()
                            if agent.clip_grad is not None and agent.clip_grad > 0:
                                torch.nn.utils.clip_grad_norm_(agent.qnetwork_local.parameters(), agent.clip_grad)
                            agent.optimizer.step()

                            # --- Phase 2: Auxiliary Tasks ---
                            reward_prediction_losses = []
                            grl_losses = []
                            dro_losses = []
                            irm_losses = []
                            vrex_losses = []
                            
                            for i, env in enumerate(envs):
                                experience = sampled_experiences[i]
                                # Re-calculate features since network weights changed
                                _, features = calculate_q_loss(agent, experience)
                                
                                grl_loss, _ = calculate_grl_loss(agent, features, experience)
                                dro_loss, _ = calculate_dro_loss(agent, features, experience, i)
                                irm_loss, _ = calculate_irm_loss(agent, features, experience)
                                vrex_loss = calculate_vrex_loss(agent, features, experience)
                                
                                grl_losses.append(grl_loss)
                                dro_losses.append(dro_loss)
                                irm_losses.append(irm_loss)
                                vrex_losses.append(vrex_loss)
                                reward_prediction_losses.append(vrex_loss)

                            grl_loss = torch.stack(grl_losses).mean()
                            dro_loss = torch.stack(dro_losses).sum()
                            irm_loss = torch.stack(irm_losses).sum()
                            vrex_loss = torch.var(torch.stack(vrex_losses))
                            reward_prediction_loss = torch.stack(reward_prediction_losses).mean()
                            
                            aux_loss = agent.grl_lambda * (reward_prediction_loss + agent.grl_weight * grl_loss) + \
                                       agent.dro_lambda * (reward_prediction_loss + agent.dro_weight * dro_loss) + \
                                       agent.irm_lambda * (reward_prediction_loss + agent.irm_penalty_multiplier * irm_loss) + \
                                       agent.vrex_lambda * (reward_prediction_loss + agent.vrex_penalty_multiplier * vrex_loss)

                            agent.optimizer_aux.zero_grad()
                            aux_loss.backward()
                            if agent.clip_grad is not None and agent.clip_grad > 0:
                                torch.nn.utils.clip_grad_norm_(agent.qnetwork_local.parameters(), agent.clip_grad)
                            agent.optimizer_aux.step()
                            
                            agent.dro_loss_fn.update()
                            
                            # --- Logging ---
                            loss_values["q_loss"].append(q_loss.item())
                            loss_values["reward_prediction_loss"].append(reward_prediction_loss.item())
                            
                            loss_values["grl_loss"].append(grl_loss.item())
                            loss_values["grl_loss_weigthed"].append(grl_loss.item() * agent.grl_weight)
                            loss_values["grl_loss_total_loss"].append((reward_prediction_loss + agent.grl_weight * grl_loss).item())
                            loss_values["grl_loss_total_loss_weigthed"].append((reward_prediction_loss + agent.grl_weight * grl_loss).item() * agent.grl_lambda)

                            loss_values["dro_loss"].append(dro_loss.item())
                            loss_values["dro_loss_weigthed"].append(dro_loss.item() * agent.dro_lambda)
                            loss_values["dro_loss_total_loss"].append((reward_prediction_loss + agent.dro_weight * dro_loss).item())
                            loss_values["dro_loss_total_loss_weigthed"].append((reward_prediction_loss + agent.dro_weight * dro_loss).item() * agent.dro_lambda)

                            loss_values["irm_loss"].append(irm_loss.item())
                            loss_values["irm_weigthed"].append(irm_loss.item() * agent.irm_penalty_multiplier)
                            loss_values["irm_total_loss"].append((reward_prediction_loss + agent.irm_penalty_multiplier * irm_loss).item())
                            loss_values["irm_total_loss_weigthed"].append((reward_prediction_loss + agent.irm_penalty_multiplier * irm_loss).item() * agent.irm_lambda)

                            loss_values["vrex_loss"].append(vrex_loss.item())
                            loss_values["vrex_weigthed"].append(vrex_loss.item() * agent.vrex_penalty_multiplier)
                            loss_values["vrex_total_loss"].append((reward_prediction_loss + agent.vrex_penalty_multiplier * vrex_loss).item())
                            loss_values["vrex_total_loss_weigthed"].append((reward_prediction_loss + agent.vrex_penalty_multiplier * vrex_loss).item() * agent.vrex_lambda)
                            
                            loss_values["aux_loss"].append((aux_loss).item())

                        agent.irm_penalty_multiplier = 0.0 # float(min(i_episode, 1500)/1500.0*100.0) ** 1.6
                        agent.vrex_penalty_multiplier = 0.0 # float(min(i_episode, 1500)/1500.0*100.0) ** 1.6
                        agent.soft_update(agent.qnetwork_local, agent.qnetwork_target, agent.tau)
                        if agent.lr_decay is not None and agent.lr_decay != 1.0:
                            for param_group in agent.optimizer.param_groups:
                                param_group["lr"] *= agent.lr_decay
                        if loss_values:
                            agent.last_loss = loss_values
                            agent.last_lr = float(agent.optimizer.param_groups[0]["lr"])  

            if all(dones):
                break

        for i in range(n_envs):
            ep_ret = float(episode_scores[i])
            train_returns[i].append(ep_ret)
            scores_window[i].append(ep_ret)
            train_mean_returns[i].append(np.mean(scores_window[i]))
            writer.add_scalar(f"train/episode_return_env{i}", ep_ret, i_episode)
            writer.add_scalar(
                f"train/avg100_return_env{i}",
                float(np.mean(scores_window[i])),
                i_episode,
            )

        eps = max(epsilon_final, epsilon_decay * eps)
        writer.add_scalar("train/epsilon", eps, i_episode)
        writer.add_scalar("train/lr", agent.last_lr, i_episode)
        writer.add_scalar(f"train/irm_penalty_multiplier", float(agent.irm_penalty_multiplier), i_episode)
        writer.add_scalar(f"train/vrex_penalty_multiplier", float(agent.vrex_penalty_multiplier), i_episode)
        if agent.last_loss is not None:
            for each_key, value in agent.last_loss.items():
                writer.add_scalar(f"train/{each_key}", float(np.mean(value)), i_episode)
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
            eval_means = []
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
                eval_mean = float(np.mean(eval_list))
                eval_var = float(np.var(eval_list))
                env_eval_returns[i].append(eval_mean)
                eval_means.append(eval_mean)
                writer.add_scalar(f"eval/env{i}_mean", eval_mean, i_episode)
                writer.add_scalar(f"eval/env{i}_var", eval_var, i_episode)
                
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
                test_mean = float(np.mean(eval_list))
                test_var = float(np.var(eval_list))
                test_eval_returns[i].append(test_mean)
                writer.add_scalar(f"eval_test/env{i}_mean", test_mean, i_episode)
                writer.add_scalar(f"eval_test/env{i}_var", test_var, i_episode)
            if eval_means:
                eval_mean_over_envs = float(np.mean(eval_means))
                writer.add_scalar("eval/mean_over_envs", eval_mean_over_envs, i_episode)
                if save_best and eval_mean_over_envs > best_metric:
                    best_metric = eval_mean_over_envs
                    agent.save_checkpoint(
                        best_path,
                        metadata={
                            "episode": i_episode,
                            "epsilon": eps,
                            "best_metric": best_metric,
                        },
                    )
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
        if save_freq > 0 and i_episode % save_freq == 0:
            agent.save_checkpoint(
                checkpoint_path,
                metadata={
                    "episode": i_episode,
                    "epsilon": eps,
                    "best_metric": best_metric,
                },
            )
        writer.flush()
    writer.close()


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
