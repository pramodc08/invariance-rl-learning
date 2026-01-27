from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# Import your previously created modules
# Assuming they are named 'networks' and 'replay_buffers'
from model import DeepQNetwork, RecurrentDeepQNetwork
from buffers import ReplayBuffer, HiddenStateReplayBuffer

class Agent:
    def __init__(self, state_size, action_size, device, network_hidden_layers=(256, 128, 64), recurrent=False,
                 lr=1e-4, buffer_size=50000, batch_size=64, gamma=0.99,
                 tau=1e-3, update_every=5, n_epochs=1, steps_before_learning=0,
                 clip_grad=1.0, lr_decay=1.0, n_envs=3, n_reward_steps=3):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.recurrent = recurrent
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.n_epochs = n_epochs
        self.steps_before_learning = 0 if steps_before_learning is None else max(0, int(steps_before_learning))
        self.clip_grad = clip_grad
        self.lr_decay = lr_decay
        self.network_hidden_layers=network_hidden_layers
        self.n_reward_steps = n_reward_steps

        # Q-Network & Memory Initialization
        if self.recurrent:
            # LSTM Mode
            print("Initializing Recurrent DQN Agent (LSTM)...")
            self.qnetwork_local = RecurrentDeepQNetwork(state_size, action_size, n_envs, n_reward_steps, device, network_hidden_layers=network_hidden_layers).to(device)
            self.qnetwork_target = RecurrentDeepQNetwork(state_size, action_size, n_envs, n_reward_steps, device, network_hidden_layers=network_hidden_layers).to(device)
            self.memory = HiddenStateReplayBuffer(action_size, buffer_size, batch_size, device, n_reward_steps=n_reward_steps)
        else:
            # Vanilla MLP Mode
            print("Initializing Vanilla DQN Agent (MLP)...")
            self.qnetwork_local = DeepQNetwork(state_size, action_size, n_envs, n_reward_steps, device, network_hidden_layers=network_hidden_layers).to(device)
            self.qnetwork_target = DeepQNetwork(state_size, action_size, n_envs, n_reward_steps, device, network_hidden_layers=network_hidden_layers).to(device)
            self.memory = ReplayBuffer(action_size, buffer_size, batch_size, device, n_reward_steps=n_reward_steps)
        
        print(self.qnetwork_local)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.t_step = 0
        self.total_steps = 0
        self.last_loss = None
        self.last_lr = lr

    def step(self, state, action, reward, next_state, done, env_id=0, hidden=None, next_hidden=None):
        """
        Interacts with memory and triggers learning.
        Arguments 'hidden' and 'next_hidden' are only required if recurrent=True.
        """
        # Save experience in replay memory
        if self.recurrent:
            self.memory.add(state, action, reward, next_state, done, env_id, hidden, next_hidden)
        else:
            self.memory.add(state, action, reward, next_state, done, env_id)

        # Learn every UPDATE_EVERY time steps (after warmup)
        self.total_steps += 1
        self.t_step = (self.t_step + 1) % self.update_every
        if self.total_steps < self.steps_before_learning:
            return
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.0, hidden=None):
        """
        Returns actions for given state as per current policy.
        If recurrent, also returns the new hidden state.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = torch.tensor([0]*self.n_reward_steps).long().unsqueeze(0).to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            if self.recurrent:
                # LSTM forward pass expects (state, hidden)
                (action_values, _, _, _), new_hidden = self.qnetwork_local(state, action, hidden)
            else:
                action_values, _, _, _ = self.qnetwork_local(state, action)
                new_hidden = None

        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if np.random.uniform() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = np.random.choice(np.arange(self.action_size))
            
        return action, new_hidden

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples."""
        
        # Unpack experiences based on mode
        if self.recurrent:
            states, actions, rewards, next_states, dones, env_ids, f_rewards, f_actions, hidden, next_hidden = experiences
        else:
            states, actions, rewards, next_states, dones, env_ids, f_rewards, f_actions  = experiences

        # ------------------- Compute Target Q ------------------- #
        with torch.no_grad():
            if self.recurrent:
                # Double DQN for LSTM:
                # 1. Select best action using Local Net + Next Hidden State
                local_features, _ = self.qnetwork_local(next_states, f_actions, next_hidden)
                local_next_out = local_features[0]
                best_actions = local_next_out.argmax(1).unsqueeze(1)
                
                # 2. Evaluate using Target Net + Next Hidden State
                target_features, _ = self.qnetwork_target(next_states, f_actions, next_hidden)
                target_next_out = target_features[0]
                Q_targets_next = target_next_out.gather(1, best_actions)
            else:
                # Double DQN for MLP:
                local_features = self.qnetwork_local(next_states, f_actions)
                best_actions = local_features[0].argmax(1).unsqueeze(1)
                target_features = self.qnetwork_target(next_states, f_actions)
                Q_targets_next = target_features[0].gather(1, best_actions)

            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # ------------------- Compute Expected Q ------------------- #
        # We perform multiple epochs of updates on the sampled batch (optional, often 1 is standard)
        loss_values = []
        for _ in range(self.n_epochs):
            if self.recurrent:
                # Forward pass with the STORED hidden state from the buffer
                features, _ = self.qnetwork_local(states, f_actions, hidden)
                Q_expected = features[0].gather(1, actions)
            else:
                features = self.qnetwork_local(states, f_actions)
                Q_expected = features[0].gather(1, actions)

            # Compute loss
            loss = F.smooth_l1_loss(Q_expected, Q_targets)

            # Minimize loss
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad is not None and self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.clip_grad)
            self.optimizer.step()
            loss_values.append(loss.item())
        if self.lr_decay is not None and self.lr_decay != 1.0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.lr_decay
        if loss_values:
            self.last_loss = float(np.mean(loss_values))
            self.last_lr = float(self.optimizer.param_groups[0]["lr"])

        # ------------------- Update Target Network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save_checkpoint(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "qnetwork_local": self.qnetwork_local.state_dict(),
            "qnetwork_target": self.qnetwork_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "t_step": self.t_step,
            "total_steps": self.total_steps,
            "recurrent": self.recurrent,
            "network_hidden_layers": self.network_hidden_layers,
            "gamma": self.gamma,
            "tau": self.tau,
            "update_every": self.update_every,
            "n_epochs": self.n_epochs,
            "steps_before_learning": self.steps_before_learning,
            "clip_grad": self.clip_grad,
            "lr_decay": self.lr_decay,
        }
        if metadata:
            checkpoint["metadata"] = metadata
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, strict=True):
        checkpoint = torch.load(path, map_location=self.device)
        self.qnetwork_local.load_state_dict(checkpoint["qnetwork_local"], strict=strict)
        target_state = checkpoint.get("qnetwork_target", checkpoint["qnetwork_local"])
        self.qnetwork_target.load_state_dict(target_state, strict=strict)
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.t_step = checkpoint.get("t_step", 0)
        self.total_steps = checkpoint.get("total_steps", 0)
        self.steps_before_learning = checkpoint.get("steps_before_learning", self.steps_before_learning)
        self.clip_grad = checkpoint.get("clip_grad", self.clip_grad)
        self.lr_decay = checkpoint.get("lr_decay", self.lr_decay)
        return checkpoint.get("metadata", {})


def calculate_loss(agent, experiences):
    """Update value parameters using given batch of experience tuples."""
    if agent.recurrent:
        states, actions, rewards, next_states, dones, env_ids, f_rewards, f_actions, hidden, next_hidden = experiences
    else:
        states, actions, rewards, next_states, dones, env_ids, f_rewards, f_actions  = experiences


    # ------------------- Compute Target Q ------------------- #
    with torch.no_grad():
        if agent.recurrent:
            # Double DQN for LSTM:
            # 1. Select best action using Local Net + Next Hidden State
            local_features, _ = agent.qnetwork_local(next_states, f_actions, next_hidden)
            local_next_out = local_features[0]
            best_actions = local_next_out.argmax(1).unsqueeze(1)
            
            # 2. Evaluate using Target Net + Next Hidden State
            target_features, _ = agent.qnetwork_target(next_states, f_actions, next_hidden)
            target_next_out = target_features[0]
            Q_targets_next = target_next_out.gather(1, best_actions)
        else:
            # Double DQN for MLP:
            local_features = agent.qnetwork_local(next_states, f_actions)
            best_actions = local_features[0].argmax(1).unsqueeze(1)
            target_features = agent.qnetwork_target(next_states, f_actions)
            Q_targets_next = target_features[0].gather(1, best_actions)

        Q_targets = rewards + (agent.gamma * Q_targets_next * (1 - dones))

    if agent.recurrent:
        # Forward pass with the STORED hidden state from the buffer
        features, _ = agent.qnetwork_local(states, f_actions, hidden)
        Q_expected = features[0].gather(1, actions)
    else:
        features = agent.qnetwork_local(states, f_actions)
        Q_expected = features[0].gather(1, actions)

    # Compute loss
    loss = F.smooth_l1_loss(Q_expected, Q_targets)
    return loss, features