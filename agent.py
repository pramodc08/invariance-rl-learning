from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# Import your previously created modules
# Assuming they are named 'networks' and 'replay_buffers'
from model import DeepQNetwork, RecurrentDeepQNetwork
from buffers import ReplayBuffer, HiddenStateReplayBuffer

class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


def grad_reverse(x, scale):
    return _GradReverse.apply(x, scale)


class RobustLoss(torch.nn.Module):
    def __init__(self, num_groups, mode='soft_dro', dro_step_size=0.01, group_decay=0.0, device='cuda'):
        """
        Args:
            mode: 'hard_max' (train on worst group) or 'soft_dro' (standard weighted average).
            dro_step_size: Learning rate for the group weights (eta).
            group_decay: Regularization pulling weights toward uniform (smoothing).
        """
        super().__init__()
        self.num_groups = num_groups
        self.mode = mode
        self.dro_step_size = dro_step_size
        self.group_decay = group_decay
        self.device = device
        
        # Persistent Group Weights (The "Adversary")
        self.group_weights = torch.ones(num_groups).to(device) / num_groups
        
        # Internal buffer to track raw losses during the loop
        self.raw_loss_buffer = torch.zeros(num_groups).to(device)
        self.counts = torch.zeros(num_groups).to(device) # To handle multiple hits per env if needed

    def forward(self, predicted_rewards, true_rewards, env_idx):
        """
        Calculates loss for backprop and internally tracks raw loss.
        Returns ONLY loss_dro.
        """
        # 1. Compute Raw Loss (MSE / Smooth L1)
        raw_loss = F.smooth_l1_loss(predicted_rewards, true_rewards)
        
        # 2. Track it internally (detach to stop graph retention)
        with torch.no_grad():
            self.raw_loss_buffer[env_idx] += raw_loss.detach()
            self.counts[env_idx] += 1
            
        # 3. Apply Strategy to get Backward-able Loss
        if self.mode == 'hard_max':
            # Binary mask: 1 if this is the worst group, 0 otherwise
            is_worst = (env_idx == torch.argmax(self.group_weights))
            loss_dro = raw_loss if is_worst else raw_loss * 0.0
            
        else: # 'soft_dro'
            # Standard DRO: weighted loss using current distribution q
            # Detach weight so we don't backprop into the adversary
            weight = self.group_weights[env_idx].detach()
            loss_dro = raw_loss * weight

        return loss_dro

    def update(self):
        """
        Updates group weights based on buffered losses, then resets buffer.
        Call this ONCE after processing all environments.
        """
        with torch.no_grad():
            # Average raw losses if we hit an env multiple times
            # (Avoid division by zero)
            avg_raw_losses = self.raw_loss_buffer / (self.counts + 1e-8)
            
            if self.mode == 'hard_max':
                # Hard Max Update: Shift weights towards the max loss group
                max_idx = torch.argmax(avg_raw_losses)
                target_weights = torch.zeros_like(self.group_weights)
                target_weights[max_idx] = 1.0
                
                # Smooth update (alpha=0.1) prevents oscillation
                self.group_weights = 0.9 * self.group_weights + 0.1 * target_weights
                
            elif self.mode == 'soft_dro':
                # 1. Exponentiated Gradient Update: w' = w * exp(eta * loss)
                self.group_weights *= torch.exp(self.dro_step_size * avg_raw_losses)
                
                # 2. Group Decay (Regularization): Pull towards uniform
                if self.group_decay > 0:
                    uniform = torch.ones_like(self.group_weights) / self.num_groups
                    self.group_weights = (1 - self.group_decay) * self.group_weights + \
                                         (self.group_decay) * uniform

                # 3. Normalize to sum to 1
                self.group_weights /= self.group_weights.sum()

            # Reset buffers for next step
            self.raw_loss_buffer.zero_()
            self.counts.zero_()


class Agent:
    def __init__(self, state_size, action_size, device, network_hidden_layers=(256, 128, 64), recurrent=False,
                 lr=1e-4, buffer_size=50000, batch_size=64, gamma=0.99,
                 tau=1e-3, update_every=5, n_epochs=1, steps_before_learning=0,
                 clip_grad=1.0, lr_decay=1.0, n_envs=3, n_reward_steps=3,
                 weight_decay=0.0001,
                 irm_lambda=0.0, irm_penalty_multiplier=1.0,
                 dro_lambda=0.0, dro_weight=1.0, dro_mode='hard_max', dro_step_size=0.01, dro_group_decay=0.0, 
                 grl_lambda=0.0, grl_weight=1.0, grl_alpha=1.0,
                 vrex_lambda=1.0, vrex_penalty_multiplier=10.0
                ):
        self.weight_decay = weight_decay
        # IRM-v1
        self.irm_lambda = float(irm_lambda)
        self.irm_penalty_multiplier = float(irm_penalty_multiplier)
        # VReX
        self.vrex_lambda = float(vrex_lambda)
        self.vrex_penalty_multiplier = float(vrex_penalty_multiplier)
        # DRO
        self.dro_lambda = float(dro_lambda)
        self.dro_weight = float(dro_weight)
        self.dro_step_size = float(dro_step_size)
        self.dro_group_decay = float(dro_group_decay)
        self.dro_mode = dro_mode
        self.dro_loss_fn = RobustLoss(
            num_groups=n_envs, 
            mode=dro_mode,       
            dro_step_size=dro_step_size,
            group_decay=dro_group_decay,       # Set > 0 for Group Decay comparison
            device=device
        )
        # GRL
        self.grl_alpha = float(grl_alpha)
        self.grl_lambda = float(grl_lambda)
        self.grl_weight = float(grl_weight)

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
        self.optimizer = optim.AdamW(self.qnetwork_local.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay)
        self.optimizer_aux = optim.AdamW(self.qnetwork_local.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay)
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
            "optimizer_aux": self.optimizer_aux.state_dict(),
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
        if "optimizer_aux" in checkpoint:
            self.optimizer_aux.load_state_dict(checkpoint["optimizer_aux"])
        self.t_step = checkpoint.get("t_step", 0)
        self.total_steps = checkpoint.get("total_steps", 0)
        self.steps_before_learning = checkpoint.get("steps_before_learning", self.steps_before_learning)
        self.clip_grad = checkpoint.get("clip_grad", self.clip_grad)
        self.lr_decay = checkpoint.get("lr_decay", self.lr_decay)
        return checkpoint.get("metadata", {})


def calculate_q_loss(agent, experiences):
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


def calculate_grl_loss(agent, features, experiences):
    if agent.recurrent:
        states, actions, rewards, next_states, dones, env_ids, f_rewards, f_actions, hidden, next_hidden = experiences
    else:
        states, actions, rewards, next_states, dones, env_ids, f_rewards, f_actions  = experiences
    encoding = features[3]
    
    predicted_rewards = features[2]
    raw_loss = F.smooth_l1_loss(predicted_rewards, f_rewards, reduction="none").mean()

    rev_encoding = grad_reverse(encoding, agent.grl_alpha)
    env_logits = agent.qnetwork_local.env_classifier(rev_encoding)
    grl_loss = F.cross_entropy(env_logits, env_ids)

    return grl_loss, raw_loss


def calculate_dro_loss(agent, features, experiences, env_index):
    """
    Docstring for calculate_dro_loss
    REquires l2 penalty regularizatio heavily
    """

    if agent.recurrent:
        states, actions, rewards, next_states, dones, env_ids, f_rewards, f_actions, hidden, next_hidden = experiences
    else:
        states, actions, rewards, next_states, dones, env_ids, f_rewards, f_actions  = experiences
    predicted_rewards = features[2]

    raw_loss = F.smooth_l1_loss(predicted_rewards, f_rewards, reduction="none").mean()
    loss_dro = agent.dro_loss_fn(predicted_rewards, f_rewards, env_idx=env_index)
    
    return loss_dro, raw_loss

"""
Why IRM Fails on Standard Q-Loss

    Non-Stationary Targets: In Q-learning, the "label" is the target value (r+γmaxQtarget​). This target changes every time the network updates. IRM tries to find a feature representation that is optimal across environments permanently. If the target keeps moving, the "optimal" representation keeps shifting, and the IRM penalty oscillates wildly.

    Bootstrapping: Q-learning bootstraps (uses its own predictions to update itself). Penalizing the gradient of a bootstrapped loss creates complex second-order effects that usually destroy learning stability.

The Solution: Reward Prediction (Auxiliary Task)

To fix this, researchers typically decouple the invariant feature learning from the policy learning.

    Shared Encoder: You have a feature extractor Φ(s).

    Reward Head: A simple linear layer predicts the immediate reward R(s). This is a stable, supervised regression task (the ground truth reward never changes).

    Q-Head: The standard Q-network uses the same features Φ(s) to estimate returns.

You apply the IRM penalty ONLY to the Reward Head. This forces Φ(s) to learn features that are stable and causal (invariant predictors of reward), which the Q-head then exploits to learn a robust policy.
"""

def calculate_irm_loss(agent, features, experiences):
    # 1. Dummy Scale for IRM
    scale = torch.nn.Parameter(torch.Tensor([1.0])).to(agent.device)
    # scale = torch.tensor(1.0, device=agent.device, requires_grad=True)

    if agent.recurrent:
        states, actions, rewards, next_states, dones, env_ids, f_rewards, f_actions, hidden, next_hidden = experiences
    else:
        states, actions, rewards, next_states, dones, env_ids, f_rewards, f_actions  = experiences

    predicted_rewards = features[2] * scale
    
    raw_loss = F.smooth_l1_loss(predicted_rewards, f_rewards, reduction="none")
    
    grad = torch.autograd.grad(raw_loss.mean(), scale, create_graph=True)[0]
    grad_loss = torch.sum(grad ** 2)

    return grad_loss, raw_loss.mean()

def calculate_vrex_loss(agent, features, experiences):
    if agent.recurrent:
        states, actions, rewards, next_states, dones, env_ids, f_rewards, f_actions, hidden, next_hidden = experiences
    else:
        states, actions, rewards, next_states, dones, env_ids, f_rewards, f_actions  = experiences

    predicted_rewards = features[2]
    raw_loss = F.smooth_l1_loss(predicted_rewards, f_rewards)
    return raw_loss