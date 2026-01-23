import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# Import your previously created modules
# Assuming they are named 'networks' and 'replay_buffers'
from model import DeepQNetwork, RecurrentDeepQNetwork
from buffers import ReplayBuffer, HiddenStateReplayBuffer

class Agent:
    def __init__(self, state_size, action_size, device, recurrent=False, 
                 lr=5e-4, buffer_size=100000, batch_size=64, gamma=0.99, 
                 tau=1e-3, update_every=5, n_epochs=1):
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.recurrent = recurrent
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.n_epochs = n_epochs

        # Q-Network & Memory Initialization
        if self.recurrent:
            # LSTM Mode
            print("Initializing Recurrent DQN Agent (LSTM)...")
            self.qnetwork_local = RecurrentDeepQNetwork(state_size, action_size, device).to(device)
            self.qnetwork_target = RecurrentDeepQNetwork(state_size, action_size, device).to(device)
            self.memory = HiddenStateReplayBuffer(action_size, buffer_size, batch_size, device)
        else:
            # Vanilla MLP Mode
            print("Initializing Vanilla DQN Agent (MLP)...")
            self.qnetwork_local = DeepQNetwork(state_size, action_size, device).to(device)
            self.qnetwork_target = DeepQNetwork(state_size, action_size, device).to(device)
            self.memory = ReplayBuffer(action_size, buffer_size, batch_size, device)
        
        print(self.qnetwork_local)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, hidden=None, next_hidden=None):
        """
        Interacts with memory and triggers learning.
        Arguments 'hidden' and 'next_hidden' are only required if recurrent=True.
        """
        # Save experience in replay memory
        if self.recurrent:
            self.memory.add(state, action, reward, next_state, done, hidden, next_hidden)
        else:
            self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_every
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
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            if self.recurrent:
                # LSTM forward pass expects (state, hidden)
                action_values, new_hidden = self.qnetwork_local(state, hidden)
            else:
                action_values = self.qnetwork_local(state)
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
            states, actions, rewards, next_states, dones, hidden, next_hidden = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        # ------------------- Compute Target Q ------------------- #
        with torch.no_grad():
            if self.recurrent:
                # Double DQN for LSTM:
                # 1. Select best action using Local Net + Next Hidden State
                local_next_out, _ = self.qnetwork_local(next_states, next_hidden)
                best_actions = local_next_out.argmax(1).unsqueeze(1)
                
                # 2. Evaluate using Target Net + Next Hidden State
                target_next_out, _ = self.qnetwork_target(next_states, next_hidden)
                Q_targets_next = target_next_out.gather(1, best_actions)
            else:
                # Double DQN for MLP:
                best_actions = self.qnetwork_local(next_states).argmax(1).unsqueeze(1)
                Q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions)

            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # ------------------- Compute Expected Q ------------------- #
        # We perform multiple epochs of updates on the sampled batch (optional, often 1 is standard)
        for _ in range(self.n_epochs):
            if self.recurrent:
                # Forward pass with the STORED hidden state from the buffer
                current_out, _ = self.qnetwork_local(states, hidden)
                Q_expected = current_out.gather(1, actions)
            else:
                Q_expected = self.qnetwork_local(states).gather(1, actions)

            # Compute loss
            loss = F.smooth_l1_loss(Q_expected, Q_targets)

            # Minimize loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
            self.optimizer.step()

        # ------------------- Update Target Network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

