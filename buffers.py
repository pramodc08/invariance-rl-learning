import torch
import numpy as np
from collections import namedtuple, deque


class ReplayBuffer:
    """
    Standard Replay Buffer.
    """
    def __init__(self, action_size, buffer_size, batch_size, device, n_reward_steps=3):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        self.n_reward_steps = n_reward_steps

        self.temporal_buffer = deque(maxlen=n_reward_steps)
        self.experience = namedtuple("Experience", 
            field_names=[
            "state", "action", "reward", "next_state", "done",
            "env_id",             # <--- New: Which environment is this?
            "future_rewards",     # <--- New: Vector of next N rewards
            "future_actions"      # <--- New: Vector of next N actions
        ])

    def add(self, state, action, reward, next_state, done, env_id):
        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "env_id": env_id
        }
        self.temporal_buffer.append(transition)

        if len(self.temporal_buffer) >= self.n_reward_steps:
            self._commit_transition()

        if done:
            while len(self.temporal_buffer) > 0:
                self._commit_transition(force_flush=True)

    def _commit_transition(self, force_flush=False):
        """
        Internal: Takes the oldest item in temporal_buffer, gathers future info, 
        and saves it to main memory.
        """
        # The transition we are trying to save is at index 0
        current = self.temporal_buffer[0]
        # We need to look ahead at indices 1..N
        future_rewards = []
        future_actions = []

        for i in range(0, self.n_reward_steps):
            if i < len(self.temporal_buffer):
                step = self.temporal_buffer[i]
                clip_reward = float(step["reward"]) # float(np.clip(step["reward"], -1.0, 1.0))
                future_rewards.append(clip_reward)
                future_actions.append(step["action"])
            else:
                # Padding if episode ended early (zeros)
                future_rewards.append(0.0)
                future_actions.append(0) # Assuming 0 is a safe padding action

        # Create the final namedtuple
        e = self.experience(
            current["state"], current["action"], current["reward"], current["next_state"], current["done"],
            current["env_id"],
            np.array(future_rewards, dtype=np.float32),
            np.array(future_actions, dtype=np.int64)
        )
        self.memory.append(e)
        
        # Remove the processed item from the temporal buffer (unless we are flushing everything)
        if not force_flush:
            self.temporal_buffer.popleft()
        else:
            # During flush, we just popped 0 via logic, but we need to ensure the loop continues externally
            self.temporal_buffer.popleft()

    def sample(self):
        indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False)
        experiences = [self.memory[i] for i in indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        env_ids = torch.from_numpy(np.vstack([e.env_id for e in experiences])).long().to(self.device).squeeze(1)
        f_rewards = torch.from_numpy(np.vstack([e.future_rewards for e in experiences])).float().to(self.device)
        f_actions = torch.from_numpy(np.vstack([e.future_actions for e in experiences])).long().to(self.device)

        return (states, actions, rewards, next_states, dones, env_ids, f_rewards, f_actions)

    def __len__(self):
        return len(self.memory)


class HiddenStateReplayBuffer:
    """
    LSTM Replay Buffer (Stored State).
    Stores the LSTM hidden states (h, c) alongside the transition.
    This removes the need to store episodes or sample sequences.
    """
    def __init__(self, action_size, buffer_size, batch_size, device, n_reward_steps=3):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        self.n_reward_steps = n_reward_steps

        self.temporal_buffer = deque(maxlen=n_reward_steps)
        # We add 'hidden' and 'next_hidden' to the experience
        self.experience = namedtuple("Experience", 
            field_names=[
                "state", "action", "reward", "next_state", "done", 
                "hidden", "next_hidden",
                "env_id",             # <--- New: Which environment is this?
                "future_rewards",     # <--- New: Vector of next N rewards
                "future_actions"      # <--- New: Vector of next N actions
        ])
        

    def add(self, state, action, reward, next_state, done, env_id, hidden, next_hidden):
        """
        hidden: tuple (h, c) representing hidden state BEFORE action
        next_hidden: tuple (h, c) representing hidden state AFTER action (for target calculation)
        """
        # Store (h, c) as numpy arrays to save memory/detach from graph
        # Assuming hidden is (h, c) from PyTorch: (num_layers, batch, hidden_size)
        
        # Helper to convert tensor hidden state to numpy
        def to_numpy(h_tuple):
            return (h_tuple[0].detach().cpu().numpy(), h_tuple[1].detach().cpu().numpy())

        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "env_id": env_id,
            "hidden": to_numpy(hidden),
            "next_hidden": to_numpy(next_hidden),
        }
        self.temporal_buffer.append(transition)
        if len(self.temporal_buffer) >= self.n_reward_steps:
            self._commit_transition()

        if done:
            while len(self.temporal_buffer) > 0:
                self._commit_transition(force_flush=True)
    
    def _commit_transition(self, force_flush=False):
        """
        Internal: Takes the oldest item in temporal_buffer, gathers future info, 
        and saves it to main memory.
        """
        # The transition we are trying to save is at index 0
        current = self.temporal_buffer[0]
        # We need to look ahead at indices 1..N
        future_rewards = []
        future_actions = []

        for i in range(0, self.n_reward_steps):
            if i < len(self.temporal_buffer):
                step = self.temporal_buffer[i]
                clip_reward = float(step["reward"]) # float(np.clip(step["reward"], -1.0, 1.0))
                future_rewards.append(clip_reward)
                future_actions.append(step["action"])
            else:
                # Padding if episode ended early (zeros)
                future_rewards.append(0.0)
                future_actions.append(0) # Assuming 0 is a safe padding action

        # Create the final namedtuple
        e = self.experience(
            current["state"], current["action"], current["reward"], current["next_state"], current["done"],
            current["hidden"], current["next_hidden"],
            current["env_id"],
            np.array(future_rewards, dtype=np.float32),
            np.array(future_actions, dtype=np.int64)
        )
        self.memory.append(e)
        
        # Remove the processed item from the temporal buffer (unless we are flushing everything)
        if not force_flush:
            self.temporal_buffer.popleft()
        else:
            # During flush, we just popped 0 via logic, but we need to ensure the loop continues externally
            self.temporal_buffer.popleft()

    def sample(self):
        """
        Returns a batch of transitions including the hidden states.
        Output shapes for hidden states: (num_layers, batch_size, hidden_size)
        """
        indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False)
        experiences = [self.memory[i] for i in indices]

        # Standard components
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)

        # Hidden State components
        # e.hidden is a tuple (h, c). We need to stack 'h's together and 'c's together.
        # h shape in buffer: (layers, 1, hidden) -> stack along dim 1 -> (layers, batch, hidden)
        
        # Current Hidden (h, c)
        h_batch = np.stack([e.hidden[0] for e in experiences], axis=1).squeeze(2) 
        c_batch = np.stack([e.hidden[1] for e in experiences], axis=1).squeeze(2)
        
        # Next Hidden (h, c)
        nh_batch = np.stack([e.next_hidden[0] for e in experiences], axis=1).squeeze(2)
        nc_batch = np.stack([e.next_hidden[1] for e in experiences], axis=1).squeeze(2)

        # Convert to tensors
        hidden = (torch.from_numpy(h_batch).float().to(self.device), 
                  torch.from_numpy(c_batch).float().to(self.device))
        
        next_hidden = (torch.from_numpy(nh_batch).float().to(self.device), 
                       torch.from_numpy(nc_batch).float().to(self.device))
        
        env_ids = torch.from_numpy(np.vstack([e.env_id for e in experiences])).long().to(self.device).squeeze(1)
        f_rewards = torch.from_numpy(np.vstack([e.future_rewards for e in experiences])).float().to(self.device)
        f_actions = torch.from_numpy(np.vstack([e.future_actions for e in experiences])).long().to(self.device)

        return (states, actions, rewards, next_states, dones, env_ids, f_rewards, f_actions, hidden, next_hidden,)

    def __len__(self):
        return len(self.memory)
    