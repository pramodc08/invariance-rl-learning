import torch
import numpy as np
from collections import namedtuple, deque

class ReplayBuffer:
    """
    Standard Replay Buffer.
    """
    def __init__(self, action_size, buffer_size, batch_size, device):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False)
        experiences = [self.memory[i] for i in indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class HiddenStateReplayBuffer:
    """
    LSTM Replay Buffer (Stored State).
    Stores the LSTM hidden states (h, c) alongside the transition.
    This removes the need to store episodes or sample sequences.
    """
    def __init__(self, action_size, buffer_size, batch_size, device):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        # We add 'hidden' and 'next_hidden' to the experience
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "hidden", "next_hidden"])
        self.device = device

    def add(self, state, action, reward, next_state, done, hidden, next_hidden):
        """
        hidden: tuple (h, c) representing hidden state BEFORE action
        next_hidden: tuple (h, c) representing hidden state AFTER action (for target calculation)
        """
        # Store (h, c) as numpy arrays to save memory/detach from graph
        # Assuming hidden is (h, c) from PyTorch: (num_layers, batch, hidden_size)
        
        # Helper to convert tensor hidden state to numpy
        def to_numpy(h_tuple):
            return (h_tuple[0].detach().cpu().numpy(), h_tuple[1].detach().cpu().numpy())

        e = self.experience(state, action, reward, next_state, done, to_numpy(hidden), to_numpy(next_hidden))
        self.memory.append(e)

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

        return (states, actions, rewards, next_states, dones, hidden, next_hidden)

    def __len__(self):
        return len(self.memory)
    