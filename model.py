import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    """
    Applies Orthogonal Initialization to Linear layers.
    Helps prevents vanishing/exploding gradients in Deep/Recurrent networks.
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    # Optional: Initialize LSTM weights if needed
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)


class DeepQNetwork(nn.Module):
    """
    Vanilla MLP (Feed-Forward) DQN.
    """
    def __init__(self, input_dims: int, n_actions: int, device, network_hidden_layers=(128, 64)):
        super().__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.device = device

        dims = [input_dims] + list(network_hidden_layers) + [n_actions]

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        self.apply(init_weights)
        self.to(self.device)

    def forward(self, state):
        x = state
        for layer in self.fcs[:-1]:
            x = F.relu(layer(x))
        q = self.fcs[-1](x)
        return q
    
    def init_hidden(self, *args, **kwargs):
        """
        dummy function
        """
        return None


class RecurrentDeepQNetwork(nn.Module):
    """
    Sequential LSTM based DQN.
    Config-driven structure from network_hidden_layers:
      h0 -> encoder dim
      h1 -> LSTM hidden size
      h2.. -> head MLP dims (optional)
    """
    def __init__(self, input_dims, n_actions, device, network_hidden_layers, num_layers=1):
        super().__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.device = device
        self.num_layers = num_layers

        if not isinstance(network_hidden_layers, (list, tuple)) or len(network_hidden_layers) < 1:
            raise ValueError("network_hidden_layers must be a list/tuple with at least 1 element")

        # --- Parse config list without adding any new config fields ---
        h = list(network_hidden_layers)

        encoder_dim = int(h[0])
        lstm_hidden = int(h[1]) if len(h) >= 2 else int(h[0])   # fallback if only one provided
        head_dims = [int(x) for x in h[2:]]                     # may be empty

        self.hidden_size = lstm_hidden

        # 1) Encoder (single layer as per "one encoder")
        self.encoder = nn.Linear(self.input_dims, encoder_dim)

        # 2) LSTM core
        self.lstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # 3) Head MLP (0..N layers) then output
        dims = [self.hidden_size] + head_dims + [self.n_actions]
        self.head = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

        self.apply(init_weights)
        self.to(self.device)

    def forward(self, state, hidden=None):
        # state: (B, input_dims)
        x = F.relu(self.encoder(state))     # (B, encoder_dim)

        x = x.unsqueeze(1)                  # (B, 1, encoder_dim)
        lstm_out, new_hidden = self.lstm(x, hidden)
        x = lstm_out.squeeze(1)             # (B, hidden_size)

        # head MLP
        for layer in self.head[:-1]:
            x = F.relu(layer(x))
        q = self.head[-1](x)                # (B, n_actions)

        return q, new_hidden

    def init_hidden(self, batch_size):
        """
        Helper to initialize a fresh hidden state (h0, c0).
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return (h0, c0)
    