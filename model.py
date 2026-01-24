import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    """
    Applies Orthogonal Initialization to Linear layers.
    Helps prevents vanishing/exploding gradients in Deep/Recurrent networks.
    """
    # 1) Linear Layers
    if isinstance(m, nn.Linear):
        # Using gain for ReLU/GELU (approx sqrt(2))
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    # 2) LayerNorm Layers
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

    # 3) LSTM Layers
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                # Orthogonal is critical for the hidden-to-hidden weights 
                # to prevent vanishing gradients over long sequences
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)
                # Forget gate bias trick: setting it to 1.0 can help memory 
                # flow better at the start of training.
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)


class DeepQNetwork(nn.Module):
    def __init__(self, input_dims: int, n_actions: int, device, network_hidden_layers=(128, 64)):
        super().__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.device = device

        dims = [input_dims] + list(network_hidden_layers)
        
        # Build Layers and Norms
        self.fcs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            self.fcs.append(nn.Linear(dims[i], dims[i+1]))
            self.norms.append(nn.LayerNorm(dims[i+1]))

        # Final Output Layer (No Norm/GELU on the final Q-values)
        self.output = nn.Linear(dims[-1], n_actions)

        self.apply(init_weights)
        self.to(self.device)

    def forward(self, state):
        x = state
        for layer, norm in zip(self.fcs, self.norms):
            x = layer(x)
            x = norm(x)
            x = F.gelu(x)  # Switched to GELU
        
        q = self.output(x)
        return q
    
    def init_hidden(self, *args, **kwargs):
        """
        dummy function
        """
        return None
    
class DuelingDeepQNetwork(nn.Module):
    """
    Dueling MLP DQN:
      shared MLP trunk -> (Value head, Advantage head) -> Q(s,a)
    """
    def __init__(self, input_dims: int, n_actions: int, device, network_hidden_layers=(128, 64)):
        super().__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.device = device

        # Shared trunk: input -> hidden layers
        dims = [input_dims] + list(network_hidden_layers)
        self.trunk = nn.ModuleList()
        self.trunk_norms = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.trunk.append(nn.Linear(dims[i], dims[i+1]))
            self.trunk_norms.append(nn.LayerNorm(dims[i+1]))

        last_dim = dims[-1]

        # Value stream V(s)
        self.value = nn.Linear(last_dim, 1)

        # Advantage stream A(s,a)
        self.advantage = nn.Linear(last_dim, n_actions)

        self.apply(init_weights)
        self.to(self.device)

    def forward(self, state):
        x = state
        for layer, norm in zip(self.trunk, self.trunk_norms):
            x = layer(x)
            x = norm(x)
            x = F.gelu(x)

        v = self.value(x)
        a = self.advantage(x)

        # Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

    def init_hidden(self, *args, **kwargs):
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
        self.encoder_ln = nn.LayerNorm(encoder_dim)

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
        self.head_lns = nn.ModuleList([nn.LayerNorm(dims[i+1]) for i in range(len(dims) - 2)])

        self.apply(init_weights)
        self.to(self.device)

    def forward(self, state, hidden=None):
        # state: (B, input_dims)
        x = self.encoder(state)     # (B, encoder_dim)
        x = self.encoder_ln(x)
        x = F.gelu(x)

        x = x.unsqueeze(1)                  # (B, 1, encoder_dim)
        lstm_out, new_hidden = self.lstm(x, hidden)
        x = lstm_out.squeeze(1)             # (B, hidden_size)

        # head MLP
        for i, layer in enumerate(self.head[:-1]):
            x = layer(x)
            x = self.head_lns[i](x)
            x = F.relu(x)
        q = self.head[-1](x)                # (B, n_actions)
        return q, new_hidden

    def init_hidden(self, batch_size):
        """
        Helper to initialize a fresh hidden state (h0, c0).
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return (h0, c0)
    