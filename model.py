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
    def __init__(
        self, 
        input_dims: int, 
        n_actions: int, 
        n_envs: int, 
        n_reward_steps: int, 
        device, 
        network_hidden_layers=(256, 128, 64), # Slightly wider is better for MTL
    ):
        super().__init__()
        self.device = device

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.n_envs = n_envs
        self.n_reward_steps = n_reward_steps

        layers = []
        in_dim = input_dims
        for hidden_dim in network_hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.encoding_dim = network_hidden_layers[-1]

        self.q_head = nn.Sequential(
            nn.Linear(self.encoding_dim, self.encoding_dim),
            nn.GELU(),
            nn.Linear(self.encoding_dim, n_actions)
        )
        
        # Classifier Tower
        self.env_classifier = nn.Sequential(
            nn.Linear(self.encoding_dim, self.encoding_dim),
            nn.GELU(),
            nn.Linear(self.encoding_dim, n_envs)
        )
        
        reward_input_dim = self.encoding_dim + (n_reward_steps * n_actions)
        # Reward Predictor Tower
        self.reward_predictor = nn.Sequential(
            nn.Linear(reward_input_dim, self.encoding_dim),
            nn.GELU(),
            nn.Linear(self.encoding_dim, n_reward_steps)
        )

        self.apply(init_weights)
        self.to(self.device)

    def forward(self, state, actions):
        x = state
        encoding = self.feature_extractor(x)
        q_values = self.q_head(encoding)
        env_logits = self.env_classifier(encoding)

        ohe_actions = F.one_hot(actions, num_classes=self.n_actions).float()
        act_flat = ohe_actions.view(ohe_actions.size(0), -1)
        combined_input = torch.cat([encoding, act_flat], dim=1)
        predicted_rewards = self.reward_predictor(combined_input)
        return (q_values, env_logits, predicted_rewards, encoding)
    
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
    def __init__(
        self, 
        input_dims: int, 
        n_actions: int, 
        n_envs: int,            # New: for classifier
        n_reward_steps: int,    # New: for reward prediction
        device, 
        network_hidden_layers=(256, 128, 64), 
        num_layers=1
    ):
        super().__init__()
        self.device = device

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.n_envs = n_envs
        self.n_reward_steps = n_reward_steps
        
        self.num_layers = num_layers
        if not isinstance(network_hidden_layers, (list, tuple)) or len(network_hidden_layers) < 2:
            raise ValueError("network_hidden_layers must be len >= 2 (e.g., [enc_dim, lstm_dim, tower_dim...])")

        # --- Parse config list without adding any new config fields ---
        h = list(network_hidden_layers)
        encoder_dim = int(h[0])
        lstm_hidden_dim = int(h[1])
        tower_hidden_dims = [int(x) for x in h[2:]] # Can be empty if direct readout is desired

        self.hidden_size = lstm_hidden_dim

        # 1) Encoder (single layer as per "one encoder")
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dims, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.GELU()
        )

        # 2) LSTM core
        self.lstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        
        def build_tower(output_dim, in_dim):
            layers = []
            
            # Add config-driven hidden layers
            for d in tower_hidden_dims:
                layers.append(nn.Linear(in_dim, d))
                layers.append(nn.GELU())
                in_dim = d
            
            # Final projection
            layers.append(nn.Linear(in_dim, output_dim))
            return nn.Sequential(*layers)
        
        self.q_head = build_tower(n_actions, self.hidden_size)
        self.env_classifier = build_tower(n_envs, self.hidden_size)
        reward_input_dim = self.hidden_size + (n_reward_steps * n_actions)
        self.reward_predictor = build_tower(n_reward_steps, reward_input_dim)

        self.apply(init_weights)
        self.to(self.device)

    def forward(self, state, actions, hidden=None):
        # state: (B, input_dims)
        # x: (B, encoder_dim)
        x = self.encoder(state)
        x = x.unsqueeze(1) # (B, 1, encoder_dim)
        lstm_out, new_hidden = self.lstm(x, hidden)
        core_features = lstm_out.squeeze(1)

        q_values = self.q_head(core_features)
        env_logits = self.env_classifier(core_features)

        ohe_actions = F.one_hot(actions, num_classes=self.n_actions).float()
        act_flat = ohe_actions.view(ohe_actions.size(0), -1)
        combined_input = torch.cat([core_features, act_flat], dim=1)
        predicted_rewards = self.reward_predictor(combined_input)
        
        return (q_values, env_logits, predicted_rewards, core_features), new_hidden

    def init_hidden(self, batch_size):
        """
        Helper to initialize a fresh hidden state (h0, c0).
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return (h0, c0)
    