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
    def __init__(self, input_dims, n_actions, device, fc1_dims=128, fc2_dims=64):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.device = device
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.apply(init_weights)
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        
        return actions
    
    def init_hidden(self, *args, **kwargs):
        """
        dummy function
        """
        return None


class RecurrentDeepQNetwork(nn.Module):
    """
    Sequential LSTM based DQN.
    """
    def __init__(self, input_dims, n_actions, device, fc1_dims=128, hidden_size=64, num_layers=1):
        super(RecurrentDeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        # 1. Feature Extractor
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        
        # 2. LSTM Layer
        # batch_first=True expects input: (Batch, Seq, Feature)
        self.lstm = nn.LSTM(input_size=self.fc1_dims, 
                            hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, 
                            batch_first=True)
        
        # 3. Output Layer
        self.fc_out = nn.Linear(self.hidden_size, self.n_actions)
        
        self.apply(init_weights)
        self.to(self.device)

    def forward(self, state, hidden=None):
        """
        state: Tensor of shape (Batch, Input_Dims)
        hidden: Tuple (h_0, c_0) or None
        """
        # 1. Feature Extraction
        x = F.relu(self.fc1(state))
        
        # 2. Reshape for LSTM
        # LSTM expects a sequence dimension. Since we are doing 1 step at a time:
        # Reshape (Batch, Features) -> (Batch, 1, Features)
        x = x.unsqueeze(1)
        
        # 3. LSTM Pass
        # If hidden is None, LSTM defaults to zeros
        lstm_out, new_hidden = self.lstm(x, hidden)
        
        # 4. Output Logic
        # lstm_out shape is (Batch, 1, Hidden_Size). We need to remove the sequence dim.
        x = lstm_out.squeeze(1)
        
        actions = self.fc_out(x)
        
        return actions, new_hidden

    def init_hidden(self, batch_size):
        """
        Helper to initialize a fresh hidden state (h0, c0).
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return (h0, c0)
    