
import torch
import torch.nn as nn
class AttentionModel(nn.Module): # Self-attention model
    def __init__(self, input_dim, hidden_dim, output_dim, max_len):
        super(AttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.mlp_begin = nn.Linear(input_dim, hidden_dim)
        self.max_len = max_len
        self.position_embeddings = nn.Embedding(max_len, hidden_dim) # Positional encoding for each time step
        self.self_attn = SelfAttention_Layer(hidden_dim, hidden_dim)
        self.mlp1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.mlp2 = nn.Linear(hidden_dim//2, output_dim)

    def forward(self, input):
        position_ids = torch.arange(self.max_len, dtype=torch.long, device=input.device)
        input = self.mlp_begin(input) + self.position_embeddings(position_ids)
        attn_output = self.self_attn(input)
        hidden_out = torch.relu(self.mlp1((attn_output)))
        hidden_out = (self.mlp2(hidden_out))
        return hidden_out


class SelfAttention_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention_Layer, self).__init__()
        self.input_dim = input_dim
        self.query_transform = nn.Linear(input_dim, hidden_dim) 
        self.key_transform = nn.Linear(input_dim, hidden_dim) 
        self.value_transform = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, input): 
        query_matrix = self.query_transform(input)
        key_matrix = self.key_transform(input)
        value_matrix = self.value_transform(input)
        score = torch.bmm(query_matrix, key_matrix.transpose(1, 2))/(self.input_dim**0.5) # Caculate the attention score
        softmax_score = self.softmax(score) # Smooth the score
        weighted = torch.bmm(softmax_score, value_matrix) # Weighted average all time steps by score
        return weighted



# Define the LSTM model class, inheriting from nn.Module
class LSTMModel(nn.Module):
    # The __init__ method initializes the model with input size, hidden size, and output size
    def __init__(self, input_size, hidden_size, output_size):
        # Call the super class constructor
        super(LSTMModel, self).__init__()
        
        # Store the hidden layer size
        self.hidden_size = hidden_size
        
        # Define the LSTM layer
        # input_size: The number of expected features in the input (e.g., sequence length)
        # hidden_size: The number of features in the hidden state
        # batch_first=True ensures that input tensors are structured as (batch_size, seq_len, feature_size)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Define the first fully connected layer
        # It reduces the dimensionality from hidden_size to half (hidden_size // 2)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        
        # Define the second fully connected layer
        # It reduces the dimensionality further to the output size
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        # Pass the input sequence through the LSTM layer
        # lstm_out contains the output from the LSTM for each time step
        lstm_out, _ = self.lstm(x)
        
        # We are interested only in the output of the last time step
        lstm_out = lstm_out
        
        # Pass the output through the first fully connected layer (fc1) and apply ReLU activation
        out = torch.relu(self.fc1(lstm_out))
        
        # Pass the output through the second fully connected layer (fc2) and apply Sigmoid activation
        # Sigmoid squashes the output between 0 and 1, which is useful for binary classification
        out = (self.fc2(out))
        
        # Return the final output
        return out
