import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # Output Layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # Initial hidden state
        out, _ = self.rnn(x, h0)  # Forward pass through RNN
        out = self.fc(out[:, -1, :])  # Taking the output of the last time step
        return out

# Example usage
input_size = 10  # Number of features
hidden_size = 20  # Number of hidden neurons
output_size = 2  # Binary classification

model = SimpleRNN(input_size, hidden_size, output_size)
input_data = torch.randn(32, 5, input_size)  # Batch of 32 sequences, each with 5 time steps
output = model(input_data)
print(output.shape)  # Expected output shape: (32, 2)