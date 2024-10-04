from torch import nn, optim
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CharBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(CharBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # LSTM forward pass
        out, hidden = self.lstm(x, hidden)

        # Output layer
        out = self.fc(out)
        return out, hidden

    def predict(self):
        self.eval()

