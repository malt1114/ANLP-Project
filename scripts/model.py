from torch import nn, optim
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")

class CharBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, max_length=None, batch_size=None):
        super(CharBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_length = max_length
        self.batch_size = batch_size
        self.output_size = output_size
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)  # output_size should be vocab_size

    def forward(self, inputs, train=True):
        lstm_out, _ = self.lstm(inputs)  # lstm_out has shape (batch_size, seq_len, hidden_size * 2)
        out = self.fc(lstm_out)  # Output shape should be (batch_size, seq_len, output_size)
        if train:
            return out.view(-1, self.output_size)  # Logits for training
        else:
            log_probs = torch.nn.functional.softmax(out, dim=-1)  # Softmax over the vocabulary dimension
            return log_probs.view(-1, self.output_size)

    def predict(self, inputs):
        with torch.no_grad():
            y = self.forward(inputs, train=False)  # (batch_size, seq_len, output_size)
            predictions = torch.argmax(y, dim=-1)  # Take the argmax across the vocab dimension
            return predictions




