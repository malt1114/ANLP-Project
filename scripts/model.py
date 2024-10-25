from torch import nn, optim
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CharBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(CharBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, inputs, train=True):
        lstm_out, state = self.lstm(inputs)
        out = self.fc(lstm_out)
        # print("out", out)
        # y = out[:, -1, :]

        if train:
            return out
        else:
            log_probs = torch.nn.functional.softmax(out, dim=1)
            return log_probs

    def predict(self, inputs):
        with torch.no_grad():
            y = self.forward(inputs, train=False)
            outputs = []
            for output in y:
                outputs.append(output.tolist().index(max(output.tolist())))
            return torch.LongTensor(outputs)




