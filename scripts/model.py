from torch import nn, optim
from scripts.data import create_data_loader
import pandas as pd
import torch

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

def prepare_data(complexity_level:str, max_length: int, batch_size: int):
    train = pd.read_csv(f"data/processed/{complexity_level.lower()}/train_{complexity_level.lower()}.csv")
    validation = pd.read_csv(f"data/processed/{complexity_level.lower()}/validation_{complexity_level.lower()}.csv")
    test = pd.read_csv(f"data/processed/{complexity_level.lower()}/test_{complexity_level.lower()}.csv")
    
    train_loader = create_data_loader(train, complexity=complexity_level, max_length=max_length, batch_size=batch_size)
    validation_loader = create_data_loader(validation, complexity=complexity_level, max_length=max_length, batch_size=batch_size) 
    test_loader = create_data_loader(test, complexity=complexity_level, max_length=max_length, batch_size=batch_size)

    return train_loader, validation_loader, test_loader

def train_model(complexity_level, model, epochs, train_loader, validation_loader, max_length, loss_function, optimizer):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    BEST_VAL_LOSS = float("inf")
    LAST_SAVED = 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0    
        for batch in train_loader:
            typo_batch, sentence_batch = batch  
            
            typo_batch = typo_batch.to(device)
            sentence_batch =sentence_batch.to(device)

            sentence_batch = sentence_batch.view(-1)
            typo_batch = typo_batch.reshape(-1, max_length, 1)

            y = model.forward(typo_batch, train=False)  
            loss = loss_function(y, sentence_batch)  
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        epoch_loss_avg = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():  
            for val_batch in validation_loader:
                typo_val_batch, sentence_val_batch = val_batch
                
                typo_val_batch = typo_val_batch.to(device)
                sentence_val_batch = sentence_val_batch.to(device)
                
                sentence_val_batch = sentence_val_batch.view(-1)
                typo_val_batch = typo_val_batch.reshape(-1, max_length, 1)
                
                val_y = model.forward(typo_val_batch, train=False)
                
                val_loss_batch = loss_function(val_y, sentence_val_batch)
                val_loss += val_loss_batch.item()
                    
        val_loss_avg = val_loss / len(validation_loader)
        scheduler.step()

        #Save if loss is the lowest yet
        if val_loss_avg < BEST_VAL_LOSS:
            BEST_VAL_LOSS = val_loss_avg
            model_path = f'models/{complexity_level.lower()}/model_{epoch + 1}.pt'
            torch.save(model.state_dict(), model_path)
            LAST_SAVED = 0
        else:
            LAST_SAVED += 1
        
        #Save if have not been save for 20 epochs
        if LAST_SAVED - 20 == 0:
            model_path = f'models/{complexity_level}/model_{epoch + 1}.pt'
            torch.save(model.state_dict(), model_path)
            LAST_SAVED = 0
        
        with open(f'models/{complexity_level.lower()}/stats.csv', 'a') as f:
            f.write(f'{epoch + 1},{epoch_loss_avg:.8f},{val_loss_avg:.8f}\n')
        
        print(f"Epoch {epoch + 1}/{epochs} Train Loss: {epoch_loss_avg:.8f} Val Loss: {val_loss_avg:.8f}", flush= True)