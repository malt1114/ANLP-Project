from scripts.model import prepare_data, CharBiLSTM, train_model
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device, flush = True)
#Hyperparameters
complexity_level = 'Hard'
max_length = 150
batch_size = 2**10

input_size = 1  # Uses the raw index in alphabet as input
output_size = 27 +  1  # Outputs probabilities for each character in vocabulary + padding
num_layers = 1
epochs = 100

#Create stat file
with open(f'models/{complexity_level.lower()}/stats.csv', 'a') as f:
    f.write(f'Epoch,Train_loss,Val_loss\n')


train_loader, validation_loader, test_loader = prepare_data(complexity_level = complexity_level,
                                                            max_length = max_length,
                                                            batch_size = batch_size)


model = CharBiLSTM(input_size, hidden_size, output_size, num_layers, max_length, batch_size).to(device)
# Padding is value -1, therefore we want to ignore it in our loss function
loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum = 0.9)
print('Training started', flush= True)
train_model(complexity_level,
            model, 
            epochs, 
            train_loader, 
            validation_loader, 
            max_length, 
            loss_function, 
            optimizer)