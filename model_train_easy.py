from scripts.model import prepare_data, CharBiLSTM, train_model
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device, flush = True)
print(torch.cuda.is_available())

#Hyperparameters
complexity_level = 'Easy'
max_length = 151
batch_size = 2**9

input_size = 1  # Uses the raw index in alphabet as input
hidden_size = 2**9 # Maps to hidden size
output_size = 28 +  1  # Outputs probabilities for each character in vocabulary + padding
num_layers = 1
epochs = 1000

model = CharBiLSTM(input_size, hidden_size, output_size, num_layers, max_length, batch_size).to(device)
# Padding is value -1, therefore we want to ignore it in our loss function
loss_function = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if __name__ == "__main__":

    #Create stat file
    with open(f'models/{complexity_level.lower()}/stats.csv', 'a') as f:
        f.write(f'Epoch,Train_loss,Val_loss\n')

    train_loader, validation_loader, test_loader = prepare_data(complexity_level = complexity_level,
                                                                max_length = max_length,
                                                                batch_size = batch_size)

    print(f'Training of {complexity_level} started', flush= True)
    train_model(complexity_level,
                model,
                epochs,
                train_loader,
                validation_loader,
                max_length,
                loss_function,
                optimizer)