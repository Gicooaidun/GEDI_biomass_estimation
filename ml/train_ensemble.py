import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import *
import wandb
import pickle
import h5py
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split

# ###################################################
class Args:
    def __init__(self):
        self.latlon = True
        self.bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        self.bm = False #change from True to False to exclude the icesat data
        self.patch_size = [15,15]
        self.norm_strat = 'pct'
        self.norm = False
        self.epochs = 1
        self.lr = 0.001
        self.optimizer = optim.Adam
        self.name = 'train_ensemble'
        self.train = True
        self.patience = 6

args = Args()
# Initialize an empty dictionary to store the data
data = {'train': [], 'val': [], 'test': []} 
path_h5 = 'dataset'
fnames = ['data_0-5.h5', 'data_1-5.h5', 'data_2-5.h5', 'data_3-5.h5', 'data_4-5.h5']


all_tiles = []
# Iterate over all the h5 files
for fname in os.listdir(path_h5):
    if fname.endswith('.h5'):
        with h5py.File(os.path.join(path_h5, fname), 'r') as f:
            # Get the list of all tiles in the file
            all_tiles.extend(list(f.keys()))

train_tiles, test_and_val_tiles = train_test_split(all_tiles, test_size=0.35, random_state=42)
val_tile, test_tile = train_test_split(test_and_val_tiles, test_size=0.6, random_state=42)
data['val'].extend(val_tile)
data['test'].extend(test_tile)
data['train'].extend(train_tiles)

with open('dataset/mapping.pkl', 'wb') as f:
    pickle.dump(data, f)

###################################################

class RMSE(nn.Module):
    """ 
        Weighted RMSE.
    """

    def __init__(self):
        super(RMSE, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='none')
        
    def __call__(self, prediction, target, weights = 1):
        # prediction = prediction[:, 0]
        return torch.sqrt(torch.mean(weights * self.mse(prediction,target)))

###################################################

class SimpleFCN(nn.Module):
    def __init__(self,
                 in_features=16,
                 channel_dims = (16, 32, 64, 128, 64, 32, 16),
                 num_outputs=1,
                 kernel_size=3,
                 stride=1):
        """
        A simple fully convolutional neural network.
        """
        if(args.bm):
            in_features = 18
        super(SimpleFCN, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        layers = list()
        for i in range(len(channel_dims)):
            in_channels = in_features if i == 0 else channel_dims[i-1]
            layers.append(nn.Conv2d(in_channels=in_channels, 
                                    out_channels=channel_dims[i], 
                                    kernel_size=kernel_size, stride=stride, padding=1, padding_mode='reflect'))
            layers.append(nn.BatchNorm2d(num_features=channel_dims[i]))
            layers.append(self.relu)
        # print(layers)
        self.conv_layers = nn.Sequential(*layers)
        
        self.conv_output = nn.Conv2d(in_channels=channel_dims[-1], out_channels=num_outputs, kernel_size=1,
                                     stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.conv_output(x)

        return x
    
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models

    def forward(self, x):
        if torch.cuda.is_available():
            self.models = [model.cuda() for model in self.models]
        outputs = [model(x) for model in self.models]
        mean = torch.mean(torch.stack(outputs), dim=0)
        std = torch.std(torch.stack(outputs), dim=0)
        return mean, std
    
###################################################

def train(model, epochs = 10, modelname = 'overwritten_ensemble', patience = 5):
    wandb.init(name=modelname)
    wandb.watch(model, log_freq=100)

    if torch.cuda.is_available():
        model = model.cuda()
    # Define loss function and optimizer
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = args.optimizer(model.parameters(), lr=args.lr)

    mode = 'train'
    ds_training = GEDIDataset({'h5': path_h5, 'norm': 'dataset', 'map': 'dataset'}, fnames = fnames, chunk_size = 1, mode = mode, args = args)
    trainloader = DataLoader(dataset = ds_training, batch_size = 512, shuffle = True, num_workers = 8)
    mode = 'val'
    ds_validation = GEDIDataset({'h5': path_h5, 'norm': 'dataset', 'map': 'dataset'}, fnames = fnames, chunk_size = 1, mode = mode, args = args)
    validloader = DataLoader(dataset = ds_validation, batch_size = 512, shuffle = False, num_workers = 8)

    min_valid_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    # Training loop
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        i=0
        for inputs, targets in trainloader:
            i+=1
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = RMSE()(outputs[:,:,7,7].squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if i%20==0:
                # print(f'Epoch {epoch+1} \t Batch {i} \t Training Loss: {train_loss / i}')
                wandb.log({'train_loss': train_loss / i})

        
        valid_loss = 0.0
        i=0
        model.eval()
        for inputs, targets in validloader:
            i+=1
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model(inputs)
            loss = RMSE()(outputs[:,:,7,7].squeeze(), targets)
            valid_loss += loss.item()
            if i%20==0:
                # print(f'Epoch {epoch+1} \t Batch {i} \t Validation Loss: {valid_loss / i}')
                wandb.log({'valid_loss': valid_loss / i})
    
        print(f'Epoch {epoch+1} Training Loss: {train_loss / len(trainloader)} Validation Loss: {valid_loss / len(validloader)}')
        
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss}--->{valid_loss}) Saving The Model')
            min_valid_loss = valid_loss
            epochs_no_improve = 0
            # Saving State Dict
            torch.save(model.state_dict(), f'models/{modelname}.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('Early stopping!')
                early_stop = True
                break

        if early_stop:
            print("Stopped")
            break

        # print(f"Epoch {epoch+1} completed")

def test(model_architecture, model_path = 'models/modelname.pth', ensemble_models = []):
    if model_architecture == 'SimpleFCN':
        model = SimpleFCN()
        model.load_state_dict(torch.load(model_path))
    elif model_architecture == 'EnsembleModel':
        models = []
        for path in ensemble_models:
            temp_model = SimpleFCN()
            temp_model.load_state_dict(torch.load(path))
            temp_model.eval()
            models.append(temp_model)
        model = EnsembleModel(models)
    else:
        print('Model architecture not found')
        return

    if torch.cuda.is_available():
        # print('Using GPU')
        model = model.cuda()

    mode = 'test'
    ds_testing = GEDIDataset({'h5':path_h5, 'norm': 'dataset', 'map': 'dataset'}, fnames = fnames, chunk_size = 1, mode = mode, args = args)
    testloader = DataLoader(dataset = ds_testing, batch_size = 256, shuffle = False, num_workers = 8)

    # Testing loop
    wandb.watch(model, log_freq=100)
    model.eval()
    test_loss = 0.0
    batch_test_loss = 0.0
    i = 0
    for inputs, targets in testloader:
        i += 1
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs, std = model(inputs)
        loss = RMSE()(outputs[:,:,7,7].squeeze(), targets)
        test_loss += loss.item()
        batch_test_loss += loss.item()
        if i % 20 == 0:
            # print(f'Batch {i} \t Testing Loss: {test_loss / i}')
            batch_test_loss = 0.0
            wandb.log({'test_loss': test_loss / i})

    print(f'Testing Loss: {test_loss / len(testloader)}')

###################################################

models = []
torch.manual_seed(42)
np.random.seed(42)
model0 = SimpleFCN()
models.append(model0)
torch.manual_seed(42+1)
np.random.seed(42+1)
model1 = SimpleFCN()
models.append(model1)
torch.manual_seed(42+2)
np.random.seed(42+2)
model2 = SimpleFCN()
models.append(model2)
torch.manual_seed(42+3)
np.random.seed(42+3)
model3 = SimpleFCN()
models.append(model3)
torch.manual_seed(42+4)
np.random.seed(42+4)
model4 = SimpleFCN()
models.append(model4)

if args.train:
    i=0
    for model in models:
        train(model, epochs = args.epochs, modelname = f'{args.name}_sub_ensemble_model_{i}', patience = args.patience)
        model.eval()
        i+=1

    model = EnsembleModel(models)
    torch.save(model.state_dict(), f'models/{args.name}_ensemble.pth')

ensemble_models = [f'models/{args.name}_sub_ensemble_model_{i}.pth' for i in range(len(models))]

test(model_architecture='EnsembleModel', model_path= f'models/{args.name}_ensemble.pth', ensemble_models = ensemble_models)

wandb.finish()