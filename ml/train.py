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
        self.bm = True #change from True to False to exclude the icesat data
        self.patch_size = [15,15]
        self.norm_strat = 'pct'
        self.norm = False
        self.epochs = 500
        self.lr = 0.001
        self.optimizer = optim.Adam
        self.name = 'balanced_huge_icesat'
        self.seed = 24
        

args = Args()
torch.manual_seed(args.seed)
# Initialize an empty dictionary to store the data
data = {'train': [], 'val': [], 'test': []} 
path_h5 = '/dataset'
fnames = ['data_0-5.h5', 'data_1-5.h5', 'data_2-5.h5', 'data_3-5.h5', 'data_4-5.h5']

all_tiles = []
# Iterate over all the h5 files
for fname in os.listdir(path_h5):
    if fname.endswith('.h5'):
        with h5py.File(os.path.join(path_h5, fname), 'r') as f:
            # Get the list of all tiles in the file
            all_tiles.extend(list(f.keys()))

train_tiles, test_and_val_tiles = train_test_split(all_tiles, test_size=0.35, random_state=args.seed)
val_tile, test_tile = train_test_split(test_and_val_tiles, test_size=0.6, random_state=args.seed)
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
                 channel_dims = (16, 32, 64, 128, 256, 512, 1024, 512, 256, 128, 64, 32, 16),
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

###################################################

wandb.init(project='code-ml-better_logging', name=f'{args.name}_{args.epochs}Epochs_LR{args.lr}_{args.optimizer.__name__}_seed{args.seed}')
model = SimpleFCN()
wandb.watch(model, log_freq=100)

if torch.cuda.is_available():
    model = model.cuda()
# Define loss function and optimizer
optimizer = args.optimizer(model.parameters(), lr=args.lr)

mode = 'train'
ds_training = GEDIDataset({'h5':'dataset', 'norm': 'dataset', 'map': 'dataset'}, fnames = fnames, chunk_size = 1, mode = mode, args = args)
trainloader = DataLoader(dataset = ds_training, batch_size = 512, shuffle = True, num_workers = 8)
mode = 'val'
ds_validation = GEDIDataset({'h5':'dataset', 'norm': 'dataset', 'map': 'dataset'}, fnames = fnames, chunk_size = 1, mode = mode, args = args)
validloader = DataLoader(dataset = ds_validation, batch_size = 512, shuffle = False, num_workers = 8)
mode = 'test'
ds_testing = GEDIDataset({'h5':'dataset', 'norm': 'dataset', 'map': 'dataset'}, fnames = fnames, chunk_size = 1, mode = mode, args = args)
testloader = DataLoader(dataset = ds_testing, batch_size = 256, shuffle = False, num_workers = 8)

# 
min_valid_loss = float('inf')

for epoch in range(args.epochs):  # 100 epochs
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
        # print(loss.item())
        if i%20==0:
            # print(f'Epoch {epoch+1} \t Batch {i} \t Training Loss: {train_loss / i}')
            wandb.log({'train_loss': train_loss / i})
    wandb.log({'total_train_loss': train_loss / len(trainloader)})

    #Validation loop
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
    wandb.log({'total_valid_loss': valid_loss / len(validloader)})
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss}--->{valid_loss}) Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), f'models/{args.name}_{args.epochs}Epochs_LR{args.lr}_{args.optimizer.__name__}_seed{args.seed}.pth')

    # Testing loop
    test_loss = 0.0
    i = 0
    for inputs, targets in testloader:
        i += 1
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        loss = RMSE()(outputs[:,:,7,7].squeeze(), targets)
        test_loss += loss.item()
        if i % 20 == 0:
            wandb.log({'test_loss': test_loss / i})
            # print(f'Batch {i} \t Testing Loss: {test_loss / i}')
    wandb.log({'total_test_loss': test_loss / len(testloader)})

    print(f'Epoch {epoch+1} Training Loss: {train_loss / len(trainloader)} Validation Loss: {valid_loss / len(validloader)} Testing Loss: {test_loss / len(testloader)}')

torch.save(model.state_dict(), f'models/{args.name}_{args.epochs}Epochs_LR{args.lr}_{args.optimizer.__name__}_seed{args.seed}_final.pth')
wandb.finish()
