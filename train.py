import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from datetime import datetime

from dataset import SuperResolutionDataset
from maanet import MAANet

import config as cfg

def validation_step(model, loss):
    v_loss = 0
    run = 0
    model.eval()
    for i, (hr, lr) in enumerate(val_dataloader):
        with torch.no_grad():
            outs = model(lr.to(device))
            l = loss(outs, hr.to(device))
            v_loss += l.item()
            run += 1
    return v_loss / run

def train():
    train_dataset = SuperResolutionDataset(hr_path=cfg.hr_train_path, lr_path=cfg.lr_train_path, transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

    val_dataset = SuperResolutionDataset(hr_path=cfg.hr_valid_path, lr_path=cfg.lr_valid_path, transform=transforms.ToTensor())
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    #https://discuss.pytorch.org/t/using-variable-sized-input-is-padding-required/18131/7

    print(f'Train Loader Size:{len(train_dataloader)}.\nValidation Loader Size:{len(val_dataloader)}.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_epochs = 1e6
    model = MAANet().to(device)
    # model = torch.load(cfg.best_model_path + cfg.best_model_name).to(device)
    loss = nn.L1Loss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=2e-4)

    # Training Process
    min_loss = 9999.99
    print('Training Started...............')
    for e in range(n_epochs):
        e_loss = 0
        run = 0
        model.train()
        start = datetime.now()
        for i, (hr, lr) in enumerate(train_dataloader):
            # print(f'\tWorking on {i+1}/{len(train_dataloader)}')
            hr = hr.to(device)
            lr = lr.to(device)

            optimizer.zero_grad()
            outs = model(lr)
            l = loss(outs, hr)
            
            e_loss += l.item()
            run += 1

            l.backward()
            optimizer.step()
        val_loss = validation_step(model, loss)
        end = datetime.now()
        print(f"Epoch:{e+1}/{n_epochs}, Time Taken:{end-start}, Train-Loss:{e_loss/run:.4f}, Val-Loss:{val_loss:.4f}")
        
        if val_loss < min_loss:
            min_loss = val_loss
            if not os.path.exists(cfg.best_model_path):
                os.mkdir(cfg.best_model_path)
            torch.save(model, cfg.best_model_path + cfg.best_model_name)
            print(f'Model Saved at Epoch:{e+1}')

if __name__ == '__main__':
    train()