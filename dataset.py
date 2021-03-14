import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob
from skimage import io

class SuperResolutionDataset(Dataset):
    ''' Dataloader to load HR and LR Images '''
    def __init__(self, hr_path, lr_path, transform=None):
        '''
            Args:
                hr_path: str. High Resolution Images path
                lr_path: str. Low Resoultion Images path
        '''
        self.hr_path = hr_path
        self.lr_path = lr_path
        self.transform = transform
        self.hr_dims = 256
        self.lr_dims = 128
        
        self.hr_images = sorted(glob(self.hr_path + '/*.png'))
        self.lr_images = sorted(glob(self.lr_path + '/*.png'))

    def __len__(self):
        total_hr_images = len(glob(self.hr_path + '/*.png'))
        total_lr_images = len(glob(self.lr_path + '/*.png'))
        assert total_hr_images == total_lr_images, 'Total number mismatch in hr_path:{} and lr_path:{}'.format(total_hr_images,total_lr_images)
        return total_hr_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        hr_image = io.imread(self.hr_images[idx])
        lr_image = io.imread(self.lr_images[idx])

        w, h, _ = hr_image.shape
        rand_point_x = np.random.randint(2, w//4)
        rand_point_y = np.random.randint(2, h//4)

        hr_image = hr_image[rand_point_y:rand_point_y + self.hr_dims, rand_point_x:rand_point_x + self.hr_dims]
        lr_image = lr_image[rand_point_y//2:rand_point_y//2 + self.lr_dims, rand_point_x//2:rand_point_x//2 + self.lr_dims]

        # if self.transform:
        hr_image = self.transform(hr_image)
        # hr_image = hr_image.permute(1,2,0)

        lr_image = self.transform(lr_image)
        # lr_image = lr_image.permute(1,2,0)
        
        # (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
        # hr_image = (hr_image - torch.min(hr_image)) / (torch.max(hr_image) - torch.min(hr_image))
        # lr_image = (lr_image - torch.min(lr_image)) / (torch.max(lr_image) - torch.min(lr_image))

        while hr_image.shape[1:] != (256,256) or lr_image.shape[1:] != (128,128):
            hr_image = io.imread(self.hr_images[idx])
            lr_image = io.imread(self.lr_images[idx])

            w, h, _ = hr_image.shape
            rand_point_x = np.random.randint(2, w//4)
            rand_point_y = np.random.randint(2, h//4)

            hr_image = hr_image[rand_point_y:rand_point_y + self.hr_dims, rand_point_x:rand_point_x + self.hr_dims]
            lr_image = lr_image[rand_point_y//2:rand_point_y//2 + self.lr_dims, rand_point_x//2:rand_point_x//2 + self.lr_dims]

            # if self.transform:
            hr_image = self.transform(hr_image)
            # hr_image = hr_image.permute(1,2,0)

            lr_image = self.transform(lr_image)
            # lr_image = lr_image.permute(1,2,0)
            
            # (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
            # hr_image = (hr_image - torch.min(hr_image)) / (torch.max(hr_image) - torch.min(hr_image))
            # lr_image = (lr_image - torch.min(lr_image)) / (torch.max(lr_image) - torch.min(lr_image))
        
        return hr_image, lr_image

if __name__ == '__main__':

    main_path = "/content/drive/My Drive/DIV25Dataset/Dataset"
    hr_train_path = os.path.join(main_path, "DIV2K_train_HR")
    hr_valid_path = os.path.join(main_path, "DIV2K_valid_HR")
    lr_train_path = os.path.join(main_path, "DIV2K_train_LR_bicubic/X2")
    lr_valid_path = os.path.join(main_path, "DIV2K_valid_LR_bicubic/X2")

    train_dataset = SuperResolutionDataset(hr_path=hr_train_path, lr_path=lr_train_path, transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

    val_dataset = SuperResolutionDataset(hr_path=hr_valid_path, lr_path=lr_valid_path, transform=transforms.ToTensor())
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f'Train Dataset Size:{len(train_dataset)}. Train Loader Size:{len(train_loader)}.\nValid Dataset Size:{len(valid_dataset)}. Valid Loader Size:{len(valid_loader)}.')