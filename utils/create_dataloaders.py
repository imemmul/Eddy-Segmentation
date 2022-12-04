import os
from xmlrpc.client import Boolean
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import matplotlib.image as mpimg

NUM_WORKERS = os.cpu_count()

class EddyDatasetTrain(Dataset):
    def __init__(self, input_image_dir, mask_image_dir, feature_extractor, split) -> None:
        self.input_image_dir = input_image_dir
        self.mask_image_dir = mask_image_dir
        self.feature_extractor = feature_extractor

        image_file_names = [f for f in os.listdir(self.input_image_dir)]
        mask_file_names = [f for f in os.listdir(self.mask_image_dir)]
        self.images = (sorted(image_file_names))[0:split]
        self.masks = (sorted(mask_file_names))[0:split]
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_file = sio.loadmat(self.input_image_dir + self.images[index])
        img_x = img_file["vxSample"]
        img_y = img_file["vySample"]
        input_img = np.stack((img_x, img_y, np.zeros(img_x.shape)), -1)
        mask_img = label_img = mpimg.imread(self.mask_image_dir + self.masks[index])
        encoded_inputs = self.feature_extractor(input_img, mask_img, return_tensors="pt")

        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()
        
        return encoded_inputs

class EddyDatasetValid(Dataset):
    def __init__(self, input_image_dir, mask_image_dir, feature_extractor, split) -> None:
        self.input_image_dir = input_image_dir
        self.mask_image_dir = mask_image_dir
        self.feature_extractor = feature_extractor

        image_file_names = [f for f in os.listdir(self.input_image_dir)]
        mask_file_names = [f for f in os.listdir(self.mask_image_dir)]
        self.images = (sorted(image_file_names))[split:]
        self.masks = (sorted(mask_file_names))[split:]
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_file = sio.loadmat(self.input_image_dir + self.images[index])
        img_x = img_file["vxSample"]
        img_y = img_file["vySample"]
        input_img = np.stack((img_x, img_y, np.zeros(img_x.shape)), -1)
        mask_img = label_img = mpimg.imread(self.mask_image_dir + self.masks[index])
        encoded_inputs = self.feature_extractor(input_img, mask_img, return_tensors="pt")

        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()
        
        return encoded_inputs



def create_dataloaders(
    train_dir: str=None, 
    test_dir: str=None,
    test_data=None,
    train_data=None,
    data_folder_imported: Boolean=False,
    transform=None, 
    batch_size: int=32, 
    num_workers: int=NUM_WORKERS):

  train_dataloader, test_dataloader = 0, 0
  train_data, test_data = train_data, test_data
  # Use ImageFolder to create dataset(s)
  if data_folder_imported:
    train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True
    )
    test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True
    )
  else:
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
    )
    test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
    )

  # Get class names

  return train_dataloader, test_dataloader, class_names