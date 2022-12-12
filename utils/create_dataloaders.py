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
import matplotlib.pyplot as plt
def visualize_random_image(training_files, label_files, aug_files, aug_labels):
    high_index = len(training_files)
    rand_ind = np.random.randint(high_index)
    mat_file_aug = sio.loadmat(file_name= f"../downloads/data4test/aug_data/{aug_files[rand_ind-1]}")
    label_img_aug = mpimg.imread(f"../downloads/data4test/aug_label/{aug_labels[rand_ind-1]}")
    mat_x_aug = mat_file_aug["vxSample"]
    mat_y_aug = mat_file_aug["vySample"]
    input_image_aug = np.stack((mat_x_aug, mat_y_aug, np.zeros(mat_x_aug.shape)), -1)
    # not-augmented data
    mat_file = sio.loadmat(file_name= f"../downloads/data4test/data/{training_files[rand_ind]}")
    label_img = mpimg.imread(f"../downloads/data4test/label/{label_files[rand_ind]}")
    mat_x = mat_file["vxSample"]
    mat_y = mat_file["vySample"]
    input_image = np.stack((mat_x, mat_y, np.zeros(mat_x.shape)), -1)
    print(f"data:{training_files[rand_ind]}")
    print(f"label:{label_files[rand_ind]}")
    print(f"aug data:{aug_files[rand_ind]}")
    print(f"aug labels:{aug_labels[rand_ind]}")
    fig = plt.figure(figsize=(15, 10))
    rows = 2
    columns = 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(input_image)
    plt.axis(False)
    plt.title("Data")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(label_img)
    plt.axis(False)
    plt.title("Label")
    fig.add_subplot(rows, columns, 3)
    plt.imshow(input_image_aug)
    plt.title("Aug Data")
    plt.axis(False)
    fig.add_subplot(rows, columns, 4)
    plt.title("Aug Label")
    plt.imshow(label_img_aug)