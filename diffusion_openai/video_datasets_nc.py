import os
import torch
from torch.utils.data import DataLoader, Dataset
from netCDF4 import Dataset as NetCDFDataset
import numpy as np
from PIL import Image

def load_data(
    *, data_file, variable_name, batch_size, image_size, class_cond=False, transforms = None, deterministic=False, rgb=False, seq_len=20
):

    if not data_file:
        raise ValueError("unspecified data file")
    
    dataset = NetCDFVideoDataset(
        image_size,
        data_file,
        variable_name,
        transforms=transforms,
        rgb=rgb,
        seq_len=seq_len
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True
        )
    
    return loader

class NetCDFVideoDataset(Dataset):
    def __init__(self, image_size, file_path, variable_name, transforms=None, rgb=False, seq_len=20):
        self.image_size = image_size
        self.file_path = file_path
        self.rgb = rgb
        self.seq_len = seq_len
        self.variable_name = variable_name
        self.transforms = transforms
        self.crop_size = 64
        self.step_size = 10
        self.valid_crops = []
        self._load_and_process_images()
    
    def __len__(self):
        
        return len(self.valid_crops) - self.seq_len + 1
    
    def _load_and_process_images(self):
            
            dataset_chl = NetCDFDataset(self.file_path[0], 'r')
            var_chl = dataset_chl.variables[self.variable_name[0]]
            dataset_u = NetCDFDataset(self.file_path[1], 'r')
            var_u = dataset_u.variables[self.variable_name[1]]
            dataset_v = NetCDFDataset(self.file_path[2], 'r')
            var_v = dataset_v.variables[self.variable_name[2]]

            z_level = 0  
            data_slice_chl = var_chl[0, z_level, :, :]
            for i in range(0, data_slice_chl.shape[0] - self.crop_size + 1, self.step_size):
                    for j in range(0, data_slice_chl.shape[1] - self.crop_size + 1, self.step_size):
                            cropped_region = data_slice_chl[i:i + self.crop_size, j:j + self.crop_size]
                            if np.all(cropped_region != 0):
                                for t in range(3 * self.seq_len, var_chl.shape[0]):
                                    cropped_image_chl = var_chl[t, z_level, :, :][i:i + self.crop_size, j:j + self.crop_size]
                                    cropped_image_u = var_u[t, z_level, :, :][i:i + self.crop_size, j:j + self.crop_size]
                                    cropped_image_v = var_v[t, z_level, :, :][i:i + self.crop_size, j:j + self.crop_size]
                                    
                                    cropped_image_chl = self._process_images(cropped_image_chl, 'chl')

                                    cropped_image_u = self._process_images(cropped_image_u)
                                    cropped_image_v = self._process_images(cropped_image_v)
                                    
                                    cropped_image = np.stack([cropped_image_chl, cropped_image_u, cropped_image_v], axis=0)
                                    
                                    self.valid_crops.append(cropped_image)
            
            print(f"Number of training images: {len(self.valid_crops)}")
            print(f"Number of sequences: {len(self.valid_crops) - self.seq_len + 1}")

    def _process_images(self, images, name=None):
        cropped_image = np.expand_dims(images, axis=0)
        cropped_image = cropped_image.astype(np.float32)
        if name == 'chl':
            epsilon = 1e-8
            cropped_image = np.log(cropped_image + epsilon)
        cropped_image = (cropped_image - cropped_image.min()) / (cropped_image.max() - cropped_image.min())   
        if self.transforms is not None:
            cropped_image = self.transforms(torch.tensor(cropped_image))
        return cropped_image
    
    def _process(self, images):
        cropped_image = np.expand_dims(images, axis=0)
        cropped_image = cropped_image.astype(np.float32)
        
        return cropped_image
            
    def __getitem__(self, idx):
        start_idx = idx #* self.seq_len
        end_idx = start_idx + self.seq_len
        
        sequence = self.valid_crops[start_idx:end_idx]
        sequence = np.array(sequence)
        sequence = np.transpose(sequence, (1, 2, 0, 3, 4)) 
        video_tensor = torch.tensor(sequence, dtype=torch.float32)
        
        return video_tensor
    