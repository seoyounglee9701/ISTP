# Imports
import torch
import logging
import os

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from PIL import Image
import torchvision.models as models
import torchvision.transforms as T
from torch.autograd import Variable

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """
    Creates a PyTorch dataset from folder, returning two tensor images.
    Args: 
    main_dir : directory where images are stored.
    transform (optional) : torchvision transforms to be applied while making dataset
    """

    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)
                
        all_files = os.listdir(self.main_dir)
        
        path_list=[]
        for file in all_files:
            path = main_dir+file
            path_list.append(path)
            image = Image.open(path).convert("RGB")
            if self.transform is not None:
                tensor_image = self.transform(image)

        path_list = sorted(path_list)
        self.path_list = path_list   


    def __len__(self):
        print(f"total images: {len(self.all_imgs)}")
        return len(self.all_imgs)

    def __getitem__(self, index):
        """
        Returns at given index
        :return image_tensor :  3*640*360 image to tensor
        :return idx: instance id (mainly for debugging)
        """
        img_path = self.path_list[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            tensor_image = self.transform(image)
        out = tensor_image

        return out