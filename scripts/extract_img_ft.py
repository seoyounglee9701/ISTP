# exteact image feature using CNN backbone
# image file to tensor then save .pkl flles

import argparse
import os
import torch
import torch.nn as nn
import pickle as pk
from typing import Tuple
from torchvision.models import resnet18, resnet50


import sys

_path = os.path.dirname(__file__)
_path = _path.split("/")[:-1]
_path = "/".join(_path)
sys.path.append(_path)

from mmt.data.loader_img import img_loader 
from mmt.utils import get_img_dset_path
import logging

parser = argparse.ArgumentParser()

# Dataset options
parser.add_argument('--dataset_name', default='waterloo', type=str)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--loader_num_workers', default=0, type=int)

def trim_network_at_index(network: nn.Module, index: int = -1) -> nn.Module:
    """
    Returns a new network with all layers up to index from the back.
    :param network: Module to trim.
    :param index: Where to trim the network. Counted from the last layer.
    """
    assert index < 0, f"Param index must be negative. Received {index}."
    return nn.Sequential(*list(network.children())[:index])

def calculate_backbone_feature_dim(backbone, input_shape: Tuple[int, int, int]) -> int:
    """ Helper to calculate the shape of the fully-connected regression layer. """
    tensor = torch.ones(1, *input_shape)
    output_feat = backbone.forward(tensor)
    return output_feat.shape[-1]


RESNET_VERSION_TO_MODEL = {'resnet18': resnet18, 'resnet50' : resnet50}

class ResNetBackbone(nn.Module):
    """
    Outputs tensor after last convolution before the fully connected layer.
    Allowed versions: resnet18, resnet50.
    """

    def __init__(self, version: str):
        """
        Inits ResNetBackbone
        :param version: resnet version to use.
        """
        super().__init__()

        if version not in RESNET_VERSION_TO_MODEL:
            raise ValueError(f'Parameter version must be one of {list(RESNET_VERSION_TO_MODEL.keys())}'
                             f'. Received {version}.')

        self.backbone = trim_network_at_index(RESNET_VERSION_TO_MODEL[version](), -1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Outputs features after last convolution.
        :param input_tensor:  Shape [batch_size, n_channels, length, width].
        :return: Tensor of shape [batch_size, n_convolution_filters]. For resnet18,
            the shape is [batch_size, 512].
        """
        backbone_features = self.backbone(input_tensor)
        return torch.flatten(backbone_features, start_dim=1)


def main(args):

    # load image files
    dir_num = [769, 770, 771, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785]
    len_dir_num = len(dir_num)

    for i in range(len(dir_num)):
        image_tensor_list=[]
        image_feature_list = []

        if i >= 0 and i <= 7: # train
            data_dir = get_img_dset_path(args.dataset_name) + "/train/" + str(dir_num[i]) + "/"
        elif i > 7 and i <=11: # val
            data_dir = get_img_dset_path(args.dataset_name) + "/val/" + str(dir_num[i]) + "/"
        elif i > 11 and i <= 13: # test
            data_dir = get_img_dset_path(args.dataset_name) + "/test/" + str(dir_num[i]) + "/"
        else:
            print("Error")

        
        all_files = [os.path.join(data_dir, path) for path in os.listdir(data_dir) if path[0] != "." and path.endswith(".jpg")] # img file path
        len_file = len(all_files)

        print("============================================================================================")
        print(f"dir_path: {data_dir}")
        print(f"dir_num: {dir_num[i]} | {i} / {len_dir_num}")
        print(f"file num in dir: {len_file}")
        print("============================================================================================")

        all_files=sorted(all_files) # sort
        # print(all_files)

        train_path = data_dir
        print(f"train_path: {train_path}")

        logger = logging.getLogger(__name__)
        logger.info("Initializing train dataset")
        
        train_dset, _ = img_loader(args, train_path) 

        for k in range(len_file):
            image_tensor_list.append(train_dset[k].unsqueeze(0)) # dimension extension // torch.Size([1, 3, 360, 640])

        print("--------------------------------------------------------------------------------------------")
        print(f"Directory : {dir_num[i]} | image_tensor extracted")
        print("--------------------------------------------------------------------------------------------")

        rn_18 = ResNetBackbone('resnet18')
        for j in range(len_file):        
            image_feature_list.append(rn_18(image_tensor_list[j]))# .squeeze(0) # shape : [1, 512] (feature vector for each frame) 
            pkl_store_name = all_files[j].split(sep='/')[-1].split(sep='.jpg')[0]+'.pkl' 
            pkl_store_path = os.path.join(data_dir, pkl_store_name)

            with open(pkl_store_path, 'wb') as f:

                pk.dump(image_feature_list[j],f, protocol=4) #

        print(f"Directory {dir_num[i]} | image_feature .pkl file saved")
        print("============================================================================================")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
