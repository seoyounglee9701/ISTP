# ResNetBackbone으로 수행

import os
import torch
import torch.nn as nn
import pickle as pk
from typing import Tuple
from torchvision.models import resnet18

from loader_image3 import img_loader #####
import logging

# arg 생성
class CreateArg():
    def __init__(self):
        # Dataset options
        self.loader_num_workers = 0
        self.batch_size = 1                #### batch_size check !!!!!!

args = CreateArg() 

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


RESNET_VERSION_TO_MODEL = {'resnet18': resnet18, }

class ResNetBackbone(nn.Module):
    """
    Outputs tensor after last convolution before the fully connected layer.

    Allowed versions: resnet18, resnet34, resnet50, resnet101, resnet152.
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
        :return: Tensor of shape [batch_size, n_convolution_filters]. For resnet50,
            the shape is [batch_size, 2048].
        """
        backbone_features = self.backbone(input_tensor)
        return torch.flatten(backbone_features, start_dim=1)
    
# 이미지 파일 불러오기

dir_num=[769, 770, 771, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785]

for i in range(len(dir_num)):
    image_tensor_list=[]
    image_feature_list = []

    if i>=0 and i<=7: # 8
        data_dir="./mmt/datasets/waterloo/img/train/"+ str(dir_num[i])+"/"
    elif i>7 and i<=11: # 4
        data_dir="./mmt/datasets/waterloo/img/val/"+ str(dir_num[i])+"/"
    elif i>11 and i<=13: # 1
        data_dir="./mmt/datasets/waterloo/img/test/"+ str(dir_num[i])+"/"
    else:
        print("Error")

    
    all_files = [os.path.join(data_dir, path) for path in os.listdir(data_dir) if path[0] != "." and path.endswith(".jpg")] # 이미지 파일 경로
    len_file = len(all_files)

    print("-----------------------------------------")
    print(f"dir_num: {dir_num[i]}")
    print(f"file num in dir: {len_file}")
    print("-----------------------------------------")

    all_files=sorted(all_files) # 번호순으로 정렬
    # print(all_files)

    train_path = data_dir
    print(f"train_path: {train_path}")

    logger = logging.getLogger(__name__)
    logger.info("Initializing train dataset")
    
    # train_dset은 index를 통해 이미지를 tensor로 변경하여 출력 
    train_dset, _ = img_loader(args, train_path) # train_dset은 ImageDataset, train_loader는 DataLoader

    # train_dset은 index를 통해 이미지를 tensor로 변경하여 출력한다. 

    # preprocess = transforms.Compose([
    #     transforms.Resize(256), # 이미지 크기 변경
    #     transforms.CenterCrop(224), # 중앙 부분을 잘라서 크기 조절
    #     transforms.ToTensor(), # tensor.Tensor 형식으로 변경 [0, 255] ->  [0,1]
    # ])

    for k in range(len_file):
        image_tensor_list.append((train_dset[k]).unsqueeze(0)) # 전처리 # torch.Size([1, 3, 360, 640])
    print("------------------------------------------------------------")
    print(f"directory : {dir_num[i]} | original image_tensor done")
    print("------------------------------------------------------------")

    for j in range(len_file):        
        image_feature_list.append(resnet(image_tensor_list[j]))# .squeeze(0)) # squeeze 
        pkl_store_name = all_files[j].split(sep='/')[-1].split(sep='.jpg')[0]+'.pkl'
        pkl_store_path = os.path.join(data_dir, pkl_store_name)

        with open(pkl_store_path, 'wb') as f:

            pk.dump(image_feature_list[j],f, protocol=4) #

    print(f"directory {dir_num[i]} | image_feature .pkl file saved")
    print("==========================================================")
    print("==========================================================")

    