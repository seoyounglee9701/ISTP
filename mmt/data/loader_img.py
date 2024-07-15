from torch.utils.data import DataLoader
from mmt.data.img_data import ImageDataset  ####

import torchvision.transforms as T

def img_loader(args, path):
    transforms = T.Compose([T.ToTensor()])
    img_dset = ImageDataset(
        path,
        transforms) # .tensor_list 

    img_loader = DataLoader(
        img_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        collate_fn=None) # collate_fn=None
    
    return img_dset, img_loader
