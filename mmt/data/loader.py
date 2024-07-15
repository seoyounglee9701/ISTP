from torch.utils.data import DataLoader
from mmt.datasets.waterloo_all import TrajectoryDataset, seq_collate # dataset with traffic light, image  

import torchvision.transforms as T

def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        state_version=args.state_type, # default=2, type=int
        obs_len=args.obs_len,
        pred_len= args.pred_len, # args.pred_len, # 4, 8, 12
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)
    

    return dset, loader 
