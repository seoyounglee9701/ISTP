from torch.utils.data import DataLoader
from mmt.datasets.sdd_load import TrajectoryDataset, seq_collate # dataset without image  

import torchvision.transforms as T

def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        state_type=args.state_type, # default=, type=int
        obs_len=args.obs_len,
        pred_len= args.pred_len, # args.pred_len, # 
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)
    

    return dset, loader 
