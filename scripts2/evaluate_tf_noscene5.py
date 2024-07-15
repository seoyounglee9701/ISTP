import argparse
import os
import torch

from attrdict import AttrDict

import sys

# _path = os.path.abspath(__file__)
_path = os.getcwd()
_path = _path.split("/")[:-1]
_path = "/".join(_path)
print(f"sys에 등록할 경로 [_path]:{_path}")

sys.path.append(_path)

from mmt.data.loader import data_loader
from mmt.losses import displacement_error, final_displacement_error

from mmt.models2.mmtf_noscene_v5 import TrajectoryGenerator # model 불러오기
from mmt.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--num_lstm_layers', default=1, type=int)
parser.add_argument('--num_tf_layers', default=4, type=int)

parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--model_path', default=os.getcwd() + '/output/noscene_v5/state1/obs8/1', type=str)
# parser.add_argument('--state_type', default=2, type=int) # v0: no state, v1: acc1+acc2+speed+ang, v2: acc1+acc2+speed, v3: acc1+acc2+ang, v4: speed
# parser.add_argument('--obs_len', default=8, type=int)
# parser.add_argument('--pred_len', default=8, type=int)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        state_type = args.state_type, # 
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_lstm_layers=args.num_lstm_layers,
        num_tf_layers=args.num_tf_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        # neighborhood_size=args.neighborhood_size,
        # grid_size=args.grid_size,
        batch_norm=args.batch_norm)

    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator

def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_

def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, obs_sel_state, obs_traffic,             
            pred_traj_gt, _, _,             
            obs_traj_rel, pred_traj_gt_rel,             
            _, loss_mask, seq_start_end, _ 
            ) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end,
                    obs_sel_state, obs_traffic
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde

def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        ckpt_path = path 
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        ade, fde = evaluate(_args, loader, generator, args.num_samples)
        print('Loaded ckpt path: {}'.format(ckpt_path))
        print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
            _args.dataset_name, _args.pred_len, ade, fde))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
