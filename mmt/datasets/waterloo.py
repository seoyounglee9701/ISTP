import logging
import os
import math

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

import pickle as pk
import torchvision.transforms as T

logger = logging.getLogger(__name__)


def seq_collate(data):                      
    (
        obs_seq_list, 
        obs_state_list,
        obs_traffic_list,
        
        pred_seq_list, 
        pred_state_list,
        pred_traffic_list,
        
        obs_seq_rel_list, 
        pred_seq_rel_list,
        
        non_linear_ped_list,
        loss_mask_list,
        img_list
    ) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    obs_state = torch.cat(obs_state_list, dim=0).permute(2, 0, 1)
    obs_traffic = torch.cat(obs_traffic_list, dim=0).permute(2, 0, 1)
    
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    pred_state = torch.cat(pred_state_list, dim=0).permute(2, 0, 1)
    pred_traffic = torch.cat(pred_traffic_list, dim=0).permute(2, 0, 1)
    
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    img_list = torch.cat(img_list, dim=0).repeat(obs_traj.size(1),1,1) ###
    
    out = [                 
        obs_traj, 
        obs_state,
        obs_traffic,
        
        pred_traj, 
        pred_state,
        pred_traffic,
        
        obs_traj_rel,     
        pred_traj_rel, 
        
        non_linear_ped,
        
        loss_mask, 
        seq_start_end,
        img_list
    ]

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    print(f"read_file:{_path}")
    delim = '\t'
    with open(_path, 'r') as f:
        for line in f:
            # print(line)
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, 
             threshold
             ):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len) # np.linspace(start point, end point, num in traj)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold: # error
       return 1.0
    else:
       return 0.0
    return 0.0

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, 
        data_dir,
        obs_len=8, 
        pred_len=12, 
        skip=1, 
        threshold=0.002, 
        min_agent=1, 
        delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format (only directory path)
        - <frame_id> <agent_id> <x> <y> <speed> <tan_acc> <lat_acc> <angle> <tl_code> <time>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_agent: Minimum number of agents that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir # train_path = "~/mmt/datasets/waterloo/train"
        # print(f"self.data_dir:{self.data_dir}")
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len # 
        self.delim = delim

        all_files = os.listdir(self.data_dir) # load all files to list, 
        # print(f"data_dir file: {all_files}") # check
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files] # data_dir\_path
        all_files = sorted(all_files)
        num_agents_in_seq = []
        
        seq_list = []
        seq_list2 = [] # state
        seq_list3 = [] # traffic 
        
        seq_list_rel = []

        loss_mask_list = []
        non_linear_agent = []

        fet_map = {} # img
        fet_list = []

        idx_num=0
        file_count=0
        for path in all_files:
            
            dir_name, file_name_ext = os.path.split(path) 
            file_name, ext = os.path.splitext(file_name_ext)
            # path = dir_name + '\\str(folder_num)\\' + file_name + ext 
            path2 = dir_name + '2/'+ file_name + '2'+ ext
            path3 = dir_name + '2/'+ file_name + '3'+ ext
            
            data = read_file(path, delim) # ex : 0769_prep.txt (10 columns including <frame_id> <agent_id> <x> <y> <speed> <tan_acc> <lat_acc> <angle> <tl_code> <time>)
            data2 = read_file(path2, delim) # <frame_id> <agent_id> <speed> <tan_acc> <lat_acc> <angle> 
            data3 = read_file(path3, delim) # <frame_id> <tl_code> 
            
            frames = np.unique(data[:, 0]).tolist() # array with redundant values removed, Slicing only the 0th column (<frame_id>) of every row.
            # print(f"len(frames) : {len(frames)}")
            # print(f"frame : {frames}")
            frame_data_all = []
            frame_data = []
            state_data = []
            traffic_data = []

        # img data
            img_path = os.path.split(dir_name)[0]+'/img' # ~/mmt/datasets/waterloo/img

            train_type = os.path.split(os.path.split(path)[0])[1]


            if train_type=='train':
                img_path = img_path + '/train/' # ~/mmt/datasets/waterloo/img/train/
                img_dir_num = os.listdir(img_path) 
                img_dir_num = sorted(img_dir_num) # ['769', '770', '771', '775', '776', '777', '778', '779']
                print(img_dir_num)
                img_path = img_path + str(img_dir_num[idx_num]) # ~/mmt/datasets/waterloo/img/train/760

            elif train_type =='val':
                img_path = img_path + '/val/'                 
                img_dir_num = os.listdir(img_path)
                img_dir_num = sorted(img_dir_num) 
                print(img_dir_num)
                img_path = img_path + str(img_dir_num[idx_num])

            elif train_type =='test':
                img_path = img_path + '/test/'  
                img_dir_num = os.listdir(img_path)
                img_dir_num = sorted(img_dir_num) 
                print(img_dir_num)
                img_path = img_path + str(img_dir_num[idx_num])
            else:
                print("error")
        

            print(f"img_path:{img_path}")

            file_count = len(os.listdir(img_path))//2

            for f_num in range(file_count):
                frame_num = 0
                pkl_name =  str(os.path.split(img_path)[1]) + "_frame_" + str(frame_num)+".pkl" # 769_frame_0.pkl

                pkl_path = img_path + '/' + pkl_name            # ~/mmt/datasets/waterloo/img/train/760/760.pkl

                with open(pkl_path, 'rb') as handle:
                    new_fet = pk.load(handle, encoding='bytes') # load feature values ​​of images

                fet_map[pkl_name] = new_fet.unsqueeze(0) # key : pkl filename, value : feature vector
            
            print(f"{idx_num+1} of {len(img_dir_num)} (Current Dir / Total Dir) ")
            idx_num+=1


            for frame in frames:
                frame_data_all.append(data[frame == data[:, 0], :4])    # all
                frame_data.append(data[frame == data[:, 0], :4])        # frame_data, frame data for each frame (agent idx, x, y) : 2D list
                state_data.append(data2[frame == data2[:, 0], :])
                traffic_data.append(data3[frame == data3[:, 0], ])

            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip)) # sequences num, ex : (7994-20)/1 = 7975

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(                         # create curr seq_data by spliting num_sequnce
                    frame_data[idx : idx + self.seq_len], axis=0
                )
                curr_seq_data2 = np.concatenate(                        
                    state_data[idx:idx + self.seq_len], axis=0          # ex) seq 1: 1-20 frame, seq 2: 2-21 frame ... 
                )
                curr_seq_data3 = np.concatenate(                         
                    traffic_data[idx:idx + self.seq_len], axis=0              
                )
            
                agents_in_curr_seq = np.unique(curr_seq_data[:, 1])       # list of agents currently in seq, slicing the 1st (agent information) column of every row.

                curr_seq_rel = np.zeros((len(agents_in_curr_seq), 2,        # (agents num of curr seq, 2, seq_len) 
                                         self.seq_len))
                
                curr_seq = np.zeros((len(agents_in_curr_seq), 2, self.seq_len)) # (agents num of curr seq, 2, seq_len) 
                curr_seq2 = np.zeros((len(agents_in_curr_seq), 4, self.seq_len)) # (agents num of curr seq, s_n, seq_len) --> state (4, 2)
                curr_seq3 = np.zeros((len(agents_in_curr_seq), 1, self.seq_len)) # ((agents num of curr seq, 1, seq_len)  --> traffic light 
                                                
                curr_loss_mask = np.zeros((len(agents_in_curr_seq),
                                           self.seq_len))
                
                num_agents_considered = 0
                _non_linear_agent = []
                
                for _, agent_id in enumerate(agents_in_curr_seq):           # agents currently in seq
                    curr_agent_seq = curr_seq_data[ curr_seq_data[:, 1] ==  # agent's position in curr seqeunce : (20, 2)
                                                 agent_id, :]
                    curr_agent_seq2 = curr_seq_data2[ curr_seq_data2[:, 1] ==  # (20, s_n), (n = 4, 2, 1)
                                                 agent_id, :]
                    curr_agent_seq3 = curr_seq_data3[ curr_seq_data3[:, 1] ==  # (20, 1)
                                                 agent_id, :]
                    
                    curr_agent_seq = np.around(curr_agent_seq, decimals=4)  # (20, 2)
                    curr_agent_seq2 = np.around(curr_agent_seq2, decimals=4)  # (20, 4)
                    curr_agent_seq3 = np.around(curr_agent_seq3, decimals=4)  # (20, 1)
                    
                    agent_front = frames.index(curr_agent_seq[0, 0]) - idx
                    agent_end = frames.index(curr_agent_seq[-1, 0]) - idx + 1
                    
                    if agent_end - agent_front != self.seq_len:
                        continue
                    curr_agent_seq = np.transpose(curr_agent_seq[:, 2:])
                    curr_agent_seq2 = np.transpose(curr_agent_seq2[:, 2:])
                    curr_agent_seq3 = np.transpose(curr_agent_seq3[:, 2:])
                    
                    curr_agent_seq = curr_agent_seq
                    
                    # Make coordinates relative
                    rel_curr_agent_seq = np.zeros(curr_agent_seq.shape)
                    
                    rel_curr_agent_seq[:, 1:] = \
                        curr_agent_seq[:, 1:] - curr_agent_seq[:, :-1]
                    _idx = num_agents_considered
                    
                    curr_seq[_idx, :, agent_front:agent_end] = curr_agent_seq
                    curr_seq2[_idx, :, agent_front:agent_end] = curr_agent_seq2
                    curr_seq3[_idx, :, agent_front:agent_end] = curr_agent_seq3
                    
                    curr_seq_rel[_idx, :, agent_front:agent_end] = rel_curr_agent_seq
                    
                    # Linear vs Non-Linear Trajectory
                    _non_linear_agent.append(
                        poly_fit(curr_agent_seq, pred_len 
                                 ,threshold
                                 )
                                 )
                    curr_loss_mask[_idx, agent_front:agent_end] = 1
                    num_agents_considered += 1

                if num_agents_considered > min_agent:
                    non_linear_agent += _non_linear_agent
                    num_agents_in_seq.append(num_agents_considered)
                    loss_mask_list.append(curr_loss_mask[:num_agents_considered])
                    
                    seq_list.append(curr_seq[:num_agents_considered])
                    seq_list2.append(curr_seq2[:num_agents_considered])
                    seq_list3.append(curr_seq3[:num_agents_considered])
                    
                    seq_list_rel.append(curr_seq_rel[:num_agents_considered])
                    fet_list.append(pkl_name)

        self.num_seq = len(seq_list)
        
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list2 = np.concatenate(seq_list2, axis=0)
        seq_list3 = np.concatenate(seq_list3, axis=0)
        
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)

        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_agent = np.asarray(non_linear_agent)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.obs_state = torch.from_numpy(
            seq_list2[:, :, :self.obs_len]).type(torch.float)  
        self.obs_traffic = torch.from_numpy(
            seq_list3[:, :, :self.obs_len]).type(torch.float)    
        
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.pred_state = torch.from_numpy(
            seq_list2[:, :, self.obs_len:]).type(torch.float)
        self.pred_traffic = torch.from_numpy(
            seq_list3[:, :, self.obs_len:]).type(torch.float)
    
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_agent = torch.from_numpy(non_linear_agent).type(torch.float)
        
        cum_start_idx = [0] + np.cumsum(num_agents_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        self.fet_map = fet_map
        self.fet_list = fet_list
        

    def __len__(self):              # len(train_dset)
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :],                        # 0
            self.obs_state[start:end, :],                       # 1
            self.obs_traffic[start:end, :],                     # 2
                        
            self.pred_traj[start:end, :],                       # 3
            self.pred_state[start:end, :],                      # 4
            self.pred_traffic[start:end, :],                    # 5
            
            self.obs_traj_rel[start:end, :],                    # 6                            
            self.pred_traj_rel[start:end, :],                   # 7 
                        
            self.non_linear_agent[start:end],                   # 8
            self.loss_mask[start:end, :],                       # 9
            self.fet_map[self.fet_list[index]] # tuple          # 10
                                
        ]
        return out