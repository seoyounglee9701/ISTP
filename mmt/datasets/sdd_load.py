# Created Date: 24.06.23
# args.state_type 값에 따라서 state encoder의 값에 다른 입력 
# state_type=0, no state
# state_type=1, velocityy+acc
# state_type=2, velocity
# state_type=3, acc


import logging
import os
import math
import sys
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

def check_for_duplicate_data(data, agent_id):
    agent_data = data[data[:, 1] == agent_id, :]
    unique_frames = np.unique(agent_data[:, 0])
    if len(unique_frames) != agent_data.shape[0]:
        print(f"Agent ID {agent_id} has duplicate frames.")
        return True
    return False

def seq_collate(data):                      
    (
        obs_seq_list, 
        obs_sel_state_list,
        
        pred_seq_list,         
        pred_sel_state_list,
        
        obs_seq_rel_list, 
        pred_seq_rel_list,
        
        non_linear_ped_list,
        loss_mask_list,
        # img_list
    ) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    obs_sel_state = torch.cat(obs_sel_state_list, dim=0).permute(2, 0, 1)

    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)    
    pred_sel_state = torch.cat(pred_sel_state_list, dim=0).permute(2, 0, 1)

    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    # img_list = torch.cat(img_list, dim=0).repeat(obs_traj.size(1),1,1) ###
    
    out = [                 
        obs_traj,           # 0
        obs_sel_state,      # 1

        pred_traj,          # 2         
        pred_sel_state,     # 3

        obs_traj_rel,       # 4
        pred_traj_rel,      # 5
        
        non_linear_ped,     # 5
        loss_mask,          # 6
        seq_start_end,      # 7

        # img_list            # 8
    ]

    return tuple(out)


def read_file(_path, delim='\t', expected_length=6):
    data = []                   # 파일의 각 줄을 저장할 리스트
    # print(f"read_file:{_path}")
    delim = ',' #'\t'
    with open(_path, 'r', encoding='utf-8-sig') as f: # # utf-8-sig를 사용하여 BOM을 제거
        for line_num, line in enumerate(f, start=1):
            # print(line)
            line = line.strip().split(delim) # 읽어온 줄에서 양 끝 공백 문자 제거, delim로 문자열 분리하여 리스트로 만듬
            float_line = []

            if len(line) != expected_length:
                print(f"Warning: Line {line_num} does not have {expected_length} elements: {line}")
                continue  # 요소 개수가 다르면 스킵하고 다음 라인으로 이동

            for i in line:
                try:
                    float_line.append(float(i))
                except ValueError:
                    print(f"Warning: could not convert {i} to float")
                    float_line.append(0.0)  # ValueError가 발생하면 0.0으로 설정
            if len(float_line) > 0:  # 변환된 숫자가 있는 경우에만 추가
                data.append(float_line)
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

def detect_separator(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline()
        for sep in ['\t', ',', ' ', ';']:
            if sep in first_line:
                return sep
    return None

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, 
        data_dir,
        state_type,
        obs_len, 
        pred_len, 
        skip=1, 
        threshold=0.002, 
        min_agent=1, 
        delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format (only directory path)
        - <frame_id> <agent_id> <x> <y> <speed> <acc> 
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_agent: Minimum number of agents that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir # train_path = "~/mmt/datasets/eth/train"
        # print(f"self.data_dir:{self.data_dir}")
        self.obs_len = obs_len
        print(f"obs_len:{obs_len}")
        self.pred_len = pred_len
        print(f"pred_len:{pred_len}")
        self.state_type = state_type
        print(f"state_type:{state_type}")
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len # 
        print(f"seq_len: {self.seq_len}")
        self.delim = delim

        all_files = os.listdir(self.data_dir) # load all files to list, 
        # print(f"data_dir file: {all_files}") # check
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files] # data_dir\_path => all_files = ["~/mmt/datasets/eth/train/file1.txt", "", ...]
        all_files = sorted(all_files) # 정렬
        len_all_files=len(all_files)

        num_agents_in_seq = []
        
        seq_list = []
        seq_list_sel = [] # state selected

        seq_list_rel = []

        loss_mask_list = []
        non_linear_agent = []

        fet_map = {} # img
        fet_list = []

        # idx_num=0
        # file_count=0
        f_num=1

        for file_path in all_files:

            
            dir_name, file_name_ext = os.path.split(file_path) # 디렉토리명, 파일이름+확장자 // ~/mmt/datasets/eth/train, file1.txt
            dataset_split_type = os.path.split(dir_name)[1]
                        
            print(f"{dataset_split_type} / file_path:{file_path}") # "~/mmt/datasets/eth/train/file1.txt"
            print(f"{f_num} of total {len_all_files} ")
            f_num+=1
            if state_type ==1:    # 속도, 가속도
                column_indices = [0, 1, 4, 5]
                count_of_state_attributes = 2  # 상태 속성의 갯수
            elif state_type==2:   # 속도
                column_indices = [0, 1, 4]
                count_of_state_attributes = 1
            elif state_type==3:   # 가속도
                column_indices = [0, 1, 5]
                count_of_state_attributes = 1
            elif state_type==0: # no state
                count_of_state_attributes = 0  
                column_indices=[0, 1] # (f_id, a_id, x, y)
            else:
                raise Exception("invalid state_type")
            # 구분자 감지
            separator = detect_separator(file_path)

            data = read_file(file_path, separator) # return np.asarray(data) // (6 columns including (frame_id, agent_id, x, y, speed, acc) )
            # print(f"data shape:{data.shape}") # (count of rows, 6)

            data_state = read_file(file_path, separator) # 
            # print(f"data_state shape:{data_state.shape}") ## (count of rows, 6)
            
            frames = np.unique(data[:, 0]).tolist() # 배열의 첫 번째 열(모든 행의 첫 번째 요소) 선택 후 고유한 값을 찾음(배열), 그 후 리스트로 변환
            # print(frames)
            # print(f"len(frames) : {len(frames)}")
            # print(f"frame : {frames}")

            frame_data = []
            sel_state_data = [] 


            ################################# img data 불러오기 #################################
            # img_path = os.path.split(dir_name)[0]+'/img' # ~/mmt/datasets/eth/img

            # dataset_split_type = os.path.split(dir_name)[1]

            # if dataset_split_type=='train':
            #     img_path = img_path + '/train/' # ~/mmt/datasets/waterloo/img/train/
            #     img_dir_num = os.listdir(img_path) 
            #     img_dir_num = sorted(img_dir_num) # ['769', '770', '771', '775', '776', '777', '778', '779']
            #     print(img_dir_num)
            #     img_path = img_path + str(img_dir_num[idx_num]) # ~/mmt/datasets/waterloo/img/train/760

            # elif dataset_split_type =='val':
            #     img_path = img_path + '/val/'                 
            #     img_dir_num = os.listdir(img_path)
            #     img_dir_num = sorted(img_dir_num) 
            #     print(img_dir_num)
            #     img_path = img_path + str(img_dir_num[idx_num])

            # elif dataset_split_type =='test':
            #     img_path = img_path + '/test/'  
            #     img_dir_num = os.listdir(img_path)
            #     img_dir_num = sorted(img_dir_num) 
            #     print(img_dir_num)
            #     img_path = img_path + str(img_dir_num[idx_num])
            # else:
            #     raise Exception("invalid train type")
        
            # print(f"Scene_Path:{img_path}")

            # file_count = len(os.listdir(img_path))//2

            # for f_num in range(file_count):
            #     frame_num = 0
            #     pkl_name =  str(os.path.split(img_path)[1]) + "_frame_" + str(frame_num)+".pkl" # 769_frame_0.pkl

            #     pkl_path = img_path + '/' + pkl_name            # ~/mmt/datasets/waterloo/img/train/760/760.pkl

            #     with open(pkl_path, 'rb') as handle:
            #         new_fet = pk.load(handle, encoding='bytes') # load feature values ​​of images

            #     fet_map[pkl_name] = new_fet.unsqueeze(0) # key : pkl filename, value : feature vector
            
            # print(f"{idx_num+1} of {dataset_split_type} {len(img_dir_num)} ") # (Current Dir / Total Dir) 
            # idx_num+=1
            ###################################################################################################

            # 프레임 추출 및 데이터 준비

            for frame in frames: 

                # data 배열에서 첫 번째 열이 현재 frame과 같은 행을 선택, 그러한 행의 첫 4열을 추출
                frame_data.append(data[frame == data[:, 0], :4])        # frame_data, frame data for each frame (agent idx, x, y) : 2D list
                
                # state
                # 불리언 마스크 기법을 사용하여 행 선택
                selected_rows = data_state[frame == data_state[:, 0], :]
                # 선택된 행에서 특정 열 선택
                selected_columns = selected_rows[:, column_indices]
                # 결과를 리스트에 추가
                sel_state_data.append(selected_columns) 

            print(f"len frames:{len(frames)}") # 14065
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip)) # 14050 (고유한 프레임 총 수 - 시퀀스 길이)/skip(시퀀스 사이 간격) -> 시퀀스를 생성할 수 있는 총 개수 계산
            print(f"num_sequences:{num_sequences}")
            e_count=0
            # sys.exit()
            # 시퀀스 처리 : 프레임 시퀀스에 대해 데이터를 추출하고, 에이전트 별로 데이터를 준비함
            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(                         # create curr seq_data by spliting num_sequnce
                    frame_data[idx : idx + self.seq_len], axis=0
                )                                                         # axis=0은 행을 기준으로(수직 방향) 연결하라는 의미, 여러 프레임 데이터를 하나의 큰 배열로 만듬
                curr_seq_sel_data = np.concatenate(                         
                    sel_state_data[idx:idx + self.seq_len], axis=0              
                )                

                agents_in_curr_seq = np.unique(curr_seq_data[:, 1])       # list of agents currently in seq, slicing the 1st (agent information) column of every row.

                curr_seq_rel = np.zeros((len(agents_in_curr_seq), 2, self.seq_len)) # (리스트에 포함된 고유한 에이전트의 수, 2, 시퀀스 길이)(agents num of curr seq, 2, seq_len) 
                
                curr_seq = np.zeros((len(agents_in_curr_seq), 2, self.seq_len)) # (agents num of curr seq, 2, seq_len) 

                curr_seq_sel = np.zeros((len(agents_in_curr_seq), count_of_state_attributes, self.seq_len)) # (agents num of curr seq, c_s_a, seq_len)      

                curr_loss_mask = np.zeros((len(agents_in_curr_seq), self.seq_len))
                
                num_agents_considered = 0
                _non_linear_agent = []
                
                # 에이전트별 데이터 추출
                for for_idx , agent_id in enumerate(agents_in_curr_seq):           # 현재 시퀀스에 있는 각 에이전트 
                    
                    # agent's position in curr seqeunce : (seq_len, 2)

                    curr_agent_seq = curr_seq_data[ curr_seq_data[:, 1] ==  agent_id, :] # curr_seq_data 배열에서 특정 에이전트(ID)의 시퀀스 데이터 추출 -> 특정 에이전트 궤적 패턴 처리를 위해 사용
                    curr_agent_seq_sel = curr_seq_sel_data[curr_seq_sel_data[:, 1] == agent_id, :]   
                
                    curr_agent_seq = np.around(curr_agent_seq, decimals=4)  # 소수점 이하 4자리까지 반올림. (seq_len, 2)수
                    curr_agent_seq_sel = np.around(curr_agent_seq_sel, decimals=4)  # (seq_len, c_s_a)

                    # 중복 에이전트 데이터 확인
                    if check_for_duplicate_data(curr_seq_data, agent_id):
                        print(f"Duplicate data found for agent ID {agent_id}.")
                    
                    agent_front = frames.index(curr_agent_seq[0, 0]) - idx
                    agent_end = frames.index(curr_agent_seq[-1, 0]) - idx + 1
                    
                    if agent_end - agent_front != self.seq_len:
                        # print(f"Skipping agent_id {agent_id} due to sequence length mismatch. : {agent_end - agent_front}")
                        continue

                    # 시퀀스 배열에 데이터 저장

                    # [0,1] 컬럼은 frame_num, agent_num
                    curr_agent_seq = np.transpose(curr_agent_seq[:, 2:]) #
                    curr_agent_seq_sel = np.transpose(curr_agent_seq_sel[:, 2:]) # 

                    # Make coordinates relative
                    rel_curr_agent_seq = np.zeros(curr_agent_seq.shape)
                    
                    rel_curr_agent_seq[:, 1:] = \
                        curr_agent_seq[:, 1:] - curr_agent_seq[:, :-1]
                    _idx = num_agents_considered # 0

                    # 크기 확인
                    # print(curr_seq.shape)  # (num_agents, 2, seq_len)
                    # print(curr_agent_seq.shape) # (2, seq_len)
                    if curr_agent_seq.shape[1] != self.seq_len:
                        e_count+=1
                        continue
                    # print(idx, for_idx)

                    curr_seq[_idx, :, agent_front:agent_end] = curr_agent_seq
                    curr_seq_sel[_idx, :, agent_front:agent_end] = curr_agent_seq_sel
                   
                    curr_seq_rel[_idx, :, agent_front:agent_end] = rel_curr_agent_seq
                    
                    # Linear vs Non-Linear Trajectory
                    _non_linear_agent.append(
                        poly_fit(curr_agent_seq, pred_len 
                                 ,threshold
                                 )
                                 )
                    curr_loss_mask[_idx, agent_front:agent_end] = 1
                    num_agents_considered += 1

                # print("curr_seq_sel shape:", curr_seq_sel.shape)  # Debugging 출력 추가
                # print("curr_agent_seq_sel shape:", curr_agent_seq_sel.shape)
                
                if num_agents_considered > min_agent:
                    non_linear_agent += _non_linear_agent
                    num_agents_in_seq.append(num_agents_considered)
                    loss_mask_list.append(curr_loss_mask[:num_agents_considered])
                    
                    seq_list.append(curr_seq[:num_agents_considered])
                    seq_list_sel.append(curr_seq_sel[:num_agents_considered])

                    seq_list_rel.append(curr_seq_rel[:num_agents_considered])
                    # fet_list.append(pkl_name)
            print(f"e_count:{e_count}")
        self.num_seq = len(seq_list)
        
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_sel = np.concatenate(seq_list_sel, axis=0)

        seq_list_rel = np.concatenate(seq_list_rel, axis=0)

        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_agent = np.asarray(non_linear_agent)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.obs_sel_state = torch.from_numpy(
            seq_list_sel[:, :, :self.obs_len]).type(torch.float)  
        
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.pred_sel_state = torch.from_numpy(
            seq_list_sel[:, :, self.obs_len:]).type(torch.float)
    
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
        # self.fet_map = fet_map
        # self.fet_list = fet_list
        

    def __len__(self):              # len(train_dset)
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :],                        # 0
            self.obs_sel_state[start:end, :],                   # 1
            
            self.pred_traj[start:end, :],                       # 2
            self.pred_sel_state[start:end, :],                  # 3

            self.obs_traj_rel[start:end, :],                    # 4                            
            self.pred_traj_rel[start:end, :],                   # 5
                        
            self.non_linear_agent[start:end],                   # 6
            self.loss_mask[start:end, :],                       # 7
            # self.fet_map[self.fet_list[index]] # tuple          # 8
                                
        ]
        return out