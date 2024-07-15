# Update Date: 24.06.17
# args.state_type 값에 따라서 state encoder의 값에 다른 입력 
# state_type=1, speed+acc1+acc2+angle
# state_type=2, speed+acc1+acc2
# state_type=3, speed+angle
# state_type=4, speed

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
        obs_sel_state_list,
        obs_traffic_list,
        
        pred_seq_list,         
        pred_sel_state_list,
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
    obs_sel_state = torch.cat(obs_sel_state_list, dim=0).permute(2, 0, 1)
    obs_traffic = torch.cat(obs_traffic_list, dim=0).permute(2, 0, 1)

    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)    
    pred_sel_state = torch.cat(pred_sel_state_list, dim=0).permute(2, 0, 1)
    pred_traffic = torch.cat(pred_traffic_list, dim=0).permute(2, 0, 1)

    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    img_list = torch.cat(img_list, dim=0).repeat(obs_traj.size(1),1,1) ###
    
    out = [                 
        obs_traj,           # 0
        obs_sel_state,      # 1
        obs_traffic,        # 2

        pred_traj,          # 3         
        pred_sel_state,     # 4
        pred_traffic,       # 5

        obs_traj_rel,       # 6
        pred_traj_rel,      # 7
        
        non_linear_ped,     # 8
        loss_mask,          # 9
        seq_start_end,      # 10

        img_list            # 11
    ]

    return tuple(out)

# 파일을 읽어서 2차원 리스트로 저장 후 numpy 배열로 변환
# def read_file(_path, delim='\t'):
#     data = []                   # 파일의 각 줄을 저장할 리스트
#     print(f"read_file:{_path}")
#     delim = ',' #'\t'
#     with open(_path, 'r', encoding='utf-8-sig') as f: # # utf-8-sig를 사용하여 BOM을 제거
#         for line in f:
#             # print(line)
#             line = line.strip().split(delim) # 읽어온 줄에서 양 끝 공백 문자 제거, delim로 문자열 분리하여 리스트로 만듬
#             line = [float(i) for i in line] # 리스트 각 요소(문자열이었던 숫자)를 float 타입으로 변환
#             data.append(line)   # 변환된 리스트를 data 리스트에 추가
#     return np.asarray(data) # data 리스트를 numpy 배열로 변환
def read_file(_path, delim='\t', expected_length=10):
    data = []                   # 파일의 각 줄을 저장할 리스트
    print(f"read_file:{_path}")
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

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, 
        data_dir,
        state_version,
        obs_len, 
        pred_len, # 4, 8, 12 
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
        print(f"obs_len:{obs_len}")
        self.pred_len = pred_len
        print(f"pred_len:{pred_len}")
        self.state_version = state_version
        print(f"state_version:{state_version}")
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len # 
        print(f"seq_len: {self.seq_len}")
        self.delim = delim

        
        all_files = os.listdir(self.data_dir) # load all files to list, 
        # print(f"data_dir file: {all_files}") # check
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files] # data_dir\_path => all_files = ["~/mmt/datasets/waterloo/train/file1.txt", "", ...]
        all_files = sorted(all_files) # 정렬
        num_agents_in_seq = []
        
        seq_list = []
        seq_list_sel = [] # state selected
        seq_list3 = [] # traffic 

        seq_list_rel = []

        loss_mask_list = []
        non_linear_agent = []

        fet_map = {} # img
        fet_list = []

        idx_num=0
        file_count=0
        for path in all_files:
            print(f"path:{path}")
            
            dir_name, file_name_ext = os.path.split(path) 
            file_name, ext = os.path.splitext(file_name_ext)
            # path2 = dir_name + '2/'+ file_name + '2'+ ext # v1
            # path3 = dir_name + '2/'+ file_name + '3'+ ext
            # path4 = dir_name + '2/'+ file_name + '4'+ ext # v4: 속도
            # path5 = dir_name + '2/'+ file_name + '5'+ ext # v2: 속도, 가속도1, 가속도2
            # path6 = dir_name + '2/'+ file_name + '6'+ ext # v3: 속도, 각도

            if state_version ==1:    # 속도, 횡가속도, 종가속도, 조향각
                use_slicing=False
                column_indices = [0, 1, 4, 5, 6, 7]
                state_num = 4        # 상태 속성의 갯수
            elif state_version==2:   # 속도, 횡가속도, 종가속도
                use_slicing=False
                column_indices = [0, 1, 4, 5, 6]
                state_num = 3
            elif state_version==3:   # 속도, 조향각
                use_slicing=False
                column_indices = [0, 1, 4, 7]
                state_num = 2
            elif state_version==4:   # 속도
                use_slicing=False
                column_indices = [0,1,7]
                state_num = 1
            elif state_version==0: # no state
                state_num = 3 # 
                column_indices=[0, 1, 4, 5, 6] # default
            else:
                raise Exception("invalid state version")
            
            data = read_file(path, delim) # return np.asarray(data) // (10 columns including <frame_id> <agent_id> <x> <y> <speed> <tan_acc> <lat_acc> <angle> <tl_code> <time>)
            # print(f"data shape:{data.shape}") # (row개수, 10)
            data_state = read_file(path, delim) # 
            # print(f"data_state shape:{data_state.shape}") # (row개수, 10)
            data_traffic = read_file(path, delim) # <frame_id> <tl_code> 
            
            frames = np.unique(data[:, 0]).tolist() # 배열의 첫 번째 열(모든 행의 첫 번째 요소) 선택 후 고유한 값을 찾음(배열), 그 후 리스트로 변환
            # print(f"len(frames) : {len(frames)}")
            # print(f"frame : {frames}")
            frame_data = []
            sel_state_data = [] 
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
                raise Exception("invalid train type")
        
            print(f"Scene_Path:{img_path}")

            file_count = len(os.listdir(img_path))//2

            for f_num in range(file_count):
                frame_num = 0
                pkl_name =  str(os.path.split(img_path)[1]) + "_frame_" + str(frame_num)+".pkl" # 769_frame_0.pkl

                pkl_path = img_path + '/' + pkl_name            # ~/mmt/datasets/waterloo/img/train/760/760.pkl

                with open(pkl_path, 'rb') as handle:
                    new_fet = pk.load(handle, encoding='bytes') # load feature values ​​of images

                fet_map[pkl_name] = new_fet.unsqueeze(0) # key : pkl filename, value : feature vector
            
            print(f"{idx_num+1} of {train_type} {len(img_dir_num)} ") # (Current Dir / Total Dir) 
            idx_num+=1

            # 프레임 추출 및 데이터 준비
            for frame in frames: # 각 frame에 대해

                # data 배열에서 첫 번째 열이 현재 frame과 같은 행을 선택, 그러한 행의 첫 4열을 추출
                frame_data.append(data[frame == data[:, 0], :4])        # frame_data, frame data for each frame (agent idx, x, y) : 2D list
                
                # state
                # 불리언 마스크를 사용하여 행 선택
                selected_rows = data_state[frame == data_state[:, 0], :]
                # 선택된 행에서 특정 열 선택
                selected_columns = selected_rows[:, column_indices]
                # 결과를 리스트에 추가
                sel_state_data.append(selected_columns) 

                # tl_light
                # 불리언 마스크를 사용하여 행 선택
                selected_rows_tl = data_traffic[frame == data_traffic[:, 0], :]
                # 선택된 행에서 특정 열 선택
                column_indices_tl=[0,1,8]
                selected_columns_tl = selected_rows_tl[:, column_indices_tl]
                # 결과를 리스트에 추가
                traffic_data.append(selected_columns_tl) 

            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip)) # (고유한 프레임 총 수 - 시퀀스 길이)/skip(시퀀스 사이 간격) -> 시퀀스를 생성할 수 있는 총 개수 계산

            # 시퀀스 처리 : 프레임 시퀀스에 대해 데이터를 추출하고, 에이전트 별로 데이터를 준비함
            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(                         # create curr seq_data by spliting num_sequnce
                    frame_data[idx : idx + self.seq_len], axis=0
                )                                                         # axis=0은 행을 기준으로(수직 방향) 연결하라는 의미, 여러 프레임 데이터를 하나의 큰 배열로 만듬
                curr_seq_sel_data = np.concatenate(                         
                    sel_state_data[idx:idx + self.seq_len], axis=0              
                )                
                curr_seq_data3 = np.concatenate(                         
                    traffic_data[idx:idx + self.seq_len], axis=0              
                )

                agents_in_curr_seq = np.unique(curr_seq_data[:, 1])       # list of agents currently in seq, slicing the 1st (agent information) column of every row.

                curr_seq_rel = np.zeros((len(agents_in_curr_seq), 2, self.seq_len)) # (리스트에 포함된 고유한 에이전트의 수, 2, 시퀀스 길이)(agents num of curr seq, 2, seq_len) 
                
                curr_seq = np.zeros((len(agents_in_curr_seq), 2, self.seq_len)) # (agents num of curr seq, 2, seq_len) 

                curr_seq_sel = np.zeros((len(agents_in_curr_seq), state_num, self.seq_len)) # (agents num of curr seq, s_n, seq_len)      

                curr_seq3 = np.zeros((len(agents_in_curr_seq), 1, self.seq_len)) # (agents num of curr seq, 1, seq_len)  --> traffic light 

                curr_loss_mask = np.zeros((len(agents_in_curr_seq), self.seq_len))
                
                num_agents_considered = 0
                _non_linear_agent = []
                
                # 에이전트별 데이터 추출
                for _, agent_id in enumerate(agents_in_curr_seq):           # 현재 시퀀스에 있는 각 에이전트 
                    
                    # agent's position in curr seqeunce : (16, 2)

                    curr_agent_seq = curr_seq_data[ curr_seq_data[:, 1] ==  agent_id, :] # curr_seq_data 배열에서 특정 에이전트(ID)의 시퀀스 데이터 추출 -> 특정 에이전트 궤적 패턴 처리를 위해 사용

                    # 상태 데이터 필터링: 상태 데이터를 curr_seq_sel_data에서 에이전트 ID로 필터링하여 추출
                    # 배열의 형태 확인
                    # print(curr_seq_sel_data.shape) # (235, 1)
                    # 배열의 차원 확인
                    # print(curr_seq_sel_data.ndim) # 1
                    if curr_seq_sel_data.shape[1] == 1:
                        # 첫 번째 열 기준으로 필터링
                        curr_agent_seq_sel = curr_seq_sel_data[curr_seq_sel_data[:, 0] == agent_id, :]
                    else:
                        curr_agent_seq_sel = curr_seq_sel_data[curr_seq_sel_data[:, 1] == agent_id, :]   

                    curr_agent_seq3 = curr_seq_data3[ curr_seq_data3[:, 1] ==  agent_id, :] # (16, 1)
                    
                    curr_agent_seq = np.around(curr_agent_seq, decimals=4)  # 소수점 이하 4자리까지 반올림. (16, 2)수
                    curr_agent_seq_sel = np.around(curr_agent_seq_sel, decimals=4)  # (16, s_n)
                    curr_agent_seq3 = np.around(curr_agent_seq3, decimals=4)  # (16, 1)
                    
                    agent_front = frames.index(curr_agent_seq[0, 0]) - idx
                    agent_end = frames.index(curr_agent_seq[-1, 0]) - idx + 1
                    
                    if agent_end - agent_front != self.seq_len:
                        continue

                    # 시퀀스 배열에 데이터 저장

                    # [0,1] 컬럼은 frame_num, agent_num
                    curr_agent_seq = np.transpose(curr_agent_seq[:, 2:]) #
                    curr_agent_seq_sel = np.transpose(curr_agent_seq_sel[:, 2:]) # 
                    curr_agent_seq3 = np.transpose(curr_agent_seq3[:, 2:])

                    curr_agent_seq = curr_agent_seq

                    # Make coordinates relative
                    rel_curr_agent_seq = np.zeros(curr_agent_seq.shape)
                    
                    rel_curr_agent_seq[:, 1:] = \
                        curr_agent_seq[:, 1:] - curr_agent_seq[:, :-1]
                    _idx = num_agents_considered # 0
                    
                    curr_seq[_idx, :, agent_front:agent_end] = curr_agent_seq
                    curr_seq_sel[_idx, :, agent_front:agent_end] = curr_agent_seq_sel
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

                # print("curr_seq_sel shape:", curr_seq_sel.shape)  # Debugging 출력 추가
                # print("curr_agent_seq_sel shape:", curr_agent_seq_sel.shape)
                
                if num_agents_considered > min_agent:
                    non_linear_agent += _non_linear_agent
                    num_agents_in_seq.append(num_agents_considered)
                    loss_mask_list.append(curr_loss_mask[:num_agents_considered])
                    
                    seq_list.append(curr_seq[:num_agents_considered])
                    seq_list_sel.append(curr_seq_sel[:num_agents_considered])
                    seq_list3.append(curr_seq3[:num_agents_considered])

                    seq_list_rel.append(curr_seq_rel[:num_agents_considered])
                    fet_list.append(pkl_name)

        self.num_seq = len(seq_list)
        
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_sel = np.concatenate(seq_list_sel, axis=0)
        seq_list3 = np.concatenate(seq_list3, axis=0)

        seq_list_rel = np.concatenate(seq_list_rel, axis=0)

        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_agent = np.asarray(non_linear_agent)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.obs_sel_state = torch.from_numpy(
            seq_list_sel[:, :, :self.obs_len]).type(torch.float)  
        self.obs_traffic = torch.from_numpy(
            seq_list3[:, :, :self.obs_len]).type(torch.float)
        
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.pred_sel_state = torch.from_numpy(
            seq_list_sel[:, :, self.obs_len:]).type(torch.float)
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
            self.obs_sel_state[start:end, :],                   # 1
            self.obs_traffic[start:end, :],                     # 2

            
            self.pred_traj[start:end, :],                       # 3
            self.pred_sel_state[start:end, :],                  # 4
            self.pred_traffic[start:end, :],                    # 5

            self.obs_traj_rel[start:end, :],                    # 6                            
            self.pred_traj_rel[start:end, :],                   # 7
                        
            self.non_linear_agent[start:end],                   # 8
            self.loss_mask[start:end, :],                       # 9
            self.fet_map[self.fet_list[index]] # tuple          # 10
                                
        ]
        return out