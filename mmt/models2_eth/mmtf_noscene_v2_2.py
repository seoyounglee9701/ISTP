# Title: tf_ver2_2 (TF_{en8})
# memory를 obs_len 단위로 설정
# eth_loader에 맞게 generator, discriminator
# TF Model: w/o scene context, with 1) traj, 2) state
# Create Date: 24.06.22
# Revision Date: 

import torch, gc
import torch.nn as nn
import math
from torch.autograd import Variable

from torch.nn import functional as f
from mmt.models2.position_embedding import add_seq_pos_emb
from torch.nn import Transformer

import sys
def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)
############################################################################################################################

class TrajEncoder(nn.Module):
    """ """
    def __init__(
        self, h_dim=64, embedding_dim=64
        , mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(TrajEncoder, self).__init__() 
    
        self.mlp_dim=1024    
        self.h_dim = h_dim # 64
        self.embedding_dim = embedding_dim # embedding to lstm cell : e_{i}^t --> 64
        self.num_layers = 1 # single layer to get a fixed length vector e_{i}^t --> 1
        
        self.encoder = nn.LSTM( # input_size, hidden_size, num_layers
            embedding_dim, h_dim, 1 # num_layers=1
        )
        
        self.spatial_embedding = nn.Linear( # location (x_{i}^t, y_{i}^t) + tl_code_{i}^t --> embeddings
            2, embedding_dim # input sample size, output sample size
        ) 
    
    def init_hidden(self, batch):
        
        h = torch.zeros(1, batch, self.h_dim).cuda()
        c = torch.zeros(1, batch, self.h_dim).cuda()
        return (h, c)
    
    def forward(self, obs_traj):
        """ 
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        
        # Encode observed Trajectory
        batch = obs_traj.size(1) # npeds
        # total = batch * (MAX_PEDS if padded else 1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.contiguous().view(-1,2))
        obs_traj_embedding = obs_traj_embedding.contiguous().view(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch) # self.init_hidden(total)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h 
    
# Trajectory Encoder
# obs_traj를 tf encoder를 사용하여 인코딩한 후 얻은 특징 표현
class TransformerTrajEncoder(nn.Module):
    def __init__(self, h_dim=64, embedding_dim=64, nhead=4, num_layers=1, dropout=0.0):
        super(TransformerTrajEncoder, self).__init__()
        self.obs_len = 8
        self.input_dim = 2
        self.embedding_dim = embedding_dim
        
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, input_dim)
        Output:
        - enc_traj: Tensor of shape (obs_len, batch, embedding_dim)
        """
        
        # Spatial embedding
        spatial_embedded = self.spatial_embedding(obs_traj)
        
        # Positional encoding
        spatial_embedded_pos = self.pos_encoder(spatial_embedded)
        
        # Transformer encoding
        enc_traj = self.transformer_encoder(spatial_embedded_pos)
        
        return enc_traj


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

############################################################################################################################

# Agent State Encoder

class TransformerStateEncoder(nn.Module):
    def __init__(self, state_type, h_dim=64, embedding_dim=64, num_layers=1, nhead=4, dropout=0.0):
        super(TransformerStateEncoder, self).__init__()

        self.obs_len=8

        if state_type == 1:
            state_size = 2
        elif state_type == 2:
            state_size = 1
        elif state_type == 0:
            state_size = 0
        else:
            print("invalid state_type")

        print(f"state_type:{state_type}번째 상태 인코더")
        print(f"state_size:{state_size}")

        self.input_dim = state_size
        self.embedding_dim = embedding_dim
        
        self.state_embedding = nn.Linear(state_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.state_size = state_size

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
    
    def forward(self, obs_state):
        """
        Inputs:
        - obs_state: Tensor of shape (obs_len, batch, state_size)
        Output:
        - enc_state: Tensor of shape (obs_len, batch, embedding_dim)
        """
        # State embedding
        state_embedded = self.state_embedding(obs_state)
        
        # Positional encoding
        state_embedded_pos = self.pos_encoder(state_embedded)
        
        # Transformer encoding
        enc_state = self.transformer_encoder(state_embedded_pos)
        
        return enc_state

############################################################################################################################

class PoolHiddenNet(nn.Module):

    """Pooling module as proposed in Social GAN paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim # 64
        self.bottleneck_dim = bottleneck_dim # 1024
        self.embedding_dim = embedding_dim # 64

        mlp_pre_dim = embedding_dim + h_dim # 128
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim] # [128, 512, 1024]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
            )

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 3D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.contiguous().view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.contiguous().view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.contiguous().view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h
    
############################################################################################################################

class TransformerDecoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, pred_len, 
        embedding_dim=128,
        h_dim=128, num_layers=4, num_heads=8, mlp_dim=1024,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024, 
        activation='relu', batch_norm=True, pooling_type='pool_net',
    ):
        super(TransformerDecoder, self).__init__()
        
        self.pred_len = pred_len 
        self.h_dim = h_dim
        self.embedding_dim = 128 
        self.pool_every_timestep = pool_every_timestep
        self.mlp_dim = mlp_dim 
        self.num_layers=num_layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=128, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet(
                    embedding_dim = self.embedding_dim,
                    h_dim=128,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )
            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )
        self.spatial_embedding = nn.Linear(2, 128)
        self.memory_embedding= nn.Linear(128, embedding_dim)
        self.hidden2pos = nn.Linear(128, 2)

    def generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask
    
    def forward(self, last_pos, last_pos_rel, memory, seq_start_end):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2) # [133,2]
        - last_pos_rel : Tensor of shape (batch, 2) # [133,2]
        - memory: tensor of shape (8, batch, decoder_h_dim=128) // decoder_h
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.pred_len, batch, 2)
        """
        # print(f"last_pos_rel#:{last_pos_rel}") # 
        # print(f"last_pos_rel shape#:{last_pos_rel.shape}") # torch.Size([133,2])
        # print(f"last_pos #:{last_pos}") # 
        # print(f"last_pos shape #:{last_pos.shape}") # torch.Size([133,2])
        # print(f"memory shape #:{memory.shape}") # torch.Size([8, 133, 128])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = last_pos.size(0) # npred
        pred_traj_fake_rel=[]

        decoder_input = self.spatial_embedding(last_pos_rel)

        start_target = decoder_input
        # print(f"start_target shape ##: {start_target.shape}") # torch.Size([133, 64])

        # decoder_h(결합된 입력)을 d_model 차원으로 변환
        # memory = self.memory_embedding(memory)
        # print(f"memory shape ##:{memory.shape}") # torch.Size([133, 64])

        start_target = start_target.unsqueeze(0)
        # print(f"target shape ###: {start_target.shape}") # torch.Size([1, 133, 64])

        # print(f"memory shape ###:{memory.shape}") # torch.Size([8, 133, 64])


        predicted_trajectory = [start_target]

        #  오토리그레시브 방식으로 타임 스텝별 예측
        for i in range(self.pred_len): 
            target = torch.cat(predicted_trajectory, dim=0) # 현재까지 예측된 타겟 시퀀스
            # print(f"현재까지 예측된 타겟 시퀀스 target:{target}")
            
            # 각 예측 시간 단계에 대해 마스크 생성: 각 호출에서 적절한 마스크가 생성됨
            tgt_mask = torch.triu(torch.ones(target.size(0), target.size(0)) == 1).transpose(0, 1)
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
            tgt_mask = tgt_mask.to(device)
    
            # print(f"tgt_mask shape: {tgt_mask.shape}") # torch.Size([1, 1])


            # tgt (타겟 시퀀스 길이, 배치 크기, 임베딩 차원)
            # memory (소스 시퀀스 길이=1, 배치 크기, 임베딩 차원)

            output = self.transformer_decoder(
                tgt=target, #
                memory=memory, # 
                tgt_mask=tgt_mask # 
            )
            # print(f"output:{output}")
            # print(f"output shape:{output.shape}") # torch.Size([8, 133, 64])

            rel_pos = self.hidden2pos(output[-1, :, :]) 
            # print(f"rel_pos:{rel_pos}")
            # print(f"rel_pos shape:{rel_pos.shape}") # ([133, 2])
            # rel_pos = self.hidden2pos(output.contiguous().view(-1, self.h_dim)) # 위의 코드와 아마 같은 의미 (pred=1인 경우)
            curr_pos = rel_pos+last_pos
            # print(f"curr_pos shape:{curr_pos.shape}") # ([133, 2])
            # print(f"curr_pos :{curr_pos}") 

            embedding_input = rel_pos # ([133, 2])
            decoder_input = self.spatial_embedding(embedding_input)  # torch.Size([133, 64])
            decoder_input = decoder_input.contiguous().view(1, batch, self.embedding_dim) # torch.Size([133, 64])
            pred_traj_fake_rel.append(rel_pos.contiguous().view(batch, -1))
            last_pos = curr_pos
        
        # predicted_trajectory = torch.cat(predicted_trajectory[:], dim=0) 
        # print(f"predicted_trajectory.shape:{predicted_trajectory.shape}")  # 출력 크기: (예측할 시퀀스 길이, 배치 크기, 임베딩 차원) torch.Size([7, 133, 64])
        # print(f"predicted_trajectory:{predicted_trajectory}")

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        # print(f"pred_traj_fake_rel.shape:{pred_traj_fake_rel.shape}") # torch.Size([8, 133, 2])
        # print(f"pred_traj_fake_rel:{pred_traj_fake_rel}")


        return pred_traj_fake_rel, memory
    
############################################################################################################################
class TrajectoryGenerator(nn.Module):
    def __init__(
        self, 
        obs_len, 
        pred_len, 
        state_type,
        embedding_dim=64, 
        encoder_h_dim=64, 
        decoder_h_dim= 128, #64,# 128, 
        mlp_dim=1024, 
        num_lstm_layers=1,
        num_tf_layers=4,
        noise_dim=(0, ),
        bottleneck_dim=1024,
        noise_type='gaussian', 
        noise_mix_type='ped', 
        pooling_type= 'pool_net' ,# None,
        pool_every_timestep=True, 
        dropout=0.0, 
        activation='relu', 
        batch_norm=True, 
        # neighborhood_size=2.0, 
        # grid_size=8,
    ):
        super(TrajectoryGenerator, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.state_type = state_type
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_lstm_layers = num_lstm_layers
        self.num_tf_layers = num_tf_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024

        self.encoder_lstm = TrajEncoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim, # 64
            mlp_dim=mlp_dim,
            num_layers=num_lstm_layers,
            dropout=dropout
        )

        self.encoder = TransformerTrajEncoder(
            h_dim=encoder_h_dim, 
            embedding_dim=embedding_dim, 
            nhead=4, 
            num_layers=num_tf_layers, 
            dropout=dropout)

        self.encoder2 = TransformerStateEncoder(
            state_type = state_type,# state_type,
            h_dim = encoder_h_dim, 
            embedding_dim = embedding_dim,
            num_layers = num_tf_layers,
            nhead= 4 ,
            dropout = dropout
        )
   
        if pooling_type == 'pool_net':
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim, # 64
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim, # 1024
                activation=activation,
                batch_norm=batch_norm
            )

        if self.noise_dim[0] == 0: # noise_dim=(0, ),
            self.noise_dim = None  ## 이 경우에 해당
        else:
            self.noise_first_dim = noise_dim[0]
        
        # Decoder Hidden
        if pooling_type:
            input_dim = 2 * encoder_h_dim + bottleneck_dim # 64*2+1024=1152
            # input_dim = input_dim*8
            print(f"input_dim:{input_dim}")
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            # mlp 각 층의 입력 및 출력 차원을 지정하는 리스트 (입력->중간->마지막 크기에 맞게 차원 조정)
            # input_dim: 첫 번째 층의 입력 차원
            # mlp_dim : mlp 중간 층들의 출력 차원 = 1024
            # decoder_dim-self.nosie_first_dim : 마지막 층 출력 차원 : 128
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim # mlp_dim=1024, decoder_h_dim=128, self.noise_first_dim=0
            ] # list
            
            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.decoder = TransformerDecoder(
            pred_len=pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            num_layers=num_tf_layers,
            dropout=dropout,
            activation=activation,
            pooling_type=pooling_type,
            mlp_dim=mlp_dim,
            bottleneck_dim=bottleneck_dim
        )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim: # 현재 noise_dim=None이므로 이 경우에 해당
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else: # 현재 ped
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type) # 이 경우에 해당

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].contiguous().view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h    
    
    def mlp_decoder_needed(self):
        if (
            self.noise_dim or self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False
        
    
    def forward(self, obs_traj, obs_traj_rel, seq_start_end, 
                obs_state,
                # img,
                user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        end_pos = obs_traj[-1, :, :]

        # Encode seq
        final_encoder_h = self.encoder_lstm(obs_traj_rel)
        enc_traj = self.encoder(obs_traj_rel) # 수정 obs_traj->obs_traj_rel 240619
        # enc_traj = self.encoder(obs_traj)
        final_enc_traj = enc_traj[-1] # [batch, embedding_dim]
        enc_state = self.encoder2(obs_state)
        final_enc_state = enc_state[-1] # [batch, embedding_dim]

        # Pool States
        if self.pooling_type: # 풀링을 한다면

            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos) # Tensor of shape [batch, bottleneck_dim] #1024

            # print("final_encoder_h shape:", final_encoder_h.shape) # torch.Size([1, 133, 64])
            # print("enc_traj shape:", enc_traj.shape)                # torch.Size([8, 133, 64])
            # print("enc_state shape:", enc_state.shape)              # torch.Size([8, 133, 64])
            # print("pool_h shape:", pool_h.shape)                    # torch.Size([133, 1024])  [batch, bottleneck_dim]
            # print("pool_h.unsqueeze(0) shape:", pool_h.unsqueeze(0).shape) #  [1, 133, 1024]

            # Construct input hidden states for decoder : 디코더 입력에 대한 특징 벡터


            mlp_decoder_context_input = torch.cat(
                [
                    enc_traj, #  ([8, 133, 64])
                    enc_state, #  ([8, 133, 64])
                    pool_h.unsqueeze(0).expand(8, -1, -1), # ([8, 133, 1024])
                    # img_ft     # batch, 512
                 ]
                 , dim=2) # feature concat --> 디코더 입력에 대한 특징 벡터
            # print(f"mlp_decoder_context_input shape:{mlp_decoder_context_input.shape}") # [8,133,1152]
            # sys.exit(0)

        # else:
        #     mlp_decoder_context_input = final_encoder_h.view(
        #         -1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input # --> 디코더 입력에 대한 특징 벡터

        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise)  # 노이즈가 추가된 입력벡터  # noise+feature concat --> 디코더 입력에 대한 특징 벡터
        
        memory = decoder_h
        # decoder_h = decoder_h.unsqueeze(0) # 
        # decoder_h = decoder_h.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1) # torch.Size([4, 133, 128])

        # print(f"obs_traj[-1] shape: {obs_traj[-1].shape}") # obs_traj[-1] shape: torch.Size([batch, 2])
        # print(f"obs_traj_rel[-1] shape: {obs_traj_rel[-1].shape}") # obs_traj_rel[-1] shape: torch.Size([batch, 2])
        # print(f"memory shape: {memory.shape}") # decoder_h shape: torch.Size(8,[batch, 128])
        # print(f"seq_start_end shape: {seq_start_end.shape}") # seq_start_end shape: torch.Size([8, 2])

        # sys.exit(0)
        pred_traj_fake_rel, _ = self.decoder(
            obs_traj[-1], obs_traj_rel[-1], memory, seq_start_end
        )
        
        return pred_traj_fake_rel
    
############################################################################################################################

class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, state_type, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_lstm_layers=1, num_tf_layers=4,
        activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type
        self.state_type=state_type

        self.traj_encoder = TransformerTrajEncoder(
            h_dim=embedding_dim, 
            embedding_dim=embedding_dim, 
            nhead=4,  # Number of attention heads
            num_layers=num_tf_layers, 
            dropout=dropout)
    
        # State Encoder
        # self.state_encoder = TransformerStateEncoder(
        #     state_type=state_type,
        #     h_dim=h_dim, 
        #     embedding_dim=embedding_dim,
        #     num_layers=num_tf_layers,
        #     nhead=4,
        #     dropout=dropout
        # )    
        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet( 
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim, # 1024
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - obs_state : X
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        # print(f"traj shape:{traj.shape}")  # ([16, 133, 2])
        # print(f"traj_rel shape:{traj_rel.shape}")  # ([16, 133, 2])
        # print(f"obs_state shape:{obs_state.shape}") 


        final_traj_h = self.traj_encoder(traj_rel)
        # print(f"final_traj_h shape:{final_traj_h.shape}") # torch.Size([16, 133, 64])

        # final_state_h = self.state_encoder(obs_state) 
        # print(f"final_state_h shape:{final_state_h.shape}") # torch.Size([8, 133, 64])

        # final_h = torch.cat((final_traj_h, final_state_h), dim=-1)
        final_h = final_traj_h

        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local': # 이 경우에 해당됨
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[0]
            )
        scores = self.real_classifier(classifier_input)
        return scores
    
############################################################################################################################

