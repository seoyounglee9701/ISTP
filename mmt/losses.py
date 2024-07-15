import torch
import random

# 분류 문제에서 널리 사용
# 모델 출력이 목표 라벨과 얼마나 일치하는지를 나타냄
# 학습 과정에서 손실 최소화 하는 방향으로 학습
def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores. # 모델 출력 점수로 이루어진 텐서
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets. # 목표 라벨(0, 1로 구성된 PyTorch 텐서)

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of # 미니 배치에 대한 평균 BCE 손실을 포함하는 Pytorch 텐서
      input data.
    """
    neg_abs = -input.abs() # 입력 값의 절댓값에 마이너스를 붙임
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

# 생성자는 자신이 생성한 데이터를 실제 데이터처럼 보이게 하기 위해 판별자 출력을 최대화하려고 함
# 생성자 목표: 판별자를 속이는 것
# 생성한 데이터가 실제 데이터처럼 보이도록 함
def gan_g_loss(scores_fake):
    """
    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples # 가짜 샘플에 대한 점수, 생성자가 생성한 데이터에 대한 판별자의 출력

    Output:
    - loss: Tensor of shape (,) giving GAN generator loss # 생성자의 GAN 손실로, 스칼라 값
    """
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2) # 가짜 데이터에 대한 목표 값
    return bce_loss(scores_fake, y_fake)

# 실제 샘플과 가짜 샘플에 대한 손실을 합산하여 총 손실을 반환
# 실제 데이터와 생성된 데이터를 얼마나 잘 구분하는지 나타내는 손실을 계산
# 이 손실을 최소화하도록 학습
# GAN의 학습과정에서 생성기가 더 정교한 샘플을 생성하도록 도와줌
def gan_d_loss(scores_real, scores_fake):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples # 실제 샘플에 대한 판별기 점수를 나타내는 텐서
    - scores_fake: Tensor of shape (N,) giving scores for fake samples # 가짜 샘플에 대한 판별기 점수를 나타내는 텐서

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss # 판별기 손실 값을 나타내는 텐서
    """
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2) # 실제 샘플에 대한 목표 라벨: 0.7~1.2 사이 랜덤 값을 곱한 '1'의 텐서를 생성함
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3) # 가짜 샘플에 대한 목표 라벨: 0~0.3 사이 랜덤 값을 곱한 '0' 텐서를 생성함함
    loss_real = bce_loss(scores_real, y_real) # 실제 샘플에 대한 손실 계산 
    loss_fake = bce_loss(scores_fake, y_fake) # 가짜 샘플에 대한 손실 계산
    return loss_real + loss_fake #

# 예측 궤적, 실제 궤적, 손실 마스크를 이용해 L2 손실을 계산하는 함수
def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode='average'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory. # 각 시간 단계별 예측된 위치를 포함
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len) # 손실을 적용할 마스크 텐서
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    loss = (loss_mask.unsqueeze(dim=2) *
            (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)

# 유클리드 거리 기반 이동 오차를 계산하는 함수
def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss

# 마지막 위치를 기반으로 한 최종 이동 오차를 계산하는 함수
def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos. # 예측된 마지막 위치
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth # 실제 마지막 위치
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)
