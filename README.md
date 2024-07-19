# ISTP (Interaction Scenario Trajectory Prediction with State Encoder)

## System Architecture
<img src="https://github.com/user-attachments/assets/a073a915-7853-4ba3-8896-08fd6f578294" width="560" height="330"/>

## Setup
All code was developed and tested on Ubuntu 20.04 wuth python 3.7 and PyTorch 1.13.0.

## Tensor Board 

tensorboard --logdir=directory_to_log_file
ex) tensorboard --logdir="/home/ngnadmin/dev/ngn_2024/MMT4/MMT/scripts2/log/noscene_v2_2/state2/obs8/1"

## Data
[waterloo]
* location: mmt/datasets/waterloo/~
* [preprocessed data link]()
  - img folder: image raw(.png), image feature file(.pkl) 
  - train, val, test folder: trajectory, agent state information(.csv)
    
## Models (Checkpoint files)
[google drive]()

## Evaluate Models
> MMT/scripts/ python evaluate_noscene.py

## Train a model
> MMT/scripts/ ./run_traj_noscene_1.sh

## Draw Trajectory using plt
> MMT/ python /.visualization/draw_trajectory_noscene1.py

## Dataset
[Waterloo multi-agent traffic dataset: intersection]()

## Acknowledgement

Thanks for the model structure idea and code from [sgan](https://github.com/agrimgupta92/sgan), [d2-tpred](https://github.com/VTP-TL/D2-TPred) 
