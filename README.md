# ISTP (Interaction Scenario Trajectory Prediction with State Encoder)

## System Architecture
<img src="https://github.com/user-attachments/assets/a073a915-7853-4ba3-8896-08fd6f578294" width="560" height="330"/>

## Setup
All code was developed and tested on Ubuntu 20.04 with python 3.7 and PyTorch 1.13.0.

## Visualization

tensorboard --logdir=directory_to_log_file
ex) tensorboard --logdir="tb_file_path"

## Data
[waterloo]
* location: ~/datasets/waterloo/~
* [preprocessed data link]()
  - img folder: image raw(.png), image feature file(.pkl) 
  - train, val, test folder: trajectory, agent state information(.csv)
    
## Models (Checkpoint files)
[google drive]()

## Evaluate Models
> ~/scripts/ python evaluate_noscene.py

## Train a model
> ~/scripts/ ./run_traj_noscene_1.sh

## Draw Trajectory using plt
run ./vis/(dataset-name)/(file-name).ipynb

## Dataset
* [Waterloo multi-agent traffic dataset: intersection](https://uwaterloo.ca/waterloo-intelligent-systems-engineering-lab/datasets/waterloo-multi-agent-traffic-dataset-intersection)
* [Stanford Drone Dataset-Death Circle](https://cvgl.stanford.edu/projects/uav_data/)
* [ETH Dataset](https://paperswithcode.com/dataset/eth)
* [inD Datatset](https://levelxdata.com/ind-dataset/) : (To Do)

## Acknowledgement

Thanks for the model structure idea and code from [sgan](https://github.com/agrimgupta92/sgan), [d2-tpred](https://github.com/VTP-TL/D2-TPred), [Trajectory-Transformer](https://github.com/FGiuliari/Trajectory-Transformer) 

## Changelog


## TO-DO
- [ ] refactoring folder name
- [ ] tf-based model hyper-parameter tuning
- [ ] upload ckpt file
