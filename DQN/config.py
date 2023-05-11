import os
import torch

#モデルのパラメーターの保存先
DQN_PARAM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'param/DQNparam')
GAMMA = 0.9
EPSILON = 0.01  #最小値
NUM_EPI = int(2e6)
NUM_STEP = int(1e2)
INTERVAL = 100
DEVICE = (
             "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
) 
