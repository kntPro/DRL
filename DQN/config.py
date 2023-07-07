import os
import torch
from collections import namedtuple



Transition = namedtuple('Transition',("action","state","reward","next_state","done"))
#モデルのパラメーターの保存先
DQN_PARAM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'param/DQNparam')
GAMMA = 0.99
EPSILON = 0.01  #最小値
EPSILON_COE = int(5e2) #  アニーリングを終了するステップ数
NUM_EPI = int(1e3)
NUM_STEP = int(1e3)
BATCH_SIZE =128
MEMORY =int(2e3)
INTERVAL = 100
FALL_REWARD = -10 #棒が倒れたときの負の報酬
DEVICE = (
             "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
) 
