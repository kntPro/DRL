import os
import torch

#モデルのパラメーターの保存先
DQN_PARAM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'param/DQNparam')
GAMMA = 0.9
EPSILON = 0.01  #最小値
EPSILON_COE = int(5e5) #  アニーリングを終了するステップ数
NUM_EPI = int(1e6)
NUM_STEP = int(1e2)
MEMORY = int(1e5)
INTERVAL = 100
FALL_REWARD = -10 #棒が倒れたときの負の報酬
DEVICE = (
             "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
) 
