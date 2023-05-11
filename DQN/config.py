import os


#モデルのパラメーターの保存先
DQN_PARAM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'param/DQNparam')
GAMMA = 0.9
EPSILON = 0.1
NUM_EPI = int(2e6)
NUM_STEP = int(1e2)
INTERVAL = 100
