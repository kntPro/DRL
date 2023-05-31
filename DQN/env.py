import gymnasium as gym
from gymnasium import Wrapper
from config import *
import random

class CartPoleFallReward(Wrapper): #棒が倒れたら負の報酬を与えるCartPole用のラッパー環境
    def __init__(self, env, fall):
        super().__init__(env)
        self.fReward=fall
        self.env = env

    def step(self, action,):
        obs, re, ter, trun, _ = self.env.step(action)
        if ter:
            reward = self.fReward
        else:
            reward = re
        
        return obs, reward, ter, trun, _

def main():
    e = gym.make("CartPole-v1")
    env = CartPoleFallReward(e,FALL_REWARD)
    obs,info = env.reset()
    for i in range(100):
        o,r,ter,trun, _= env.step(random.randint(0,1))
        print(f"{i}回目")
        print(f"termination:{ter}")
        print(f"truncation:{trun}")
        print(f"reward:{r}\n")

        if(ter or trun):
            break
    env.close()

if __name__ == '__main__':
    main()
