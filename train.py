from collections import deque
from random import choice

import torch
import numpy as np

import gymnasium as gym
import metaworld
from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy
from metaworld.policies.sawyer_window_close_v2_policy import SawyerWindowCloseV2Policy
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy

from skill_transformer import SkillTransformer
from buffer import Buffer


class ActionSequence():
    def __init__(self, length, act_dim):
        self.length = length
        self.act_dim = act_dim
        self.buf = deque(maxlen=length)
        self.reset()

    def reset(self):
        for _ in range(self.length):
            self.buf.append(np.zeros((self.act_dim), dtype=np.float32))

    def add_get(self, action):
        self.buf.append(action)
        return np.asarray(self.buf, dtype=np.float32)


device = torch.device('cuda')

# env = gym.make('HalfCheetah-v5')
# obs_dim = env.observation_space.shape[-1]
# act_dim = env.action_space.shape[-1]

env_name = 'window-close-v2'
# env_name = 'reach-v2'
# env_name = 'pick-place-v2'
mt1 = metaworld.MT1(env_name, seed=0)
env = mt1.train_classes[env_name]()
tasks = mt1.train_tasks
obs_dim = env.observation_space.shape[-1]
act_dim = env.action_space.shape[-1]

act_sequence_length = 16
act_seq = ActionSequence(act_sequence_length, act_dim)

# p = SawyerReachV2Policy()
p = SawyerWindowCloseV2Policy()
# p = SawyerPickPlaceV2Policy()

buffer = Buffer(10000, obs_dim, act_sequence_length, act_dim, device=device)
stf = SkillTransformer(act_dim).to(device)
optim = torch.optim.Adam(stf.parameters(), lr=0.0001)

for _ in range(10):
    env.set_task(choice(tasks))
    obs, info = env.reset()
    act_seq.reset()
    for _ in range(150):
        # act = env.action_space.sample()
        act = p.get_action(obs)
        next_obs, rwd, ter, tru, info = env.step(act)
        buffer.add(obs, act_seq.add_get(act))
        obs = next_obs

        o, a = buffer.sample(128)
        loss = stf.compute_loss(a)
        optim.zero_grad()
        loss.backward()
        optim.step()

        print(loss)

        if ter or tru: break