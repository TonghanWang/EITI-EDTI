import gym
import copy
import numpy as np


class Leftward:
    def __init__(self, args, rank):
        self.args = args
        self.t_step = 0
        self.n_state = args.size
        self.n_action = 2
        self.n_agent = 2

        self.state_n = [0, self.n_state-1]

        self.eye = np.eye(args.size)

        # Used by OpenAI baselines
        self.action_space = gym.spaces.Discrete(self.n_action)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=[args.size])
        self.num_envs = args.num_env
        self.metadata = {'render.modes': []}
        self.reward_range = (-10., 1000.)
        self.spec = 2

    def initialization(self, args):
        self.n_state = args.size
        self.n_action = 2
        self.n_agent = 2
        self.t_step = 0

        self.state_n = [0, self.n_state-1]

        self.eye = np.eye(args.size)

        # Used by OpenAI baselines
        self.action_space = gym.spaces.Discrete(self.n_action)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=[args.size])
        self.num_envs = args.num_env
        self.metadata = {'render.modes': []}
        self.reward_range = (-10., 1000.)
        self.spec = 2

    def step(self, action_n):
        self.t_step += 1
        # Agent 0
        if action_n[0] == 0:
            self.state_n[0] = max(0, self.state_n[0] - 1)
        else:
            self.state_n[0] = min(self.n_state - 1, self.state_n[0] + 1)

        # Agent 1
        if action_n[1] == 0:
            self.state_n[1] = max(0, self.state_n[1] - 1)
        else:
            self.state_n[1] = min(self.n_state - 1, self.state_n[1] + 1)

        info = {'door': [0, 0], 'state': [np.array([state, 0]) for state in self.state_n]}

        return self.obs_n(), self.reward(), self.done(), info

    def reward(self):
        if self.state_n[0] == (self.n_state-1) and self.state_n[1] == (self.n_state-1):
            return [100, 100]

        return [0, 0]

    def observation(self, i):
        return np.concatenate([self.eye[self.state_n[i]]])

    def obs_n(self):
        return [self.observation(i) for i in range(self.n_agent)]

    def reset(self):
        self.t_step = 0
        self.state_n = [0, self.n_state - 1]

        return self.obs_n()

    def start_v(self, trainers):
        v = 0
        v += np.max(trainers[0].q[0])
        v += np.max(trainers[1].q[self.n_state - 1])

        return v

    def done(self):
        return self.state_n == [self.n_state - 1, self.n_state - 1] or self.t_step >= self.args.episode_length
