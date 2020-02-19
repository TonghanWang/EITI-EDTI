import gym
import numpy as np
import copy


class Island:

	def __init__(self, args, rank):
		self.args = args
		self.rank = rank
		self.initialization(args)

	def initialization(self, args):
		self.is_print = self.rank == 0
		self.n_agent = args.n_agent
		self.n_action = 6
		self.n_dim = 2
		self.n_wolf = 1
		self.n_landmark = self.args.i_num_landmark

		self.args = args
		self.size = args.size
		self.dec_int = args.gamma_dec != 0
		self.penalty = args.penalty
		self.attack_range = 1

		self.is_full_obs = not args.island_partial_obs

		if self.is_print:
			print(args.save_path)
			print('>>>>>>>>>>>>>dec_int', self.dec_int)

		if self.args.i_num_landmark == 2:
			self.landmarks = np.array([[int(self.size * 0.8), int(self.size * 0.2)],
			                           [int(self.size // 2), int(self.size * 0.8)]])
		elif self.args.i_num_landmark == 3:
			self.landmarks = np.array([[int(self.size * 0.8), int(self.size * 0.2)],
			                           [int(self.size // 2), int(self.size * 0.8)],
			                           [int(self.size // 2), int(self.size // 2)]])
		elif self.args.i_num_landmark == 4:
			self.landmarks = np.array([[int(self.size * 0.8), int(self.size * 0.2)],
			                           [int(self.size // 2), int(self.size * 0.8)],
			                           [int(self.size * 0.8), int(self.size * 0.8)],
			                           [int(self.size // 2), int(self.size * 0.2)]])
		elif self.args.i_num_landmark == 9:
			self.landmarks = []
			location_list = [int(self.size * 0.2), int(self.size // 2), int(self.size * 0.8)]
			for x in location_list:
				for y in location_list:
					self.landmarks.append([x, y])
			self.landmarks = np.array(self.landmarks)

		self.landmark_visited = [False for _ in range(self.n_landmark)]

		# Movable Agent
		self.state_n = [np.array([0, 0]) for _ in range(self.n_agent)]
		self.agent_alive_n = [True for _ in range(self.n_agent)]
		self.done_n = [False for _ in range(self.n_agent)]

		# Movable Wolf
		self.wolf_n = [self.random_wolf() for _ in range(self.n_wolf)]
		self.wolf_alive_n = [True for _ in range(self.n_wolf)]
		self.pre_wolf_alive_n = [True for _ in range(self.n_wolf)]

		self.eye = np.eye(self.size)
		self.flag = np.eye(2)

		self.wolf_recover_time = self.args.island_wolf_recover_time

		# Power
		self.agent_max_power = self.args.island_agent_max_power
		self.wolf_max_power = self.args.island_wolf_max_power
		self.agent_power_eye = np.eye(self.agent_max_power)
		self.wolf_power_eye = np.eye(self.wolf_max_power)
		self.agent_power = np.array([self.agent_max_power - 1 for _ in range(self.n_agent)])
		self.wolf_power = np.array([self.wolf_max_power - 1 for _ in range(self.n_wolf)])
		self.died_wolf_number = 0

		self.agent_score = [[0. for _ in range(self.n_agent)] for _ in range(self.n_wolf)]

		# Used by OpenAI baselines
		self.action_space = gym.spaces.Discrete(self.n_action)
		self.observation_space = gym.spaces.Box(low=-1, high=1,
		                                        shape=[(args.size * 2 + self.agent_max_power) * self.n_agent +
		                                               (args.size * 2) * self.n_wolf +
		                                               self.n_landmark + self.n_wolf * int(self.is_full_obs)])
		self.num_envs = args.num_env
		self.metadata = {'render.modes': []}
		self.reward_range = (-200., 20000.)
		self.spec = 2

		self.t_step = 0
		self.agent_power_zeros_like = np.zeros_like(self.agent_power)

	def reset(self):

		self.landmark_visited = [False for _ in range(self.n_landmark)]

		self.state_n = [np.array([0, 0]) for _ in range(self.n_agent)]
		self.agent_alive_n = [True for _ in range(self.n_agent)]
		self.done_n = [False for _ in range(self.n_agent)]

		self.wolf_n = [self.random_wolf() for _ in range(self.n_wolf)]
		self.wolf_alive_n = [True for _ in range(self.n_wolf)]
		self.pre_wolf_alive_n = [True for _ in range(self.n_wolf)]

		self.agent_power = np.array([self.agent_max_power - 1 for _ in range(self.n_agent)])
		self.wolf_power = np.array([self.wolf_max_power - 1 for _ in range(self.n_wolf)])
		self.died_wolf_number = 0

		self.agent_score = [[0. for _ in range(self.n_agent)] for _ in range(self.n_wolf)]

		self.t_step = 0

		return self.obs_n()

	def random_wolf(self):
		while True:
			wolf = np.random.randint(0, self.size, [self.n_dim])
			flag = False
			for i, state in enumerate(self.state_n):
				if self.agent_alive_n[i] and \
						wolf[0] - 1 <= state[0] <= wolf[0] + 1 and \
						wolf[1] - 1 <= state[1] <= wolf[1] + 1:
					flag = True
					break
			if not flag:
				break
		return wolf

	def step(self, action_n):

		if self.t_step % self.wolf_recover_time == 0:
			for i in range(self.n_wolf):
				if self.wolf_alive_n[i]:
					self.wolf_power[i] = min(self.wolf_power[i] + 1, self.wolf_max_power - 1)

		self.t_step += 1

		for w_i, wolf in enumerate(self.wolf_n):
			if self.wolf_alive_n[w_i]:
				t_s = 0
				for i, action in enumerate(action_n):
					if self.agent_alive_n[i]:
						if action == 5:
							if wolf[0] - self.attack_range <= self.state_n[i][0] <= wolf[0] + self.attack_range and \
									wolf[1] - self.attack_range <= self.state_n[i][1] <= wolf[1] + self.attack_range:
								t_s += 1
				t_harm = t_s
				a_score = 1
				if t_s > 1:
					t_harm += t_s
					a_score = 1. * t_harm / t_s
				self.wolf_power[w_i] = max(self.wolf_power[w_i] - t_harm, 0)

				for i, action in enumerate(action_n):
					if self.agent_alive_n[i]:
						if action == 5:
							if wolf[0] - self.attack_range <= self.state_n[i][0] <= wolf[0] + self.attack_range and \
									wolf[1] - self.attack_range <= self.state_n[i][1] <= wolf[1] + self.attack_range:
								self.agent_score[w_i][i] += a_score

		for i, action in enumerate(action_n):
			if self.agent_alive_n[i]:
				new_row = -1
				new_column = -1
				if action == 0:
					new_row = max(self.state_n[i][0] - 1, 0)
					new_column = self.state_n[i][1]
				elif action == 1:
					new_row = self.state_n[i][0]
					new_column = min(self.state_n[i][1] + 1, self.size - 1)
				elif action == 2:
					new_row = min(self.state_n[i][0] + 1, self.size - 1)
					new_column = self.state_n[i][1]
				elif action == 3:
					new_row = self.state_n[i][0]
					new_column = max(self.state_n[i][1] - 1, 0)
				elif action == 4 or action == 5:
					new_row = self.state_n[i][0]
					new_column = self.state_n[i][1]
				self.state_n[i] = np.array([new_row, new_column])

		for w_i, wolf in enumerate(self.wolf_n):
			if self.wolf_alive_n[w_i] and self.wolf_power[w_i] == 0:
				self.wolf_alive_n[w_i] = False
				self.died_wolf_number += 1

		for w_i, wolf in enumerate(self.wolf_n):
			if self.wolf_alive_n[w_i]:
				t_sum = 0
				for i in range(self.n_agent):
					if self.agent_alive_n[i] and \
							wolf[0] - self.attack_range <= self.state_n[i][0] <= wolf[0] + self.attack_range and \
							wolf[1] - self.attack_range <= self.state_n[i][1] <= wolf[1] + self.attack_range:
						t_sum += 1
				if t_sum > 0:
					t_sum = 2 // t_sum
				for i in range(self.n_agent):
					if self.agent_alive_n[i] and \
							wolf[0] - self.attack_range <= self.state_n[i][0] <= wolf[0] + self.attack_range and \
							wolf[1] - self.attack_range <= self.state_n[i][1] <= wolf[1] + self.attack_range:
						self.agent_power[i] = max(self.agent_power[i] - t_sum, 0)

		for i, state in enumerate(self.state_n):
			if self.agent_alive_n[i] and self.agent_power[i] == 0:
				self.agent_alive_n[i] = False
				self.done_n[i] = True
			else:
				self.done_n[i] = False

		# Move Wolf
		for i, wolf in enumerate(self.wolf_n):
			if self.wolf_alive_n[i]:
				new_row = new_column = 0
				action = np.random.randint(5)

				if action == 0:
					new_row = max(wolf[0] - 1, 0)
					new_column = wolf[1]
				elif action == 1:
					new_row = wolf[0]
					new_column = min(wolf[1] + 1, self.size - 1)
				elif action == 2:
					new_row = min(wolf[0] + 1, self.size - 1)
					new_column = wolf[1]
				elif action == 3:
					new_row = wolf[0]
					new_column = max(wolf[1] - 1, 0)
				elif action == 4:
					new_row = wolf[0]
					new_column = wolf[1]

				self.wolf_n[i] = np.array([new_row, new_column])

		wolf_info = np.concatenate([[wolf[0], wolf[1]]
		                            for i, wolf in enumerate(self.wolf_n)], axis=0)
		info_state_n = []
		for i, state in enumerate(self.state_n):
			full_state = np.concatenate([state, [self.agent_power[i]], wolf_info], axis=0)
			info_state_n.append(full_state)

		info = {'wolf': self.wolf_n, 'state': copy.deepcopy(info_state_n)}

		return_obs = self.obs_n()
		return_rew, info_r = self.reward()
		return_done = self.done()

		info['rew'] = info_r

		return return_obs, return_rew, return_done, info

	def obs_n(self):
		return [self.obs() for _ in range(self.n_agent)]

	def obs(self):

		wolf = np.concatenate([np.concatenate([self.eye[wolf[0]], self.eye[wolf[1]]], axis=0)
		                       for i, wolf in enumerate(self.wolf_n)], axis=0)

		a_state = np.concatenate([np.concatenate([self.eye[state[0]],
		                                          self.eye[state[1]],
		                                          self.agent_power_eye[self.agent_power[i]]], axis=0)
		                          for i, state in enumerate(self.state_n)], axis=0)

		ans = [a_state, wolf, self.landmark_visited]
		if self.is_full_obs:
			ans.append(self.wolf_alive_n)
		ans = np.concatenate(ans).copy()
		return ans

	def reward(self):
		rew = np.array([0. for _ in range(self.n_agent)])

		num_kill_info = 0

		for w_i, wolf in enumerate(self.wolf_n):
			if self.pre_wolf_alive_n[w_i] and not self.wolf_alive_n[w_i]:
				# print(self.agent_score[w_i])
				# print(self.agent_power)
				self.pre_wolf_alive_n[w_i] = False
				rew += 300.
				num_kill_info += 1
				'''
				# A new wolf
				self.wolf_alive_n[w_i] = True
				self.wolf_n[w_i] = self.random_wolf()
				self.wolf_power[w_i] = self.wolf_max_power - 1
				'''

		landmark_num = 0
		time_length = []
		for agent_index in range(self.n_agent):
			if self.agent_alive_n[agent_index]:
				time_length.append(1.)
				for i, landmark in enumerate(self.landmarks):
					if not self.landmark_visited[i] and (landmark == self.state_n[agent_index]).all():
						landmark_num += 1
						rew += 10.
						self.landmark_visited[i] = True
			else:
				time_length.append(0.)

		info = {}
		info['kill'] = num_kill_info
		info['landmark'] = landmark_num
		info['death'] = self.done_n
		info['time_length'] = time_length

		return rew, info

	def done(self):
		if self.t_step >= self.args.episode_length or (self.agent_power == self.agent_power_zeros_like).any():
			self.reset()
			return 1

		return 0

	def close(self):
		self.reset()
