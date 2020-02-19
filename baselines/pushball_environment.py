import gym
import matplotlib.pyplot as plt
import numpy as np
import copy


class PushBall:

	def __init__(self, args, rank):
		self.args = args
		self.rank = rank
		self.initialization(args)

	def random_start(self):
		#return np.array([self.size // 2, self.size // 2])
		#return np.array(np.random.randint(self.size // 4, self.size // 4 * 3, [self.n_dim]))
		#return np.array(np.random.randint(1, self.size - 1, [self.n_dim]))
		return np.array(np.random.randint(self.random_l, self.random_r, [self.n_dim]))

	def random_start_n(self):
		ball_list = []
		for i in range(self.n_ball):
			while True:
				new_ball = self.random_start()
				flag = True
				for j in range(i):
					if (new_ball == ball_list[j]).all():
						flag = False
				for j, t_state in enumerate(self.state_n):
					if (new_ball == t_state).all():
						flag = False
				if flag:
					break
			ball_list.append(new_ball)
		return ball_list

	def initialization(self, args):
		self.is_print = self.rank == 0

		self.args = args
		self.size = args.size
		self.map = np.zeros([self.size, self.size])
		self.dec_int = args.gamma_dec != 0
		self.penalty = args.penalty

		if (self.is_print):
			print(args.save_path)
			print('>>>>>>>>>>>>>dec_int', self.dec_int)

		self.n_agent = 2
		self.n_action = 5
		self.n_dim = 2

		self.random_l = args.pushball_random_l
		self.random_r = args.pushball_random_r

		self.reward_wall = [1000, 1000, 1000, 1000]

		self.state_n = [np.array([1, 1]) for _ in range(self.n_agent)]

		self.n_ball = 1
		self.ball_n = self.random_start_n()

		self.eye = np.eye(self.size)
		self.flag = np.eye(2)


		# Used by OpenAI baselines
		self.action_space = gym.spaces.Discrete(self.n_action)
		self.observation_space = gym.spaces.Box(low=-1, high=1, shape=[args.size * 2 * self.n_agent +
		                                                               args.size * 2 * self.n_ball])
		self.num_envs = args.num_env
		self.metadata = {'render.modes': []}
		self.reward_range = (-100., 20000.)
		self.spec = 2

		self.t_step = 0

	def step(self, action_n):

		self.t_step += 1

		ball_count = [np.zeros((5)) for i in range(self.n_ball)]

		for i, action in enumerate(action_n):

			new_row = -1
			new_column = -1

			if action == 0:
				new_row = max(self.state_n[i][0] - 1, 1)
				new_column = self.state_n[i][1]
			elif action == 1:
				new_row = self.state_n[i][0]
				new_column = min(self.state_n[i][1] + 1, self.size - 2)
			elif action == 2:
				new_row = min(self.state_n[i][0] + 1, self.size - 2)
				new_column = self.state_n[i][1]
			elif action == 3:
				new_row = self.state_n[i][0]
				new_column = max(self.state_n[i][1] - 1, 1)
			elif action == 4:
				new_row = self.state_n[i][0]
				new_column = self.state_n[i][1]

			for j, ball in enumerate(self.ball_n):
				if (self.state_n[i] != ball).any() and new_row == ball[0] and new_column == ball[1]:
					ball_count[j][action] += 1
					assert (action < 5)

		new_ball_n = []
		# Move Ball
		for i, ball in enumerate(self.ball_n):

			move_x = 0
			if ball_count[i][0] - ball_count[i][2] >= 2:
				move_x = -1
			if ball_count[i][2] - ball_count[i][0] >= 2:
				move_x = 1

			move_y = 0
			if ball_count[i][3] - ball_count[i][1] >= 2:
				move_y = -1
			if ball_count[i][1] - ball_count[i][3] >= 2:
				move_y = 1

			new_ball = np.array([ball[0] + move_x, ball[1] + move_y])

			flag = True
			for j, t_ball in enumerate(self.ball_n):
				if (new_ball == t_ball).all():
					flag = False
			for j, t_state in enumerate(self.state_n):
				if (new_ball == t_state).all():
					flag = False

			if flag:
				new_ball_n.append(new_ball)
			else:
				new_ball_n.append(ball)

		self.ball_n = new_ball_n

		for i, action in enumerate(action_n):

			new_row = -1
			new_column = -1

			if action == 0:
				new_row = max(self.state_n[i][0] - 1, 1)
				new_column = self.state_n[i][1]
			elif action == 1:
				new_row = self.state_n[i][0]
				new_column = min(self.state_n[i][1] + 1, self.size - 2)
			elif action == 2:
				new_row = min(self.state_n[i][0] + 1, self.size - 2)
				new_column = self.state_n[i][1]
			elif action == 3:
				new_row = self.state_n[i][0]
				new_column = max(self.state_n[i][1] - 1, 1)
			elif action == 4:
				new_row = self.state_n[i][0]
				new_column = self.state_n[i][1]

			flag = False
			for j, ball in enumerate(self.ball_n):
				if (self.state_n[i] != ball).any() and new_row == ball[0] and new_column == ball[1]:
					assert (action < 5)
					flag = True

			if flag:
				new_row = self.state_n[i][0]
				new_column = self.state_n[i][1]

			self.state_n[i] = np.array([new_row, new_column])

		ball_info = np.concatenate([[ball[0], ball[1]] for ball in self.ball_n], axis=None)
		info_state_n = []
		for i, state in enumerate(self.state_n):
			full_state = np.concatenate([state, ball_info], axis=0)
			info_state_n.append(full_state)

		info = {'ball': self.ball_n, 'state': copy.deepcopy(info_state_n)}

		return_obs = self.obs_n()
		return_rew, info_r = self.reward()
		return_done = self.done()

		info['rew'] = info_r

		return return_obs, return_rew, return_done, info

	def reset(self):
		self.t_step = 0
		self.state_n = [np.array([1, 1]) for _ in range(self.n_agent)]
		self.ball_n = self.random_start_n()
		return self.obs_n()

	def obs_n(self):
		return [self.obs(i) for i in range(self.n_agent)]

	def obs(self, i):
		ball = np.concatenate([np.concatenate([self.eye[ball[0]], self.eye[ball[1]]], axis=0)
							   for ball in self.ball_n], axis=0)
		return np.concatenate([self.eye[self.state_n[0][0]], self.eye[self.state_n[0][1]],
		                       self.eye[self.state_n[1][0]], self.eye[self.state_n[1][1]],
		                       ball]).copy()

	def reward(self):
		reward = 0
		win_count = np.zeros(4)
		for i, ball in enumerate(self.ball_n):
			pre_reward = reward
			if ball[0] == 0:
				reward += self.reward_wall[0]
				win_count[0] += 1
			if ball[1] == 0:
				reward += self.reward_wall[1]
				win_count[1] += 1
			if ball[0] == self.size - 1:
				reward += self.reward_wall[2]
				win_count[2] += 1
			if ball[1] == self.size - 1:
				reward += self.reward_wall[3]
				win_count[3] += 1
			if reward > pre_reward:
				while True:
					new_ball = self.random_start()
					flag = True
					for j, t_ball in enumerate(self.ball_n):
						if (new_ball == t_ball).all():
							flag = False
					for j, t_state in enumerate(self.state_n):
						if (new_ball == t_state).all():
							flag = False
					if flag:
						break
				self.ball_n[i] = new_ball

		return [reward, reward], win_count

	def done(self):
		if self.t_step >= self.args.episode_length:
			self.reset()
			return 1

		return 0

	def close(self):
		self.reset()