import numpy as np
from baselines.common.runners import AbstractEnvRunner
from baselines.Curiosity import Visual_coor_rew
from baselines.data_gather import Gather
import copy


class Runner():
	"""
	We use this object to make a mini batch of experiences
	__init__:
	- Initialize the runner

	run():
	- Make a mini batch
	"""

	def __init__(self, *, env, model_n, nsteps, gamma, lam):
		self.env = env
		self.model_n = model_n
		self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
		self.batch_ob_shape = (nenv * nsteps,) + env.observation_space.shape
		self.obs_n = env.reset().copy()
		self.nsteps = nsteps
		self.states = [model.initial_state for model in model_n]
		self.e_states = [model.e_initial_state for model in model_n]
		self.c_states = [model.c_initial_state for model in model_n]
		self.dones = [False for _ in range(nenv)]

		# Lambda used in GAE (General Advantage Estimation)
		self.lam = lam
		# Discount rate
		self.gamma = gamma
		self.n_agent = env.spec

		self.game_name = self.env.args.env
		self.num_env = self.env.args.num_env
		if self.env.args.s_data_gather:
			self.data_gather = Gather(self.env.args)

		if self.env.args.t or self.env.args.tv or self.env.args.r_tv:
			self.visual_t = Visual_coor_rew(self.env.args.size, self.n_agent, self.env.args, self.env.num_envs, 't')
		if self.env.args.r or self.env.args.r_tv:
			self.visual_r = Visual_coor_rew(self.env.args.size, self.n_agent, self.env.args, self.env.num_envs, 'r')
		if self.env.args.tv or self.env.args.r_tv:
			self.visual_tv = Visual_coor_rew(self.env.args.size, self.n_agent, self.env.args, self.env.num_envs, 'tv')
		if self.env.args.v:
			self.visual_v = Visual_coor_rew(self.env.args.size, self.n_agent, self.env.args, self.env.num_envs, 'v')
		if self.env.args.r_tv:
			self.visual_r_tv = Visual_coor_rew(self.env.args.size, self.n_agent, self.env.args, self.env.num_envs,
			                                   'r_tv')
		if self.env.args.t or self.env.args.r or self.env.args.tv or self.env.args.r_tv:
			self.visual_all = Visual_coor_rew(self.env.args.size, self.n_agent, self.env.args, self.env.num_envs,
			                                  'all')

	def ma_reshape(self, obs, i):
		obs = np.array(obs)
		index = [i for i in range(len(obs.shape))]
		index[0] = i
		index[i] = 0
		return np.transpose(obs, index).copy()

	def get_coor(self, coor_rewards_show, i):
		if self.env.args.env == 'x_island_old' or self.env.args.env == 'x_pass':
			return coor_rewards_show[i]
		else:
			return coor_rewards_show

	def get_coor_mean(self, coor_rewards_show):
		if self.env.args.env == 'x_island_old' or self.env.args.env == 'x_pass':
			return np.sum(coor_rewards_show, axis=1)
		else:
			return coor_rewards_show

	def run(self):
		mb_obs_n, mb_rewards_n, mb_e_rewards_n, mb_c_rewards_n, mb_actions_n, mb_values_n, \
		mb_e_values_n, mb_c_values_n, mb_dones_n, mb_neglogpacs_n, mb_e_neglogpacs_n, mb_c_neglogpacs_n = \
			[[] for _ in range(self.n_agent)], \
			[[] for _ in range(self.n_agent)], \
			[[] for _ in range(self.n_agent)], \
			[[] for _ in range(self.n_agent)], \
			[[] for _ in range(self.n_agent)], \
			[[] for _ in range(self.n_agent)], \
			[[] for _ in range(self.n_agent)], \
			[[] for _ in range(self.n_agent)], \
			[[] for _ in range(self.n_agent)], \
			[[] for _ in range(self.n_agent)], \
			[[] for _ in range(self.n_agent)], \
			[[] for _ in range(self.n_agent)]

		# Here, we init the lists that will contain the mb of experiences
		mb_states = self.states
		mb_e_states = self.e_states
		mb_c_states = self.c_states
		epinfos = []
		infos_list = []
		all_rewards = []
		e_rewards = []
		d_rewards = []
		ce_rewards = []
		c_rewards_show = []
		c_rewards_t = []
		c_rewards_r = []
		c_rewards_v = []
		ext_rewards_v_n = []
		int_rewards_v_n = []
		c_rewards_tv = []
		ext_rewards_tv_n = []
		int_rewards_tv_n = []
		c_rewards_all = []
		p_rewards = []
		coor_t_list = []
		coor_p_list = []

		t_rewards_list = []
		done_list = []

		# For n in range number of steps
		for _ in range(self.nsteps):
			# Given observations, get action value and neglopacs
			# We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
			actions = []
			for agent_i in range(self.n_agent):
				action, values, state, neglogpacs = self.model_n[agent_i].step(self.obs_n[agent_i],
				                                                               S=self.states[agent_i],
				                                                               M=self.dones)
				actions.append(action)
				self.states[agent_i] = state
				mb_obs_n[agent_i].append(self.obs_n[agent_i].copy())
				mb_actions_n[agent_i].append(action)
				mb_values_n[agent_i].append(values)
				mb_neglogpacs_n[agent_i].append(neglogpacs)
				mb_dones_n[agent_i].append(self.dones)

				e_action, e_values, e_state, e_neglogpacs = self.model_n[agent_i].e_step(self.obs_n[agent_i],
				                                                                              S=self.e_states[agent_i],
				                                                                              M=self.dones)
				self.e_states[agent_i] = e_state
				mb_e_values_n[agent_i].append(e_values)
				mb_e_neglogpacs_n[agent_i].append(e_neglogpacs)

				c_action, c_values, c_state, c_neglogpacs = self.model_n[agent_i].c_step(self.obs_n[agent_i],
				                                                                         S=self.c_states[agent_i],
				                                                                         M=self.dones)
				self.c_states[agent_i] = c_state
				mb_c_values_n[agent_i].append(c_values)
				mb_c_neglogpacs_n[agent_i].append(c_neglogpacs)

			# Take actions in env and look the results
			# Infos contains a ton of useful informations

			actions = self.ma_reshape(actions, 1)
			self.obs_n, t_rewards, self.dones, infos = self.env.step(actions)
			infos_list.append(infos)
			t_rewards_list.append(t_rewards)
			done_list.append(self.dones)

		# batch of steps to batch of rollouts
		mb_obs_n = [np.asarray(mb_obs, dtype=self.obs_n[0].dtype) for mb_obs in mb_obs_n]
		mb_actions_n = [np.asarray(mb_actions) for mb_actions in mb_actions_n]
		mb_values_n = [np.asarray(mb_values, dtype=np.float32) for mb_values in mb_values_n]
		mb_e_values_n = [np.asarray(mb_e_values, dtype=np.float32) for mb_e_values in mb_e_values_n]
		mb_c_values_n = [np.asarray(mb_c_values, dtype=np.float32) for mb_c_values in mb_c_values_n]
		mb_neglogpacs_n = [np.asarray(mb_neglogpacs, dtype=np.float32) for mb_neglogpacs in mb_neglogpacs_n]
		mb_e_neglogpacs_n = [np.asarray(mb_e_neglogpacs, dtype=np.float32) for mb_e_neglogpacs in
		                      mb_e_neglogpacs_n]
		mb_c_neglogpacs_n = [np.asarray(mb_c_neglogpacs, dtype=np.float32) for mb_c_neglogpacs in
		                     mb_c_neglogpacs_n]
		mb_dones_n = [np.asarray(mb_dones, dtype=np.bool) for mb_dones in mb_dones_n]
		last_values = [model.value(self.obs_n[i_m], S=self.states[i_m], M=self.dones) for i_m, model in
		               enumerate(self.model_n)]
		e_last_values = [model.e_value(self.obs_n[i_m], S=self.e_states[i_m], M=self.dones) for i_m, model in
		                  enumerate(self.model_n)]
		c_last_values = [model.c_value(self.obs_n[i_m], S=self.c_states[i_m], M=self.dones) for i_m, model in
		                 enumerate(self.model_n)]

		# success rate
		kill_list = []
		landmark_list = []
		time_length_list = [[] for i in range(self.n_agent)]
		death_list = [[] for i in range(self.n_agent)]
		win_list = [[] for _ in range(4)]

		# For n in range number of steps
		for _ in range(self.nsteps):

			if self.env.args.env == 'x_pass' or self.env.args.env == 'x_island_old':
				ext_rewards, dec_rewards, coor_rewards_show, penalty_rewards, cen_rewards, coor_p, coor_t = \
					t_rewards_list[_]
			else:
				ext_rewards, dec_rewards, coor_rewards_show, penalty_rewards, cen_rewards = t_rewards_list[_]
				coor_p = np.zeros_like(ext_rewards)
				coor_t = np.zeros_like(ext_rewards)
			coor_rewards_t = copy.deepcopy(ext_rewards)
			coor_rewards_r = copy.deepcopy(ext_rewards)
			coor_rewards_tv = copy.deepcopy(ext_rewards)
			coor_rewards_v = copy.deepcopy(ext_rewards)
			ext_rewards_v = copy.deepcopy(ext_rewards)
			int_rewards_v = copy.deepcopy(ext_rewards)
			ext_rewards_tv = copy.deepcopy(ext_rewards)
			int_rewards_tv = copy.deepcopy(ext_rewards)

			infos = infos_list[_]
			dones = done_list[_]

			if self.env.args.v or self.env.args.r_tv:
				for agent_i, next_obs in enumerate(self.obs_n):
					t_s = 0.
					ext_t_s = 0.
					int_t_s = 0.
					for i in range(self.n_agent):
						if agent_i != i:
							if _ + 1 == self.nsteps:
								ext_v = e_last_values[i]
								int_v = c_last_values[i]
							else:
								ext_v = mb_e_values_n[i][_ + 1]
								int_v = mb_c_values_n[i][_ + 1]
							t_s += self.env.args.gamma_coor_v_e * ext_v + self.env.args.gamma_coor_v_c * int_v
							ext_t_s += ext_v
							int_t_s += int_v
					coor_rewards_v[agent_i] = self.env.args.gamma_coor_v * self.gamma * t_s
					ext_rewards_v[agent_i] = self.gamma * ext_t_s
					int_rewards_v[agent_i] = self.gamma * int_t_s
				if self.env.args.v:
					self.visual_v.update(infos, dones, coor_rewards_v)
			else:
				coor_rewards_v *= 0
				ext_rewards_v *= 0
				int_rewards_v *= 0

			if self.env.args.r or self.env.args.r_tv:
				for agent_i, next_obs in enumerate(self.obs_n):
					t_s = 0.
					for i in range(self.n_agent):
						if agent_i != i:
							t_s += self.env.args.gamma_coor_r_e * ext_rewards[i] + \
								   self.env.args.gamma_coor_r_c * dec_rewards[i]
					coor_rewards_r[agent_i] = \
						self.env.args.gamma_coor_r * t_s
				self.visual_r.update(infos, dones, coor_rewards_r)
			else:
				coor_rewards_r *= 0

			if self.env.args.tv or self.env.args.r_tv:
				for agent_i, next_obs in enumerate(self.obs_n):
					t_s = 0.
					ext_t_s = 0.
					int_t_s = 0.
					for i in range(self.n_agent):
						if agent_i != i:
							if _ + 1 == self.nsteps:
								ext_v = e_last_values[i]
								int_v = c_last_values[i]
							else:
								ext_v = mb_e_values_n[i][_ + 1]
								int_v = mb_c_values_n[i][_ + 1]
							t_s += self.get_coor(coor_rewards_show[agent_i], i) * \
							       (self.env.args.gamma_coor_tv_e * ext_v + self.env.args.gamma_coor_tv_c * int_v)
							ext_t_s += self.get_coor(coor_rewards_show[agent_i], i) * ext_v
							int_t_s += self.get_coor(coor_rewards_show[agent_i], i) * int_v
					coor_rewards_tv[agent_i] = self.env.args.gamma_coor_tv * self.gamma * t_s
					ext_rewards_tv[agent_i] = self.env.args.gamma_coor_tv * self.gamma * ext_t_s
					int_rewards_tv[agent_i] = self.env.args.gamma_coor_tv * self.gamma * int_t_s
				self.visual_tv.update(infos, dones, coor_rewards_tv)
			else:
				coor_rewards_tv *= 0
				ext_rewards_tv *= 0
				int_rewards_tv *= 0

			if self.env.args.t or self.env.args.tv or self.env.args.r_tv:
				for agent_i, next_obs in enumerate(self.obs_n):
					t_s = 0.
					for i in range(self.n_agent):
						if agent_i != i:
							t_s += self.get_coor(coor_rewards_show[agent_i], i)
					coor_rewards_t[agent_i] = \
						self.env.args.gamma_coor_t * t_s
				self.visual_t.update(infos, dones, coor_rewards_t)
			else:
				coor_rewards_t *= 0

			if self.env.args.r_tv:
				self.visual_r_tv.update(infos, dones, coor_rewards_r + coor_rewards_tv)

			coor_rewards_all = coor_rewards_t + coor_rewards_r + coor_rewards_tv + coor_rewards_v
			rewards = ext_rewards + dec_rewards + cen_rewards + coor_rewards_all + penalty_rewards

			if self.env.args.t or self.env.args.r or self.env.args.tv or self.env.args.r_tv:
				self.visual_all.update(infos, dones, coor_rewards_all)

			all_rewards.append(rewards)
			epinfos.append(ext_rewards[0])
			e_rewards.append(ext_rewards)
			d_rewards.append(dec_rewards)
			ce_rewards.append(cen_rewards)
			c_rewards_show.append(self.get_coor_mean(coor_rewards_show))
			c_rewards_t.append(coor_rewards_t)
			c_rewards_r.append(coor_rewards_r)
			c_rewards_v.append(coor_rewards_v)
			ext_rewards_v_n.append(ext_rewards_v)
			int_rewards_v_n.append(int_rewards_v)
			c_rewards_tv.append(coor_rewards_tv)
			c_rewards_all.append(coor_rewards_all)
			p_rewards.append(penalty_rewards[0])
			ext_rewards_tv_n.append(ext_rewards_tv)
			int_rewards_tv_n.append(int_rewards_tv)
			coor_t_list.append(coor_t)
			coor_p_list.append(coor_p)

			for agent_i in range(self.n_agent):
				mb_rewards_n[agent_i].append(rewards[agent_i, :])
				mb_e_rewards_n[agent_i].append(ext_rewards[agent_i, :])
				mb_c_rewards_n[agent_i].append(dec_rewards[agent_i, :])

			if self.game_name == 'x_island' or self.game_name == 'island':
				for k in range(self.num_env):
					info_r = infos[k]['rew']
					kill_list.append(info_r['kill'])
					landmark_list.append(info_r['landmark'])
					for j, death in enumerate(info_r['death']):
						death_list[j].append(int(death))
						time_length_list[j].append(info_r['time_length'][j])

			if self.game_name == 'pushball':
				for k in range(self.num_env):
					info_r = infos[k]['rew']
					for i in range(4):
						win_list[i].append(info_r[i])

		if self.env.args.s_data_gather:
			t_data = (ext_rewards_v_n, int_rewards_v_n, ext_rewards_tv_n, int_rewards_tv_n, all_rewards,
			          e_rewards, d_rewards, c_rewards_show, c_rewards_t, c_rewards_r, c_rewards_v,
			          c_rewards_tv, c_rewards_all, ce_rewards, mb_dones_n, infos_list,
			          kill_list, landmark_list, time_length_list, death_list, win_list, coor_p_list, coor_t_list)
			t_data = copy.deepcopy(t_data)
			self.data_gather.update(t_data)

		# rewards_n
		mb_rewards_n = [np.asarray(mb_rewards, dtype=np.float32) for mb_rewards in mb_rewards_n]
		mb_e_rewards_n = [np.asarray(mb_e_rewards, dtype=np.float32) for mb_e_rewards in mb_e_rewards_n]
		mb_c_rewards_n = [np.asarray(mb_c_rewards, dtype=np.float32) for mb_c_rewards in mb_c_rewards_n]

		# discount/bootstrap off value fn
		mb_returns_n = [np.zeros_like(mb_rewards) for mb_rewards in mb_rewards_n]
		mb_advs_n = [np.zeros_like(mb_rewards) for mb_rewards in mb_rewards_n]
		lastgaelam = [0 for _ in range(self.n_agent)]
		for t in reversed(range(self.nsteps)):
			for agent_i in range(self.n_agent):
				if t == self.nsteps - 1:
					nextnonterminal = 1.0 - self.dones[0]
					nextvalues = last_values[agent_i]
				else:
					nextnonterminal = 1.0 - mb_dones_n[agent_i][t + 1][0]
					nextvalues = mb_values_n[agent_i][t + 1]
				delta = mb_rewards_n[agent_i][t] + self.gamma * nextvalues * nextnonterminal - mb_values_n[agent_i][t]
				mb_advs_n[agent_i][t] = lastgaelam[agent_i] = delta + \
				                                              self.gamma * self.lam * \
				                                              nextnonterminal * lastgaelam[agent_i]

		for agent_i in range(self.n_agent):
			mb_returns_n[agent_i] = mb_advs_n[agent_i] + mb_values_n[agent_i]

		# discount/bootstrap off extrinsic and curiosity value fn
		mb_e_returns_n = [np.zeros_like(mb_e_rewards) for mb_e_rewards in mb_e_rewards_n]
		mb_e_advs_n = [np.zeros_like(mb_e_rewards) for mb_e_rewards in mb_e_rewards_n]
		e_lastgaelam = [0 for _ in range(self.n_agent)]
		for t in reversed(range(self.nsteps)):
			for agent_i in range(self.n_agent):
				if t == self.nsteps - 1:
					e_nextnonterminal = 1.0 - self.dones[0]
					e_nextvalues = e_last_values[agent_i]
				else:
					e_nextnonterminal = 1.0 - mb_dones_n[agent_i][t + 1][0]
					e_nextvalues = mb_e_values_n[agent_i][t + 1]
				e_delta = mb_e_rewards_n[agent_i][t] + self.gamma * e_nextvalues * e_nextnonterminal - \
				           mb_e_values_n[agent_i][t]
				mb_e_advs_n[agent_i][t] = e_lastgaelam[agent_i] = e_delta + \
				                                                    self.gamma * self.lam * \
				                                                    e_nextnonterminal * e_lastgaelam[agent_i]

		for agent_i in range(self.n_agent):
			mb_e_returns_n[agent_i] = mb_e_advs_n[agent_i] + mb_e_values_n[agent_i]

		#-------------------------------------------

		mb_c_returns_n = [np.zeros_like(mb_c_rewards) for mb_c_rewards in mb_c_rewards_n]
		mb_c_advs_n = [np.zeros_like(mb_c_rewards) for mb_c_rewards in mb_c_rewards_n]
		c_lastgaelam = [0 for _ in range(self.n_agent)]
		for t in reversed(range(self.nsteps)):
			for agent_i in range(self.n_agent):
				if t == self.nsteps - 1:
					c_nextnonterminal = 1.0 - self.dones[0]
					c_nextvalues = c_last_values[agent_i]
				else:
					c_nextnonterminal = 1.0 - mb_dones_n[agent_i][t + 1][0]
					c_nextvalues = mb_c_values_n[agent_i][t + 1]
				c_delta = mb_c_rewards_n[agent_i][t] + self.gamma * c_nextvalues * c_nextnonterminal - \
				           mb_c_values_n[agent_i][t]
				mb_c_advs_n[agent_i][t] = c_lastgaelam[agent_i] = c_delta + \
				                                                    self.gamma * self.lam * \
				                                                    c_nextnonterminal * c_lastgaelam[agent_i]

		for agent_i in range(self.n_agent):
			mb_c_returns_n[agent_i] = mb_c_advs_n[agent_i] + mb_c_values_n[agent_i]

		return (*map(sf01, (mb_obs_n, mb_returns_n, mb_dones_n, mb_actions_n, mb_values_n, mb_neglogpacs_n)),
		        *map(sf01, (mb_e_returns_n, mb_e_values_n, mb_e_neglogpacs_n)),
		        *map(sf01, (mb_c_returns_n, mb_c_values_n, mb_c_neglogpacs_n)),
		        mb_states, mb_e_states, mb_c_states, np.array(epinfos), np.array(all_rewards), np.array(d_rewards),
		        np.array(c_rewards_show), np.array(c_rewards_t), np.array(c_rewards_r), np.array(c_rewards_v),
		        np.array(c_rewards_tv), np.array(c_rewards_all), np.array(p_rewards), np.array(ce_rewards),
		        np.array(ext_rewards_tv_n), np.array(int_rewards_tv_n),
		        np.array(ext_rewards_v_n), np.array(int_rewards_v_n))


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(l):
	"""
	swap and then flatten axes 0 and 1
	"""
	for i, arr in enumerate(l):
		s = arr.shape
		l[i] = arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

	return l
