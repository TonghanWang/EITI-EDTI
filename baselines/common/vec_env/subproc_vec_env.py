import multiprocessing as mp

import numpy as np
from .vec_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars
from baselines.Curiosity import Dec, Pushball_Dec, C_points, Cen, Appro_C_points, Pushball_Appro_C_points, \
	Island_Dec, Island_Appro_C_points, Island_Cen, Island_VI_Appro_C_points, Test_Island_Appro_C_points, x_Island_Cen, \
	Pushball_Cen, Fast_Pass_VI_Appro_C_points, Fast_Island_VI_Appro_C_points, x_Island_Appro_C_points, \
	Pushball_Appro_C_points_2
import pickle
import matplotlib.pyplot as plt
import copy


def worker(remote, parent_remote, env_fn_wrapper):
	parent_remote.close()
	env = env_fn_wrapper.x()
	try:
		while True:
			cmd, data = remote.recv()
			if cmd == 'step':
				ob, reward, done, info = env.step(data)
				if done:
					ob = env.reset()
				remote.send((ob, reward, done, info))
			elif cmd == 'reset':
				ob = env.reset()
				remote.send(ob)
			elif cmd == 'render':
				remote.send(env.render(mode='rgb_array'))
			elif cmd == 'close':
				remote.close()
				break
			elif cmd == 'get_spaces_spec':
				remote.send((env.observation_space, env.action_space, env.spec))
			else:
				raise NotImplementedError
	except KeyboardInterrupt:
		print('SubprocVecEnv worker: got KeyboardInterrupt')
	finally:
		env.close()


class SubprocVecEnv(VecEnv):
	"""
	VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
	Recommended to use when num_envs > 1 and step() can be a bottleneck.
	"""

	def __init__(self, env_fns, spaces=None, context='spawn'):
		"""
		Arguments:

		env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
		"""
		self.waiting = False
		self.closed = False
		nenvs = len(env_fns)
		ctx = mp.get_context(context)
		self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(nenvs)])
		self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
		           for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
		for p in self.ps:
			p.daemon = True  # if the main process crashes, we should not cause things to hang
			with clear_mpi_env_vars():
				p.start()
		for remote in self.work_remotes:
			remote.close()

		self.remotes[0].send(('get_spaces_spec', None))
		observation_space, action_space, self.spec = self.remotes[0].recv()
		self.viewer = None
		VecEnv.__init__(self, len(env_fns), observation_space, action_space)

	def step_async(self, actions):
		self._assert_not_closed()
		for remote, action in zip(self.remotes, actions):
			remote.send(('step', action))
		self.waiting = True

	def step_wait(self):
		self._assert_not_closed()
		results = [remote.recv() for remote in self.remotes]
		self.waiting = False
		obs, rews, dones, infos = zip(*results)
		return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

	def reset(self):
		self._assert_not_closed()
		for remote in self.remotes:
			remote.send(('reset', None))
		return _flatten_obs([remote.recv() for remote in self.remotes])

	def close_extras(self):
		self.closed = True
		if self.waiting:
			for remote in self.remotes:
				remote.recv()
		for remote in self.remotes:
			remote.send(('close', None))
		for p in self.ps:
			p.join()

	def get_images(self):
		self._assert_not_closed()
		for pipe in self.remotes:
			pipe.send(('render', None))
		imgs = [pipe.recv() for pipe in self.remotes]
		return imgs

	def _assert_not_closed(self):
		assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

	def __del__(self):
		if not self.closed:
			self.close()


class SubprocVecEnv_Pass(SubprocVecEnv):
	"""
	VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
	Recommended to use when num_envs > 1 and step() can be a bottleneck.
	"""

	def __init__(self, env_fns, args):
		SubprocVecEnv.__init__(self, env_fns)
		# For debugging
		self.hot_map_save_path = args.save_path
		self.ext_rewards = []
		self.args = args
		self.dec = Dec(self.args.size, self.args.n_agent, self.args)
		self.cen = Cen(self.args.size, self.args.n_agent, self.args)
		self.key_points = Appro_C_points(self.args.size, self.args.n_action, self.args, is_print=True)
		self.pre_state_n = [None for i in range(self.num_envs)]
		self.e_step = 0

	def dec_curiosity(self, state, i):
		return self.dec.output(state, i)

	def coor_curiosity(self, data_1, data_2, i):
		return self.key_points.output(data_1, data_2, i)

	def ma_reshape(self, obs, i):
		obs = np.array(obs)
		index = [i for i in range(len(obs.shape))]
		index[0] = i
		index[i] = 0
		return np.transpose(obs, index).copy()

	def smooth(self, data):

		smoothed = []

		for i in range(len(data)):
			smoothed.append(np.mean(data[max(0, i + 1 - 100): i + 1]))

		return smoothed

	def save_results(self, er2, name, arg):
		filename = '%s/%s_data.pkl' % (arg.save_path, name)
		with open(filename, 'wb') as file:
			pickle.dump(er2, file)

		# er3 = q_cen_for_pass()

		# er1_s = smooth(er1)
		er2_s = self.smooth(er2)
		# er3_s = smooth(er3)

		m_figure = plt.figure()
		m_ax1 = m_figure.add_subplot(3, 2, 1)
		m_ax2 = m_figure.add_subplot(3, 2, 2)
		m_ax3 = m_figure.add_subplot(3, 2, 3)
		m_ax4 = m_figure.add_subplot(3, 2, 4)
		m_ax5 = m_figure.add_subplot(3, 2, 5)
		m_ax6 = m_figure.add_subplot(3, 2, 6)

		# m_ax1.plot(er1_s)
		# m_ax2.plot(er1)
		m_ax3.plot(er2_s)
		m_ax4.plot(er2)
		# m_ax5.plot(er3_s)
		# m_ax6.plot(er3)

		m_ax1.legend(['epsilon-greedy'])
		m_ax2.legend(['epsilon-greedy (unsmoothed)'])
		m_ax3.legend(['dec-curiosity'])
		m_ax4.legend(['dec-curiosity (unsmoothed)'])
		m_ax5.legend(['cen-curiosity'])
		m_ax6.legend(['cen-curiosity (unsmoothed)'])

		m_figure.savefig('%s/%s_i Boltzmann.png' % (arg.save_path, name))
		plt.close(m_figure)

	def reset(self):
		self._assert_not_closed()
		for remote in self.remotes:
			remote.send(('reset', None))
		obs_n = _flatten_obs([remote.recv() for remote in self.remotes])
		obs_n = self.ma_reshape(obs_n, 1)
		return obs_n

	def step(self, actions):
		self.step_async(actions)
		obs_n, ext_rewards, dones, infos = self.step_wait()
		obs_n = self.ma_reshape(obs_n, 1)
		ext_rewards = self.ma_reshape(ext_rewards, 1).astype('float32')
		state_n = []
		for i in range(self.num_envs):
			state_n.append(copy.deepcopy(infos[i]['state']))
			infos[i]['pre_state'] = self.pre_state_n[i]

		# estimate coor
		for j in range(self.num_envs):
			if self.pre_state_n[j] != None:
				self.key_points.update([self.pre_state_n[j], actions[j]], [state_n[j], None])

		# add intrinsic rew
		dec_rewards = ext_rewards.copy()
		coor_rewards = ext_rewards.copy()
		penalty_rewards = ext_rewards.copy()
		cen_rewards = ext_rewards.copy()
		for i in range(self.args.n_agent):
			for j in range(self.num_envs):
				dec_rewards[i, j] = self.args.gamma_dec * self.dec_curiosity(state_n[j], i)
				if self.pre_state_n[j] != None:
					coor_rewards[i, j] = self.coor_curiosity([self.pre_state_n[j], actions[j]], [state_n[j], None], i)
				else:
					coor_rewards[i, j] = 0.
				cen_rewards[i, j] = self.args.gamma_cen * self.cen.output(state_n[j])
				penalty_rewards[i, j] = self.args.penalty
		# update intrinsic rew
		for j in range(self.num_envs):
			self.dec.update(state_n[j], infos[j]['door'])
			self.cen.update(state_n[j])

		self.pre_state_n = state_n
		for i in range(self.num_envs):
			if dones[i]:
				self.pre_state_n[i] = None

		# debug
		for i in range(self.num_envs):
			if dones[i]:
				self.e_step += 1
				self.ext_rewards.append(ext_rewards[0, i])

				s_rate = self.args.t_save_rate
				if (self.e_step + 1) % (1000 * s_rate) == 0:
					print(self.e_step + 1, float(sum(self.ext_rewards[-1000 * s_rate:])) / (1000.0 * s_rate))
					self.dec.show(self.hot_map_save_path, self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.key_points.show(self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.save_results(self.ext_rewards, '%d' % (self.e_step + 1), self.args)

		return obs_n, (ext_rewards, dec_rewards, coor_rewards, penalty_rewards, cen_rewards), dones, infos


class SubprocVecEnv_x_Pass(SubprocVecEnv_Pass):
	"""
	VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
	Recommended to use when num_envs > 1 and step() can be a bottleneck.
	"""

	def __init__(self, env_fns, args):
		SubprocVecEnv_Pass.__init__(self, env_fns, args)
		self.n_agent = self.args.n_agent
		self.key_points = Fast_Pass_VI_Appro_C_points(self.args.size, self.args.n_action, self.args, is_print=True)
		self.t_step = 0
		self.flag = True
		self.flag_2 = True
		self.flag_3 = True

	def step(self, actions):
		self.step_async(actions)
		obs_n, ext_rewards, dones, infos = self.step_wait()
		obs_n = self.ma_reshape(obs_n, 1)
		ext_rewards = self.ma_reshape(ext_rewards, 1).astype('float32')
		state_n = []
		for i in range(self.num_envs):
			state_n.append(copy.deepcopy(infos[i]['state']))
			infos[i]['pre_state'] = self.pre_state_n[i]

		# add intrinsic rew
		dec_rewards = ext_rewards.copy()
		penalty_rewards = ext_rewards.copy()
		cen_rewards = ext_rewards.copy()

		for i in range(self.args.n_agent):
			for j in range(self.num_envs):
				dec_rewards[i, j] = self.args.gamma_dec * self.dec_curiosity(state_n[j], i)
				cen_rewards[i, j] = self.args.gamma_cen * self.cen.output(state_n[j])
				penalty_rewards[i, j] = self.args.penalty

		coor_rewards = np.zeros((self.n_agent, self.n_agent, self.num_envs), dtype='float32')
		coor_p = np.zeros((self.n_agent, self.n_agent, self.num_envs), dtype='float32')
		coor_t = np.zeros((self.n_agent, self.n_agent, self.num_envs), dtype='float32')
		coor_p_show = np.zeros((self.n_agent, self.num_envs), dtype='float32')
		coor_t_show = np.zeros((self.n_agent, self.num_envs), dtype='float32')
		err_p = np.zeros((self.n_agent, self.n_agent, self.num_envs), dtype='float32')
		err_t = np.zeros((self.n_agent, self.n_agent, self.num_envs), dtype='float32')

		if not self.key_points.not_run:

			C_label = []
			C_x_p = []
			C_x_t = []
			for j in range(self.num_envs):
				if self.pre_state_n[j] != None:
					label, x_p, x_t = self.key_points.make([self.pre_state_n[j], actions[j]], [state_n[j], None])
					C_label.append(label)
					C_x_p.append(x_p)
					C_x_t.append(x_t)

			num_data = len(C_label)

			if num_data > 0:

				C_label = np.transpose(np.array(C_label), (1, 0))
				C_x_p = np.transpose(np.array(C_x_p), (1, 0, 2))
				C_x_t = np.transpose(np.array(C_x_t), (1, 0, 2))
				# coor_output, t_coor_p, t_coor_t, t_coor_rew_show, err_p_show, err_t_show, out_p_show, out_t_show = \
				#	self.key_points.output(C_label, C_x_p, C_x_t)
				coor_output, t_coor_p, t_coor_t, t_coor_rew_show, err_p_show, err_t_show = \
					self.key_points.output(C_label, C_x_p, C_x_t)

				tt_stamp = 0
				for k in range(self.num_envs):
					if self.pre_state_n[k] != None:
						coor_rewards[:, :, k] = coor_output[:, :, tt_stamp]
						coor_p[:, :, k] = t_coor_p[:, :, tt_stamp]
						coor_t[:, :, k] = t_coor_t[:, :, tt_stamp]
						err_p[:, :, k] = err_p_show[:, :, tt_stamp]
						err_t[:, :, k] = err_t_show[:, :, tt_stamp]
						for i in range(self.n_agent):
							# receive reward
							coor_p_show[i, k] = 1. - t_coor_p[i, 1 - i, tt_stamp]
							coor_t_show[i, k] = 1. - t_coor_t[i, 1 - i, tt_stamp]
						tt_stamp += 1
				tt_stamp = 0
				for k in range(self.num_envs):
					if self.pre_state_n[k] != None:
						for i in range(self.n_agent):
							# check i
							if self.flag and abs(t_coor_rew_show[1 - i, i, tt_stamp]) > 0.5:
								self.flag = False
								print(self.pre_state_n[k], state_n[k], actions[k],
								      C_label[i, tt_stamp], k, tt_stamp, i,
								      round(coor_p[1 - i, i, k], 5), round(coor_t[1 - i, i, k], 5),
								      round(err_p[1 - i, i, k], 3), round(err_t[1 - i, i, k], 3),
								      round(1 - t_coor_rew_show[1 - i, i, tt_stamp], 3))
							# print(self.pre_state_n[:5])
							# print(state_n[:5])
							# print(np.round(coor_p[1 - i, i, :5], decimals=5))
							# print(np.round(err_p[1 - i, i, :5], decimals=3))
							if self.flag_2 and state_n[k][i][1] == self.args.size // 2:
								self.flag_2 = False
								print('pass door:', self.pre_state_n[k], state_n[k], actions[k],
								      C_label[i, tt_stamp], k, tt_stamp, i,
								      round(coor_p[1 - i, i, k], 5), round(coor_t[1 - i, i, k], 5),
								      round(err_p[1 - i, i, k], 3), round(err_t[1 - i, i, k], 3),
								      round(1 - t_coor_rew_show[1 - i, i, tt_stamp], 3))
							if self.flag_3 and self.pre_state_n[k][i][1] == self.args.size // 2 - 1 and \
									self.pre_state_n[k][i][0] >= int(self.args.size * 0.45) and \
									self.pre_state_n[k][i][0] < int(self.args.size * 0.55) and actions[k][i] == 1:
								self.flag_3 = False
								print('near door:', self.pre_state_n[k], state_n[k], actions[k],
								      C_label[i, tt_stamp], k, tt_stamp, i,
								      round(coor_p[1 - i, i, k], 5), round(coor_t[1 - i, i, k], 5),
								      round(err_p[1 - i, i, k], 3), round(err_t[1 - i, i, k], 3),
								      round(1 - t_coor_rew_show[1 - i, i, tt_stamp], 3))
						tt_stamp += 1

		# update intrinsic rew
		for j in range(self.num_envs):
			self.dec.update(state_n[j], infos[j]['door'])
			self.cen.update(state_n[j])

		self.pre_state_n = state_n
		for i in range(self.num_envs):
			if dones[i]:
				self.pre_state_n[i] = None

		# debug
		for i in range(self.num_envs):
			if dones[i]:
				self.e_step += 1
				self.ext_rewards.append(ext_rewards[0, i])

				s_rate = self.args.t_save_rate
				if (self.e_step + 1) % (1000 * s_rate) == 0:
					self.flag = True
					self.flag_2 = True
					self.flag_3 = True
					print(self.e_step + 1, float(sum(self.ext_rewards[-1000 * s_rate:])) / (1000.0 * s_rate))
					self.dec.show(self.hot_map_save_path, self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.key_points.show(self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.save_results(self.ext_rewards, '%d' % (self.e_step + 1), self.args)

		return obs_n, (ext_rewards, dec_rewards, coor_rewards, penalty_rewards, cen_rewards,
		               coor_p_show, coor_t_show), dones, infos


class SubprocVecEnv_ThreePass(SubprocVecEnv_Pass):
	"""
	VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
	Recommended to use when num_envs > 1 and step() can be a bottleneck.
	"""

	def __init__(self, env_fns, args):
		SubprocVecEnv_Pass.__init__(self, env_fns, args)

	def step(self, actions):
		self.step_async(actions)
		obs_n, ext_rewards, dones, infos = self.step_wait()
		obs_n = self.ma_reshape(obs_n, 1)
		ext_rewards = self.ma_reshape(ext_rewards, 1).astype('float32')
		state_n = []
		for i in range(self.num_envs):
			state_n.append(copy.deepcopy(infos[i]['state']))
			infos[i]['pre_state'] = self.pre_state_n[i]

		# estimate coor
		for j in range(self.num_envs):
			if self.pre_state_n[j] != None:
				self.key_points.update([self.pre_state_n[j], actions[j]], [state_n[j], None])

		# add intrinsic rew
		dec_rewards = ext_rewards.copy()
		coor_rewards = ext_rewards.copy()
		penalty_rewards = ext_rewards.copy()
		cen_rewards = ext_rewards.copy()
		for i in range(self.args.n_agent):
			for j in range(self.num_envs):
				dec_rewards[i, j] = self.args.gamma_dec * self.dec_curiosity(state_n[j], i)
				if self.pre_state_n[j] != None:
					coor_rewards[i, j] = self.coor_curiosity([self.pre_state_n[j], actions[j]], [state_n[j], None], i)
				else:
					coor_rewards[i, j] = 0.
				cen_rewards[i, j] = self.args.gamma_cen * self.cen.output(state_n[j])
				penalty_rewards[i, j] = self.args.penalty
		# update intrinsic rew
		for j in range(self.num_envs):
			self.dec.update(state_n[j], np.array(infos[j]['door']).any())
			self.cen.update(state_n[j])

		self.pre_state_n = state_n
		for i in range(self.num_envs):
			if dones[i]:
				self.pre_state_n[i] = None

		# debug
		for i in range(self.num_envs):
			if dones[i]:
				self.e_step += 1
				self.ext_rewards.append(ext_rewards[0, i])

				s_rate = self.args.t_save_rate
				if (self.e_step + 1) % (1000 * s_rate) == 0:
					print(self.e_step + 1, float(sum(self.ext_rewards[-1000 * s_rate:])) / (1000.0 * s_rate))
					self.dec.show(self.hot_map_save_path, self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.key_points.show(self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.save_results(self.ext_rewards, '%d' % (self.e_step + 1), self.args)

		return obs_n, (ext_rewards, dec_rewards, coor_rewards, penalty_rewards, cen_rewards), dones, infos


class SubprocVecEnv_Island(SubprocVecEnv_Pass):
	"""
	VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
	Recommended to use when num_envs > 1 and step() can be a bottleneck.
	"""

	def __init__(self, env_fns, args):
		SubprocVecEnv_Pass.__init__(self, env_fns, args)
		# For debugging
		self.dec = Island_Dec(self.args.size, self.args.island_agent_max_power,
		                      self.args.n_agent, self.args)
		self.cen = Island_Cen(self.args.size, self.args.island_agent_max_power, self.args.n_agent, self.args)
		self.key_points = Island_Appro_C_points(self.args.size, self.args.n_action,
		                                        self.args.island_agent_max_power,
		                                        self.args.island_wolf_max_power,
		                                        self.args, is_print=True)
		self.ext_rewards_list = [[] for _ in range(self.args.n_agent)]
		self.num_kill = []
		self.time_length = [[] for _ in range(self.args.n_agent)]
		self.death = [[] for _ in range(self.args.n_agent)]
		self.landmark = []

	def step(self, actions):
		self.step_async(actions)
		obs_n, ext_rewards, dones, infos = self.step_wait()
		obs_n = self.ma_reshape(obs_n, 1)
		ext_rewards = self.ma_reshape(ext_rewards, 1).astype('float32')
		state_n = []
		for i in range(self.num_envs):
			state_n.append(copy.deepcopy(infos[i]['state']))
			infos[i]['pre_state'] = self.pre_state_n[i]

		# estimate coor
		for j in range(self.num_envs):
			if self.pre_state_n[j] != None:
				self.key_points.update([self.pre_state_n[j], actions[j]], [state_n[j], None])

		# add intrinsic rew
		dec_rewards = ext_rewards.copy()
		coor_rewards = ext_rewards.copy()
		penalty_rewards = ext_rewards.copy()
		cen_rewards = ext_rewards.copy()
		for i in range(self.args.n_agent):
			for j in range(self.num_envs):
				dec_rewards[i, j] = self.args.gamma_dec * self.dec_curiosity(state_n[j], i)
				if self.pre_state_n[j] != None:
					coor_rewards[i, j] = self.coor_curiosity([self.pre_state_n[j], actions[j]], [state_n[j], None], i)
				else:
					coor_rewards[i, j] = 0.
				cen_rewards[i, j] = self.args.gamma_cen * self.cen.output(state_n[j])
				penalty_rewards[i, j] = self.args.penalty
		# update intrinsic rew
		for j in range(self.num_envs):
			self.dec.update(state_n[j])
			self.cen.update(state_n[j])

		self.pre_state_n = state_n
		for i in range(self.num_envs):
			if dones[i]:
				self.pre_state_n[i] = None

		# debug
		for i in range(self.num_envs):

			info_r = infos[i]['rew']
			self.num_kill.append(info_r['kill'])
			self.landmark.append(info_r['landmark'])
			for j, death in enumerate(info_r['death']):
				self.ext_rewards_list[j].append(ext_rewards[j, i])
				self.death[j].append(int(death))
				self.time_length[j].append(info_r['time_length'][j])

			if dones[i]:
				self.e_step += 1

				s_rate = self.args.t_save_rate
				if (self.e_step + 1) % (1000 * s_rate) == 0:

					print(self.e_step + 1)
					for i in range(self.args.n_agent):
						print('agent_%d : ' % i,
						      'ext', round(float(sum(self.ext_rewards_list[i])) / (1000.0 * s_rate), 2),
						      'death', round(float(sum(self.death[i])) / (1000.0 * s_rate), 2),
						      'time_length', round(sum(self.time_length[i]) / (1000.0 * s_rate), 2))
					print('kill', float(sum(self.num_kill)) / (1000.0 * s_rate),
					      'landmark', float(sum(self.landmark)) / (1000.0 * s_rate))

					self.ext_rewards_list = [[] for _ in range(self.args.n_agent)]
					self.num_kill = []
					self.time_length = [[] for _ in range(self.args.n_agent)]
					self.death = [[] for _ in range(self.args.n_agent)]
					self.landmark = []
					self.dec.show(self.hot_map_save_path, self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.key_points.show(self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.save_results(self.ext_rewards, '%d' % (self.e_step + 1), self.args)

		return obs_n, (ext_rewards, dec_rewards, coor_rewards, penalty_rewards, cen_rewards), dones, infos


class x_island_debug:

	def __init__(self, n_agent, name):
		self.n_agent = n_agent
		self.name = name
		self.p = []
		self.t = []
		self.err_p = []
		self.err_t = []
		self.rew = []
		self.flag = True

	def reset(self):
		self.print_x()
		self.p = []
		self.t = []
		self.err_p = []
		self.err_t = []
		self.rew = []
		self.flag = True

	def print_state(self, state):
		res = []
		for item in state:
			res.append(item[:3])
		res.append(state[0][-2:])
		print(res)

	def check_not_injured(self, pre_state, state, i):
		if not (state[i][2] < pre_state[i][2]):
			return True
		return False

	def check_single_injured(self, pre_state, state, i):
		if not (state[i][2] < pre_state[i][2]):
			return False
		for j in range(self.n_agent):
			if i != j and state[j][2] < pre_state[j][2]:
				return False
		return True

	def check_both_injured(self, pre_state, state, i):
		if not (state[i][2] < pre_state[i][2]):
			return False
		for j in range(self.n_agent):
			if i != j and state[j][2] < pre_state[j][2]:
				return True
		return False

	def update(self, data):
		pre_state_n, state_n, actions, labels, k, tt_stamp, j, \
		coor_p, coor_t, err_p, err_t, coor_rew_show = data
		if self.name == 'not injured' and not self.check_not_injured(pre_state_n, state_n, j):
			return
		if self.name == 'single injured' and not self.check_single_injured(pre_state_n, state_n, j):
			return
		for i in range(self.n_agent):
			if i != j and (self.name != 'both injured' or
			               (state_n[i][2] < pre_state_n[i][2] and state_n[j][2] < pre_state_n[j][2])):
				self.p.append(1. - coor_p[i])
				self.t.append(1. - coor_t[i])
				self.err_p.append(err_p[i])
				self.err_t.append(err_t[i])
				self.rew.append(coor_rew_show[i])
		if self.name == 'both injured' and not self.check_both_injured(pre_state_n, state_n, j):
			return
		if self.flag:
			if (self.name == 'any one' or self.name == 'not injured') and \
					not abs(np.sum(coor_rew_show)) > 0.5 * (self.n_agent - 1):
				pass
			else:
				self.flag = False
				self.print(data)

	def print(self, data):
		pre_state_n, state_n, actions, labels, k, tt_stamp, i, \
		coor_p, coor_t, err_p, err_t, t_coor_rew_show = data
		print('-------------')
		print('%s:' % self.name)
		self.flag_1 = False
		self.print_state(pre_state_n)
		self.print_state(state_n)
		print('actions:\t', actions, labels, k, tt_stamp, i)
		print('prob:\t\t', np.round(coor_p, decimals=5), np.round(coor_t, decimals=5))
		print('err and T:\t', np.round(err_p, decimals=3), np.round(err_t, decimals=3),
		      np.round(1 - t_coor_rew_show, decimals=3))

	def print_x(self):
		print('-------------')
		print('%s:' % self.name)
		print('coor_p', round(float(np.mean(self.p)), 3),
		      'coor_t', round(float(np.mean(self.t)), 3),
		      'err_p', round(float(np.mean(self.err_p)), 3),
		      'err_t', round(float(np.mean(self.err_t)), 3),
		      'coor_rew', round(float(np.mean(self.rew)), 3))


class SubprocVecEnv_x_Island_old(SubprocVecEnv_Pass):
	"""
	VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
	Recommended to use when num_envs > 1 and step() can be a bottleneck.
	"""

	def __init__(self, env_fns, args):
		SubprocVecEnv_Pass.__init__(self, env_fns, args)
		# For debugging
		self.dec = Island_Dec(self.args.size, self.args.x_island_agent_max_power,
		                      self.args.n_agent, self.args)
		self.cen = x_Island_Cen(self.args.size, self.args.x_island_agent_max_power, self.args.n_agent, self.args)
		self.key_points = Fast_Island_VI_Appro_C_points(self.args.size, self.args.n_action,
		                                                self.args.x_island_agent_max_power,
		                                                self.args.x_island_wolf_max_power,
		                                                self.args, is_print=True)
		self.n_agent = self.args.n_agent
		self.ext_rewards_list = [[] for _ in range(self.args.n_agent)]
		self.num_kill = []
		self.time_length = [[] for _ in range(self.args.n_agent)]
		self.death = [[] for _ in range(self.args.n_agent)]
		self.landmark = []

		self.debug_1 = x_island_debug(self.n_agent, 'any one')
		self.debug_2 = x_island_debug(self.n_agent, 'single injured')
		self.debug_3 = x_island_debug(self.n_agent, 'not injured')
		self.debug_4 = x_island_debug(self.n_agent, 'both injured')

	def step(self, actions):
		self.step_async(actions)
		obs_n, ext_rewards, dones, infos = self.step_wait()
		obs_n = self.ma_reshape(obs_n, 1)
		ext_rewards = self.ma_reshape(ext_rewards, 1).astype('float32')
		state_n = []
		for i in range(self.num_envs):
			state_n.append(copy.deepcopy(infos[i]['state']))
			infos[i]['pre_state'] = self.pre_state_n[i]

		# add intrinsic rew
		dec_rewards = ext_rewards.copy()
		penalty_rewards = ext_rewards.copy()
		cen_rewards = ext_rewards.copy()

		for i in range(self.args.n_agent):
			for j in range(self.num_envs):
				dec_rewards[i, j] = self.args.gamma_dec * self.dec_curiosity(state_n[j], i)
				cen_rewards[i, j] = self.args.gamma_cen * self.cen.output(state_n[j], i)
				penalty_rewards[i, j] = self.args.penalty

		coor_rewards = np.zeros((self.n_agent, self.n_agent, self.num_envs), dtype='float32')
		coor_p = np.zeros((self.n_agent, self.n_agent, self.num_envs), dtype='float32')
		coor_t = np.zeros((self.n_agent, self.n_agent, self.num_envs), dtype='float32')
		coor_p_show = np.zeros((self.n_agent, self.num_envs), dtype='float32')
		coor_t_show = np.zeros((self.n_agent, self.num_envs), dtype='float32')
		err_p = np.zeros((self.n_agent, self.n_agent, self.num_envs), dtype='float32')
		err_t = np.zeros((self.n_agent, self.n_agent, self.num_envs), dtype='float32')

		if not self.key_points.not_run:

			C_label = []
			C_x_p = []
			C_x_t = []
			for j in range(self.num_envs):
				if self.pre_state_n[j] != None:
					label, x_p, x_t = self.key_points.make([self.pre_state_n[j], actions[j]], [state_n[j], None])
					C_label.append(label)
					C_x_p.append(x_p)
					C_x_t.append(x_t)

			num_data = len(C_label)

			if num_data > 0:

				C_label = np.transpose(np.array(C_label), (1, 0))
				C_x_p = np.transpose(np.array(C_x_p), (1, 0, 2))
				C_x_t = np.transpose(np.array(C_x_t), (1, 0, 2))
				coor_output, t_coor_p, t_coor_t, t_coor_rew_show, err_p_show, err_t_show = \
					self.key_points.output(C_label, C_x_p, C_x_t)

				C_label_show = np.zeros((self.n_agent, self.n_agent, C_label.shape[1]), dtype='float32')
				t_stamp = 0
				for i in range(self.n_agent):
					for j in range(self.n_agent):
						if i != j:
							C_label_show[i, j] = C_label[t_stamp]
							t_stamp += 1

				tt_stamp = 0
				for k in range(self.num_envs):
					if self.pre_state_n[k] != None:
						coor_rewards[:, :, k] = coor_output[:, :, tt_stamp]
						coor_p[:, :, k] = t_coor_p[:, :, tt_stamp]
						coor_t[:, :, k] = t_coor_t[:, :, tt_stamp]
						err_p[:, :, k] = err_p_show[:, :, tt_stamp]
						err_t[:, :, k] = err_t_show[:, :, tt_stamp]
						coor_p_show[:, k] = self.n_agent - 1 - np.sum(t_coor_p[:, :, tt_stamp], axis=1)
						coor_t_show[:, k] = self.n_agent - 1 - np.sum(t_coor_t[:, :, tt_stamp], axis=1)
						for i in range(self.n_agent):
							# check i
							data = (self.pre_state_n[k], state_n[k], actions[k],
							        C_label_show[:, i, tt_stamp].astype(np.int8), k, tt_stamp, i,
							        coor_p[:, i, k], coor_t[:, i, k], err_p[:, i, k], err_t[:, i, k],
							        t_coor_rew_show[:, i, tt_stamp])
							self.debug_1.update(data)
							self.debug_2.update(data)
							self.debug_3.update(data)
							self.debug_4.update(data)
						tt_stamp += 1

		# update intrinsic rew
		for j in range(self.num_envs):
			self.dec.update(state_n[j])
			self.cen.update(state_n[j])

		self.pre_state_n = state_n
		for i in range(self.num_envs):
			if dones[i]:
				self.pre_state_n[i] = None

		# debug
		for i in range(self.num_envs):

			info_r = infos[i]['rew']
			self.num_kill.append(info_r['kill'])
			self.landmark.append(info_r['landmark'])
			for j, death in enumerate(info_r['death']):
				self.ext_rewards_list[j].append(ext_rewards[j, i])
				self.death[j].append(int(death))
				self.time_length[j].append(info_r['time_length'][j])

			if dones[i]:
				self.e_step += 1

				s_rate = self.args.t_save_rate
				t_time = 3000
				if (self.e_step + 1) % (t_time * s_rate) == 0:
					print(self.e_step + 1)
					for i in range(self.args.n_agent):
						print('agent_%d : ' % i,
						      'ext', round(float(sum(self.ext_rewards_list[i])) / (t_time * s_rate), 2),
						      'death', round(float(sum(self.death[i])) / (t_time * s_rate), 2),
						      'time_length', round(sum(self.time_length[i]) / (t_time * s_rate), 2))
					print('kill', round(float(sum(self.num_kill)) / (t_time * s_rate), 3),
					      'landmark', round(float(sum(self.landmark)) / (t_time * s_rate), 3))

					self.ext_rewards_list = [[] for _ in range(self.args.n_agent)]
					self.num_kill = []
					self.time_length = [[] for _ in range(self.args.n_agent)]
					self.death = [[] for _ in range(self.args.n_agent)]
					self.landmark = []
					self.dec.show(self.hot_map_save_path, self.e_step + 1)

				if (self.e_step + 1) % (5000 * s_rate) == 0:
					self.debug_1.reset()
					self.debug_2.reset()
					self.debug_3.reset()
					self.debug_4.reset()

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.key_points.show(self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.save_results(self.ext_rewards, '%d' % (self.e_step + 1), self.args)

		return obs_n, (ext_rewards, dec_rewards, coor_rewards, penalty_rewards, cen_rewards,
		               coor_p_show, coor_t_show), dones, infos


class x_island_debug_2:

	def __init__(self, n_agent, name):
		self.n_agent = n_agent
		self.name = name
		self.p = []
		self.t = []
		self.rew = []
		self.flag = True

	def reset(self):
		self.print_x()
		self.p = []
		self.t = []
		self.err_p = []
		self.err_t = []
		self.rew = []
		self.flag = True

	def print_state(self, state):
		res = []
		for item in state:
			res.append(item[:3])
		res.append(state[0][-2:])
		print(res)

	def check_not_injured(self, pre_state, state, i):
		if not (state[i][2] < pre_state[i][2]):
			return True
		return False

	def check_single_injured(self, pre_state, state, i):
		if not (state[i][2] < pre_state[i][2]):
			return False
		for j in range(self.n_agent):
			if i != j and state[j][2] < pre_state[j][2]:
				return False
		return True

	def check_both_injured(self, pre_state, state, i):
		if not (state[i][2] < pre_state[i][2]):
			return False
		for j in range(self.n_agent):
			if i != j and state[j][2] < pre_state[j][2]:
				return True
		return False

	def update(self, data):
		pre_state_n, state_n, actions, j, coor_p, coor_t, rew = data
		if self.name == 'not injured' and not self.check_not_injured(pre_state_n, state_n, j):
			return
		if self.name == 'single injured' and not self.check_single_injured(pre_state_n, state_n, j):
			return
		for i in range(self.n_agent):
			if i != j and (self.name != 'both injured' or
			               (state_n[i][2] < pre_state_n[i][2] and state_n[j][2] < pre_state_n[j][2])):
				self.p.append(1. - coor_p[i])
				self.t.append(1. - coor_t)
				self.rew.append(rew[i])
		if self.name == 'both injured' and not self.check_both_injured(pre_state_n, state_n, j):
			return
		if self.flag:
			if (self.name == 'any one' or self.name == 'not injured') and \
					not abs(np.sum(rew)) > 0.5 * (self.n_agent - 1):
				pass
			else:
				self.flag = False
				self.print(data)

	def print(self, data):
		pre_state_n, state_n, actions, i, coor_p, coor_t, rew = data
		print('-------------')
		print('%s:' % self.name)
		self.flag_1 = False
		self.print_state(pre_state_n)
		self.print_state(state_n)
		print('actions:\t', actions, i)
		print('prob:\t\t', np.round(coor_p, decimals=5), np.round(coor_t, decimals=5))
		print('reward:\t', np.round(rew, decimals=3))

	def print_x(self):
		print('-------------')
		print('%s:' % self.name)
		print('coor_p', round(float(np.mean(self.p)), 3),
		      'coor_t', round(float(np.mean(self.t)), 3),
		      'coor_rew', round(float(np.mean(self.rew)), 3))


class SubprocVecEnv_x_Island(SubprocVecEnv_Pass):
	"""
	VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
	Recommended to use when num_envs > 1 and step() can be a bottleneck.
	"""

	def __init__(self, env_fns, args):
		SubprocVecEnv_Pass.__init__(self, env_fns, args)
		# For debugging
		self.dec = Island_Dec(self.args.size, self.args.x_island_agent_max_power,
		                      self.args.n_agent, self.args)
		self.cen = x_Island_Cen(self.args.size, self.args.island_agent_max_power, self.args.n_agent, self.args)
		self.key_points = x_Island_Appro_C_points(self.args.size, self.args.n_action,
		                                          self.args.x_island_agent_max_power,
		                                          self.args.x_island_wolf_max_power,
		                                          self.args, is_print=True)
		self.n_agent = self.args.n_agent
		self.ext_rewards_list = [[] for _ in range(self.args.n_agent)]
		self.num_kill = []
		self.time_length = [[] for _ in range(self.args.n_agent)]
		self.death = [[] for _ in range(self.args.n_agent)]
		self.landmark = []

		self.debug_1 = x_island_debug_2(self.n_agent, 'any one')
		self.debug_2 = x_island_debug_2(self.n_agent, 'single injured')
		self.debug_3 = x_island_debug_2(self.n_agent, 'not injured')
		self.debug_4 = x_island_debug_2(self.n_agent, 'both injured')

	def step(self, actions):
		self.step_async(actions)
		obs_n, ext_rewards, dones, infos = self.step_wait()
		obs_n = self.ma_reshape(obs_n, 1)
		ext_rewards = self.ma_reshape(ext_rewards, 1).astype('float32')
		state_n = []
		for i in range(self.num_envs):
			state_n.append(copy.deepcopy(infos[i]['state']))
			infos[i]['pre_state'] = self.pre_state_n[i]

		# estimate coor
		for j in range(self.num_envs):
			if self.pre_state_n[j] != None:
				self.key_points.update([self.pre_state_n[j], actions[j]], [state_n[j], None])

		# add intrinsic rew
		dec_rewards = ext_rewards.copy()
		penalty_rewards = ext_rewards.copy()
		cen_rewards = ext_rewards.copy()
		for i in range(self.args.n_agent):
			for j in range(self.num_envs):
				dec_rewards[i, j] = self.args.gamma_dec * self.dec_curiosity(state_n[j], i)
				cen_rewards[i, j] = self.args.gamma_cen * self.cen.output(state_n[j], i)
				penalty_rewards[i, j] = self.args.penalty

		coor_rewards = np.zeros_like(ext_rewards)
		for j in range(self.num_envs):
			if not self.key_points.not_run and self.pre_state_n[j] != None:
				rew, c_t, c_p = \
					self.key_points.output([self.pre_state_n[j], actions[j]], [state_n[j], None])
				coor_rewards[:, j] = np.sum(rew, axis=1)
				for i in range(self.n_agent):
					data = (self.pre_state_n[j], state_n[j], actions[j], i, c_p[i], c_t, rew[i])
					self.debug_1.update(data)
					self.debug_2.update(data)
					self.debug_3.update(data)
					self.debug_4.update(data)

		# update intrinsic rew
		for j in range(self.num_envs):
			self.dec.update(state_n[j])
			self.cen.update(state_n[j])

		self.pre_state_n = state_n
		for i in range(self.num_envs):
			if dones[i]:
				self.pre_state_n[i] = None

		# debug
		for i in range(self.num_envs):

			info_r = infos[i]['rew']
			self.num_kill.append(info_r['kill'])
			self.landmark.append(info_r['landmark'])
			for j, death in enumerate(info_r['death']):
				self.ext_rewards_list[j].append(ext_rewards[j, i])
				self.death[j].append(int(death))
				self.time_length[j].append(info_r['time_length'][j])

			if dones[i]:
				self.e_step += 1

				s_rate = self.args.t_save_rate
				t_time = 3000 if self.args.nsteps == 2048 else 1000
				if (self.e_step + 1) % (t_time * s_rate) == 0:
					print(self.e_step + 1)
					for i in range(self.args.n_agent):
						print('agent_%d : ' % i,
						      'ext', round(float(sum(self.ext_rewards_list[i])) / (t_time * s_rate), 2),
						      'death', round(float(sum(self.death[i])) / (t_time * s_rate), 2),
						      'time_length', round(sum(self.time_length[i]) / (t_time * s_rate), 2))
					print('kill', round(float(sum(self.num_kill)) / (t_time * s_rate), 3),
					      'landmark', round(float(sum(self.landmark)) / (t_time * s_rate), 3))

					self.ext_rewards_list = [[] for _ in range(self.args.n_agent)]
					self.num_kill = []
					self.time_length = [[] for _ in range(self.args.n_agent)]
					self.death = [[] for _ in range(self.args.n_agent)]
					self.landmark = []
					self.dec.show(self.hot_map_save_path, self.e_step + 1)

				t_time = 5000 if self.args.nsteps == 2048 else 2000
				if (self.e_step + 1) % (t_time * s_rate) == 0 and not self.key_points.not_run:
					self.debug_1.reset()
					self.debug_2.reset()
					self.debug_3.reset()
					self.debug_4.reset()

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.key_points.show(self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.save_results(self.ext_rewards, '%d' % (self.e_step + 1), self.args)

		return obs_n, (ext_rewards, dec_rewards, coor_rewards, penalty_rewards, cen_rewards), dones, infos


class SubprocVecEnv_PushBall(SubprocVecEnv_Pass):
	"""
	VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
	Recommended to use when num_envs > 1 and step() can be a bottleneck.
	"""

	def __init__(self, env_fns, args):
		SubprocVecEnv_Pass.__init__(self, env_fns, args)
		# For debugging
		self.dec = Pushball_Dec(self.args.size, self.args.n_agent, self.args)
		self.cen = Pushball_Cen(self.args.size, self.args.n_agent, self.args)
		self.key_points = Pushball_Appro_C_points(self.args.size, self.args.n_action, self.args, is_print=True)
		self.win_rewards = [[] for _ in range(4)]

	def step(self, actions):
		self.step_async(actions)
		obs_n, ext_rewards, dones, infos = self.step_wait()
		obs_n = self.ma_reshape(obs_n, 1)
		ext_rewards = self.ma_reshape(ext_rewards, 1).astype('float32')
		state_n = []
		for i in range(self.num_envs):
			state_n.append(copy.deepcopy(infos[i]['state']))
			infos[i]['pre_state'] = self.pre_state_n[i]

		# estimate coor
		for j in range(self.num_envs):
			if self.pre_state_n[j] != None:
				self.key_points.update([self.pre_state_n[j], actions[j]], [state_n[j], None])

		# add intrinsic rew
		dec_rewards = ext_rewards.copy()
		coor_rewards = ext_rewards.copy()
		penalty_rewards = ext_rewards.copy()
		cen_rewards = ext_rewards.copy()
		for i in range(self.args.n_agent):
			for j in range(self.num_envs):
				dec_rewards[i, j] = self.args.gamma_dec * self.dec_curiosity(state_n[j], i)
				if self.pre_state_n[j] != None:
					coor_rewards[i, j] = self.coor_curiosity([self.pre_state_n[j], actions[j]], [state_n[j], None], i)
				else:
					coor_rewards[i, j] = 0.
				cen_rewards[i, j] = self.args.gamma_cen * self.cen.output(state_n[j])
				penalty_rewards[i, j] = self.args.penalty
		# update intrinsic rew
		for j in range(self.num_envs):
			self.dec.update(state_n[j])
			self.cen.update(state_n[j])

		self.pre_state_n = state_n
		for i in range(self.num_envs):
			if dones[i]:
				self.pre_state_n[i] = None

		# debug
		for i in range(self.num_envs):

			info_r = infos[i]['rew']
			for i in range(4):
				self.win_rewards[i].append(info_r[i])

			if dones[i]:
				self.e_step += 1

				s_rate = self.args.t_save_rate
				if (self.e_step + 1) % (1000 * s_rate) == 0:
					print(self.e_step + 1,
					      'up', float(sum(self.win_rewards[0])) / (1000.0 * s_rate),
					      'left', float(sum(self.win_rewards[1])) / (1000.0 * s_rate),
					      'down', float(sum(self.win_rewards[2])) / (1000.0 * s_rate),
					      'right', float(sum(self.win_rewards[3])) / (1000.0 * s_rate))
					self.win_rewards = [[] for _ in range(4)]
					self.dec.show(self.hot_map_save_path, self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.key_points.show(self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.save_results(self.ext_rewards, '%d' % (self.e_step + 1), self.args)

		return obs_n, (ext_rewards, dec_rewards, coor_rewards, penalty_rewards, cen_rewards), dones, infos


class SubprocVecEnv_Leftward(SubprocVecEnv):
	"""
	VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
	Recommended to use when num_envs > 1 and step() can be a bottleneck.
	"""

	def __init__(self, env_fns, args):
		SubprocVecEnv.__init__(self, env_fns)
		# For debugging
		self.hot_map_save_path = args.save_path
		self.ext_rewards = []
		self.args = args
		self.dec = Dec(100, self.args.n_agent, self.args)
		self.key_points = C_points(100, self.args.n_action, self.args, is_print=True)
		self.pre_state_n = [None for i in range(self.num_envs)]
		self.e_step = 0

	def dec_curiosity(self, state, i):
		return self.dec.output(state, i)

	def coor_curiosity(self, data_1, data_2, i):
		return self.key_points.output(data_1, data_2, i)

	def ma_reshape(self, obs, i):
		obs = np.array(obs)
		index = [i for i in range(len(obs.shape))]
		index[0] = i
		index[i] = 0
		return np.transpose(obs, index).copy()

	def smooth(self, data):

		smoothed = []

		for i in range(len(data)):
			smoothed.append(np.mean(data[max(0, i + 1 - 100): i + 1]))

		return smoothed

	def save_results(self, er2, name, arg):
		filename = '%s/%s_data.pkl' % (arg.save_path, name)
		with open(filename, 'wb') as file:
			pickle.dump(er2, file)

		# er3 = q_cen_for_pass()

		# er1_s = smooth(er1)
		er2_s = self.smooth(er2)
		# er3_s = smooth(er3)

		m_figure = plt.figure()
		m_ax1 = m_figure.add_subplot(3, 2, 1)
		m_ax2 = m_figure.add_subplot(3, 2, 2)
		m_ax3 = m_figure.add_subplot(3, 2, 3)
		m_ax4 = m_figure.add_subplot(3, 2, 4)
		m_ax5 = m_figure.add_subplot(3, 2, 5)
		m_ax6 = m_figure.add_subplot(3, 2, 6)

		# m_ax1.plot(er1_s)
		# m_ax2.plot(er1)
		m_ax3.plot(er2_s)
		m_ax4.plot(er2)
		# m_ax5.plot(er3_s)
		# m_ax6.plot(er3)

		m_ax1.legend(['epsilon-greedy'])
		m_ax2.legend(['epsilon-greedy (unsmoothed)'])
		m_ax3.legend(['dec-curiosity'])
		m_ax4.legend(['dec-curiosity (unsmoothed)'])
		m_ax5.legend(['cen-curiosity'])
		m_ax6.legend(['cen-curiosity (unsmoothed)'])

		m_figure.savefig('%s/%s_i Boltzmann.png' % (arg.save_path, name))
		plt.close(m_figure)

	def reset(self):
		self._assert_not_closed()
		for remote in self.remotes:
			remote.send(('reset', None))
		obs_n = _flatten_obs([remote.recv() for remote in self.remotes])
		obs_n = self.ma_reshape(obs_n, 1)
		return obs_n

	def step(self, actions):
		self.step_async(actions)
		obs_n, ext_rewards, dones, infos = self.step_wait()
		obs_n = self.ma_reshape(obs_n, 1)
		ext_rewards = self.ma_reshape(ext_rewards, 1).astype('float32')
		state_n = []
		for i in range(self.num_envs):
			state_n.append(copy.deepcopy(infos[i]['state']))
			infos[i]['pre_state'] = self.pre_state_n[i]

		# estimate coor
		for j in range(self.num_envs):
			if (self.pre_state_n[j] != None):
				self.key_points.update([self.pre_state_n[j], actions[j]], [state_n[j], None])

		# add intrinsic rew
		dec_rewards = ext_rewards.copy()
		coor_rewards = ext_rewards.copy()
		coor_rewards_show = ext_rewards.copy()
		penalty_rewards = ext_rewards.copy()
		for i in range(self.args.n_agent):
			for j in range(self.num_envs):
				dec_rewards[i, j] = self.args.gamma_dec * self.dec_curiosity(state_n[j], i)
				if (self.pre_state_n[j] != None):
					coor_rewards[i, j] = self.coor_curiosity([self.pre_state_n[j], actions[j]], [state_n[j], None], i)
					coor_rewards_show[i, j] = 1 - coor_rewards[i, j]
				else:
					coor_rewards[i, j] = 0.
					coor_rewards_show[i, j] = 0.
				penalty_rewards[i, j] = self.args.penalty
		# update intrinsic rew
		for j in range(self.num_envs):
			self.dec.update(state_n[j], infos[j]['door'])

		self.pre_state_n = state_n
		for i in range(self.num_envs):
			if dones[i]:
				self.pre_state_n[i] = None

		# debug
		for i in range(self.num_envs):
			if dones[i]:
				self.e_step += 1
				self.ext_rewards.append(ext_rewards[0, i])

				if (self.e_step + 1) % 1000 == 0:
					print(self.e_step + 1, float(sum(self.ext_rewards[-1000:])) / 1000.0)
					self.dec.show(self.hot_map_save_path, self.e_step + 1)

				if (self.e_step + 1) % 10000 == 0:
					self.key_points.show(self.e_step + 1)

				if (self.e_step + 1) % 100000 == 0:
					self.save_results(self.ext_rewards, '%d' % (self.e_step + 1), self.args)

		return obs_n, (ext_rewards, dec_rewards, coor_rewards, coor_rewards_show, penalty_rewards), dones, infos


def _flatten_obs(obs):
	assert isinstance(obs, (list, tuple))
	assert len(obs) > 0

	if isinstance(obs[0], dict):
		keys = obs[0].keys()
		return {k: np.stack([o[k] for o in obs]) for k in keys}
	else:
		return np.stack(obs)
