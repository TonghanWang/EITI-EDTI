import os
import math
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
import copy


def mkdir(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path)


class Gather_lc_3:

	def __init__(self, save_path, args, name):
		self.args = args
		self.name = name
		self.save_path = save_path + name + '/'
		self.p_step = 0
		self.buffer = []

		mkdir(self.save_path + '/figure/')
		mkdir(self.save_path + '/data/')

	def smooth(self, data):
		smoothed = []
		for i in range(len(data)):
			smoothed.append(np.mean(data[max(0, i + 1 - 100): i + 1]))
		return smoothed

	def save_img(self):
		data_s = self.smooth(self.buffer)
		m_figure = plt.figure(figsize=(16, 10))
		m_ax1 = m_figure.add_subplot(1, 2, 1)
		m_ax2 = m_figure.add_subplot(1, 2, 2)
		m_ax1.plot(self.buffer)
		m_ax2.plot(data_s)
		m_figure.savefig('%s/figure/%d.png' % (self.save_path, self.p_step))
		plt.close(m_figure)

	def save_data(self):
		filename = '%s/data/%d.pkl' % (self.save_path, self.p_step)
		with open(filename, 'wb') as file:
			pickle.dump(self.buffer, file)

	def update(self, data):
		self.buffer.append(data)
		self.p_step += 1
		if self.p_step % 100 == 0:
			self.save_data()
			self.save_img()


class Gather_hm_2:

	def __init__(self, save_path, args, name):
		self.args = args
		self.name = name
		self.save_path = save_path + name + '/'
		self.p_step = 0

		mkdir(self.save_path + '/figure/')
		mkdir(self.save_path + '/data/')

		self.visited = [np.zeros([2, args.size, args.size]) for _ in range(self.args.n_agent)]
		self.visited_old = [np.zeros([2, args.size, args.size]) for _ in range(self.args.n_agent)]
		self.value = [np.zeros([2, args.size, args.size]) for _ in range(self.args.n_agent)]
		self.value_old = [np.zeros([2, args.size, args.size]) for _ in range(self.args.n_agent)]

	def normal(self, data, total):
		t_data = data.copy()
		t_total = total.copy()
		zero = total == 0
		t_total[zero] = 1
		t_data = t_data / t_total
		t_min = np.min(t_data)
		t_max = np.max(t_data)
		t_data[zero] = t_min
		t_data = (t_data - t_min) / (t_max - t_min + 1)
		return np.log(t_data + 1)

	def save_img(self):

		figure = plt.figure(figsize=(16, 10))

		ax1 = figure.add_subplot(2, 4, 1)
		ax2 = figure.add_subplot(2, 4, 2)
		ax3 = figure.add_subplot(2, 4, 3)
		ax4 = figure.add_subplot(2, 4, 4)
		ax5 = figure.add_subplot(2, 4, 5)
		ax6 = figure.add_subplot(2, 4, 6)
		ax7 = figure.add_subplot(2, 4, 7)
		ax8 = figure.add_subplot(2, 4, 8)

		ax1.imshow(self.normal(self.value[0][0], self.visited[0][0]))
		ax2.imshow(self.normal(self.value[0][0] - self.value_old[0][0],
		                       self.visited[0][0] - self.visited_old[0][0]))
		ax3.imshow(self.normal(self.value[0][1], self.visited[0][1]))
		ax4.imshow(self.normal(self.value[0][1] - self.value_old[0][1],
		                       self.visited[0][1] - self.visited_old[0][1]))

		ax5.imshow(self.normal(self.value[1][0], self.visited[1][0]))
		ax6.imshow(self.normal(self.value[1][0] - self.value_old[1][0],
		                       self.visited[1][0] - self.visited_old[1][0]))
		ax7.imshow(self.normal(self.value[1][1], self.visited[1][1]))
		ax8.imshow(self.normal(self.value[1][1] - self.value_old[1][1],
		                       self.visited[1][1] - self.visited_old[1][1]))

		figure.savefig('%s/figure/%d.png' % (self.save_path, self.p_step))
		plt.close(figure)

	def save_data(self):
		filename = '%s/data/visited_%d.pkl' % (self.save_path, self.p_step)
		with open(filename, 'wb') as file:
			pickle.dump(self.visited, file)
		filename = '%s/data/visited_old_%d.pkl' % (self.save_path, self.p_step)
		with open(filename, 'wb') as file:
			pickle.dump(self.visited_old, file)
		filename = '%s/data/value_%d.pkl' % (self.save_path, self.p_step)
		with open(filename, 'wb') as file:
			pickle.dump(self.value, file)
		filename = '%s/data/value_old_%d.pkl' % (self.save_path, self.p_step)
		with open(filename, 'wb') as file:
			pickle.dump(self.value_old, file)

	def save_trajectory(self):
		figure = plt.figure(figsize=(16, 10))

		ax1 = figure.add_subplot(2, 2, 1)
		ax2 = figure.add_subplot(2, 2, 2)
		ax3 = figure.add_subplot(2, 2, 3)
		ax4 = figure.add_subplot(2, 2, 4)

		ax1.imshow(np.log(self.visited[0][0] + 1))
		ax2.imshow(np.log(self.visited[0][0] - self.visited_old[0][0] + 1))
		ax3.imshow(np.log(self.visited[1][0] + 1))
		ax4.imshow(np.log(self.visited[1][0] - self.visited_old[1][0] + 1))

		figure.savefig('%s/figure/%d.png' % (self.save_path, self.p_step))
		plt.close(figure)

	def update_trajectory(self, infos):
		for item_id, item in enumerate(infos):
			for i in range(self.args.num_env):
				state = item[i]['state']
				for j in range(self.args.n_agent):
					self.visited[j][0][state[j][0]][state[j][1]] += 1
		self.p_step += 1
		if self.p_step % 100 == 0:
			self.save_data()
			self.save_trajectory()
			self.visited_old = [v.copy() for v in self.visited]

	def update_rew(self, infos, rew):
		for item_id, item in enumerate(infos):
			for i in range(self.args.num_env):
				pre_state = item[i]['pre_state']
				if pre_state != None:
					for j in range(self.args.n_agent):
						self.value[j][0][pre_state[j][0]][pre_state[j][1]] += rew[item_id][j][i]
						self.visited[j][0][pre_state[j][0]][pre_state[j][1]] += 1
						self.value[1 - j][1][pre_state[1 - j][0]][pre_state[1 - j][1]] += rew[item_id][j][i]
						self.visited[1 - j][1][pre_state[1 - j][0]][pre_state[1 - j][1]] += 1
		self.p_step += 1
		if self.p_step % 100 == 0:
			self.save_data()
			self.save_img()
			self.visited_old = [v.copy() for v in self.visited]
			self.value_old = [v.copy() for v in self.value]


class Gather_lc_2:

	def __init__(self, save_path, args, name):
		self.args = args
		self.name = name
		self.save_path = save_path + name + '/'
		self.gather = [Gather_lc_3(self.save_path, self.args, 'agent_%d' % i) for i in range(self.args.n_agent)]

	def update(self, data):
		for i in range(self.args.n_agent):
			self.gather[i].update(data[i])


class Gather_hm:

	def __init__(self, save_path, args, name):
		self.args = args
		self.name = name
		self.save_path = save_path + name + '/'
		self.trajectory = Gather_hm_2(self.save_path, self.args, 'trajectory')
		self.coor_t_show = Gather_hm_2(self.save_path, self.args, 'coor_t_show')
		self.coor_v_ext_show = Gather_hm_2(self.save_path, self.args, 'coor_v_ext_show')
		self.coor_v_int_show = Gather_hm_2(self.save_path, self.args, 'coor_v_int_show')
		self.coor_tv_show = Gather_hm_2(self.save_path, self.args, 'coor_tv_show')
		self.coor_tv_ext_show = Gather_hm_2(self.save_path, self.args, 'coor_tv_ext_show')
		self.coor_tv_int_show = Gather_hm_2(self.save_path, self.args, 'coor_tv_int_show')
		self.coor_r_tv_show = Gather_hm_2(self.save_path, self.args, 'coor_r_tv_show')
		self.prob_coor_p = Gather_hm_2(self.save_path, self.args, 'prob_coor_p')
		self.prob_coor_t = Gather_hm_2(self.save_path, self.args, 'prob_coor_t')

	def update(self, data):

		ext_rewards_v_n, int_rewards_v_n, ext_rewards_tv_n, int_rewards_tv_n, \
		c_rewards_show, c_rewards_tv, c_rewards_all, coor_p_list, coor_t_list, infos_list = data

		self.trajectory.update_trajectory(infos_list)

		if self.args.env == 'x_pass' or self.args.env == 'x_island_old':
			self.prob_coor_p.update_rew(infos_list, coor_p_list)
			self.prob_coor_t.update_rew(infos_list, coor_t_list)

		if self.args.s_alg_name == 'noisy':
			pass
		elif self.args.s_alg_name == 'dec':
			pass
		elif self.args.s_alg_name == 'cen':
			pass
		elif self.args.s_alg_name == 'coor_r':
			pass
		elif self.args.s_alg_name == 'coor_v':
			self.coor_v_ext_show.update_rew(infos_list, ext_rewards_v_n)
			self.coor_v_int_show.update_rew(infos_list, int_rewards_v_n)
		elif self.args.s_alg_name == 'coor_t':
			self.coor_t_show.update_rew(infos_list, c_rewards_show)
		elif self.args.s_alg_name == 'coor_tv':
			self.coor_v_ext_show.update_rew(infos_list, ext_rewards_v_n)
			self.coor_v_int_show.update_rew(infos_list, int_rewards_v_n)
			self.coor_t_show.update_rew(infos_list, c_rewards_show)
			self.coor_tv_show.update_rew(infos_list, c_rewards_tv)
			self.coor_tv_ext_show.update_rew(infos_list, ext_rewards_tv_n)
			self.coor_tv_int_show.update_rew(infos_list, int_rewards_tv_n)
		elif self.args.s_alg_name == 'coor_r_tv':
			self.coor_v_ext_show.update_rew(infos_list, ext_rewards_v_n)
			self.coor_v_int_show.update_rew(infos_list, int_rewards_v_n)
			self.coor_t_show.update_rew(infos_list, c_rewards_show)
			self.coor_tv_show.update_rew(infos_list, c_rewards_tv)
			self.coor_tv_ext_show.update_rew(infos_list, ext_rewards_tv_n)
			self.coor_tv_int_show.update_rew(infos_list, int_rewards_tv_n)
			self.coor_r_tv_show.update_rew(infos_list, c_rewards_all)
		elif self.args.s_alg_name == 'voi_int':
			self.coor_v_ext_show.update_rew(infos_list, ext_rewards_v_n)
			self.coor_v_int_show.update_rew(infos_list, int_rewards_v_n)
			self.coor_t_show.update_rew(infos_list, c_rewards_show)
			self.coor_tv_show.update_rew(infos_list, c_rewards_tv)
			self.coor_tv_ext_show.update_rew(infos_list, ext_rewards_tv_n)
			self.coor_tv_int_show.update_rew(infos_list, int_rewards_tv_n)
			self.coor_r_tv_show.update_rew(infos_list, c_rewards_all)
		else:
			assert True


class Gather_lc:

	def __init__(self, save_path, args, name):
		self.args = args
		self.name = name
		self.save_path = save_path + name + '/'
		self.all = Gather_lc_2(self.save_path, self.args, 'all')
		self.ext = Gather_lc_2(self.save_path, self.args, 'ext')
		self.dec = Gather_lc_2(self.save_path, self.args, 'dec')
		self.cen = Gather_lc_2(self.save_path, self.args, 'cen')
		self.coor_r = Gather_lc_2(self.save_path, self.args, 'coor_r')
		self.coor_v = Gather_lc_2(self.save_path, self.args, 'coor_v')
		self.coor_v_ext = Gather_lc_2(self.save_path, self.args, 'coor_v_ext')
		self.coor_v_int = Gather_lc_2(self.save_path, self.args, 'coor_v_int')
		self.coor_t = Gather_lc_2(self.save_path, self.args, 'coor_t')
		self.coor_tv = Gather_lc_2(self.save_path, self.args, 'coor_tv')
		self.coor_r_tv = Gather_lc_2(self.save_path, self.args, 'coor_r_tv')
		self.coor_all = Gather_lc_2(self.save_path, self.args, 'coor_all')
		self.coor_tv_ext = Gather_lc_2(self.save_path, self.args, 'coor_tv_ext')
		self.coor_tv_int = Gather_lc_2(self.save_path, self.args, 'coor_tv_int')

		self.time_length = Gather_lc_2(self.save_path, self.args, 'time_length')
		self.death = Gather_lc_2(self.save_path, self.args, 'death')

		self.kill = Gather_lc_3(self.save_path, self.args, 'kill')
		self.landmark = Gather_lc_3(self.save_path, self.args, 'landmark')
		self.up = Gather_lc_3(self.save_path, self.args, 'up')
		self.left = Gather_lc_3(self.save_path, self.args, 'left')
		self.down = Gather_lc_3(self.save_path, self.args, 'down')
		self.right = Gather_lc_3(self.save_path, self.args, 'right')

	def update(self, data, total_data):

		ext_rewards_v_n, int_rewards_v_n, ext_rewards_tv_n, int_rewards_tv_n, all_rewards, e_rewards, d_rewards, \
		c_rewards_t, c_rewards_r, c_rewards_v, c_rewards_tv, c_rewards_all, \
		ce_rewards, time_length_list, death_list, win_list = data

		kill_list, landmark_list = total_data

		if self.args.env == 'pushball':
			up, left, down, right = win_list
			self.up.update(up)
			self.left.update(left)
			self.down.update(down)
			self.right.update(right)
		elif self.args.env == 'x_island' or self.args.env == 'island':
			self.time_length.update(time_length_list)
			self.death.update(death_list)
			self.kill.update(kill_list)
			self.landmark.update(landmark_list)

		if self.args.s_alg_name == 'noisy':
			self.all.update(all_rewards)
			self.ext.update(e_rewards)
		elif self.args.s_alg_name == 'dec':
			self.all.update(all_rewards)
			self.ext.update(e_rewards)
			self.dec.update(d_rewards)
		elif self.args.s_alg_name == 'cen':
			self.all.update(all_rewards)
			self.ext.update(e_rewards)
			self.cen.update(ce_rewards)
		elif self.args.s_alg_name == 'coor_r':
			self.all.update(all_rewards)
			self.ext.update(e_rewards)
			self.dec.update(d_rewards)
			self.coor_r.update(c_rewards_r)
			self.coor_all.update(c_rewards_all)
		elif self.args.s_alg_name == 'coor_v':
			self.all.update(all_rewards)
			self.ext.update(e_rewards)
			self.dec.update(d_rewards)
			self.coor_v.update(c_rewards_v)
			self.coor_v_ext.update(ext_rewards_v_n)
			self.coor_v_int.update(int_rewards_v_n)
			self.coor_all.update(c_rewards_all)
		elif self.args.s_alg_name == 'coor_t':
			self.all.update(all_rewards)
			self.ext.update(e_rewards)
			self.dec.update(d_rewards)
			self.coor_t.update(c_rewards_t)
			self.coor_all.update(c_rewards_all)
		elif self.args.s_alg_name == 'coor_tv':
			self.all.update(all_rewards)
			self.ext.update(e_rewards)
			self.dec.update(d_rewards)
			self.coor_t.update(c_rewards_t)
			self.coor_tv.update(c_rewards_tv)
			self.coor_tv_ext.update(ext_rewards_tv_n)
			self.coor_tv_int.update(int_rewards_tv_n)
			self.coor_v_ext.update(ext_rewards_v_n)
			self.coor_v_int.update(int_rewards_v_n)
			self.coor_all.update(c_rewards_all)
		elif self.args.s_alg_name == 'coor_r_tv':
			self.all.update(all_rewards)
			self.ext.update(e_rewards)
			self.dec.update(d_rewards)
			self.coor_r.update(c_rewards_r)
			self.coor_t.update(c_rewards_t)
			self.coor_tv.update(c_rewards_tv)
			self.coor_tv_ext.update(ext_rewards_tv_n)
			self.coor_tv_int.update(int_rewards_tv_n)
			self.coor_v_ext.update(ext_rewards_v_n)
			self.coor_v_int.update(int_rewards_v_n)
			self.coor_r_tv.update(c_rewards_all)
			self.coor_all.update(c_rewards_all)
		elif self.args.s_alg_name == 'voi_int':
			self.all.update(all_rewards)
			self.ext.update(e_rewards)
			self.dec.update(d_rewards)
			self.coor_r.update(c_rewards_r)
			self.coor_t.update(c_rewards_t)
			self.coor_tv.update(c_rewards_tv)
			self.coor_tv_ext.update(ext_rewards_tv_n)
			self.coor_tv_int.update(int_rewards_tv_n)
			self.coor_v_ext.update(ext_rewards_v_n)
			self.coor_v_int.update(int_rewards_v_n)
			self.coor_r_tv.update(c_rewards_all)
			self.coor_all.update(c_rewards_all)
		else:
			assert True


'''
class Gather_epi:

	def __init__(self, save_path, args, name):
		self.args = args
		self.name = name
		self.save_path = save_path + name + '/'
		self.learning_curve_gather = Gather_lc(self.save_path, self.args, 'learning_curve')
		self.heatmap_gather = Gather_hm(self.save_path, self.args, 'heatmap')

	def merge_data(self, data):
		pass

	def update(self, data):
		data = self.merge_data(data)
		self.learning_curve_gather.update(data)
		self.heatmap_gather.update(data)
'''


class Gather_update:

	def __init__(self, save_path, args, name):
		self.args = args
		self.name = name
		self.save_path = save_path + name + '/'
		self.learning_curve_gather = Gather_lc(self.save_path, self.args, 'learning_curve')
		self.heatmap_gather = Gather_hm(self.save_path, self.args, 'heatmap')

	def merge_reward(self, rew):
		rew = np.sum(np.array(rew), axis=(0, 2))
		return rew

	def merge_data(self, data):

		ext_rewards_v_n, int_rewards_v_n, ext_rewards_tv_n, int_rewards_tv_n, all_rewards, e_rewards, d_rewards, \
		c_rewards_show, c_rewards_t, c_rewards_r, c_rewards_v, c_rewards_tv, \
		c_rewards_all, ce_rewards, mb_dones_n, infos_list, \
		kill_list, landmark_list, time_length_list, death_list, win_list, coor_p_list, coor_t_list = data

		hm_list = [copy.deepcopy(ext_rewards_v_n), copy.deepcopy(int_rewards_v_n),
		           copy.deepcopy(ext_rewards_tv_n), copy.deepcopy(int_rewards_tv_n), c_rewards_show,
		           copy.deepcopy(c_rewards_tv), copy.deepcopy(c_rewards_all), coor_p_list, coor_t_list, infos_list]

		num_episode = np.sum(mb_dones_n[0]) + self.args.num_env

		rew = (ext_rewards_v_n, int_rewards_v_n, ext_rewards_tv_n, int_rewards_tv_n, all_rewards, e_rewards, d_rewards,
		       c_rewards_t, c_rewards_r, c_rewards_v, c_rewards_tv, c_rewards_all, ce_rewards)

		rew_data = []
		for item in rew:
			rew_data.append(1. * self.merge_reward(item) / num_episode)

		single_rew = (time_length_list, death_list, win_list)
		for item in single_rew:
			rew_data.append(1. * np.sum(item, axis=1) / num_episode)

		total_rew = (kill_list, landmark_list)

		total_rew_data = []
		for item in total_rew:
			total_rew_data.append(1. * np.sum(item) / num_episode)

		return rew_data, total_rew_data, hm_list

	def update(self, data):
		t_rew, t_total_rew, t_hm = self.merge_data(data)
		self.learning_curve_gather.update(t_rew, t_total_rew)
		self.heatmap_gather.update(t_hm)


class Gather:

	def __init__(self, args):
		self.args = args
		self.save_path = self.args.s_data_path + self.args.env + '/' + 'size_%d' % self.args.size + '/' + \
		                 self.args.alg + '/' + self.args.s_alg_name + '/' + 'try_%d' % self.args.s_try_num + '/'
		#self.epi_gather = Gather_epi(self.save_path, self.args, 'episode')
		self.update_gather = Gather_update(self.save_path, self.args, 'update')

	def update(self, data):
		#self.epi_gather.update(data)
		self.update_gather.update(data)
