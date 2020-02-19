import os
import math
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
import copy
import tensorflow as tf
import tensorflow_probability as tfp
import random
from baselines.proportional import Experience


class Count:

	def __init__(self, n, shape, len=2):
		self.n = n
		self.shape = shape
		self.f = np.zeros(shape + [1], dtype='int32')
		self.len = len

	def output(self, num):
		f = self.f
		for i in range(self.n):
			if isinstance(num[i], np.ndarray):
				for j in range(self.len):
					f = f[num[i][j]]
			else:
				f = f[num[i]]
		return np.squeeze(f)

	def add(self, num):
		f = self.f
		for i in range(self.n):
			if isinstance(num[i], np.ndarray):
				for j in range(self.len):
					f = f[num[i][j]]
			else:
				f = f[num[i]]

		f[0] += 1


class Count_p:

	def __init__(self, n, shape, n_1, m_1, len=2):
		self.n_1 = n_1
		self.count_1 = Count(n, shape, len=len)
		self.count_2 = Count(n - n_1, shape[m_1:], len=len)

	def output(self, num):
		ans = self.count_1.output(num)
		if ans == 0:
			return 0
		else:
			res = self.count_2.output(num[self.n_1:])
			if res == 0:
				print(num)
				print(ans)
			ans = 1. * ans / res
			return ans

	def add(self, num):
		self.count_1.add(num)
		self.count_2.add(num[self.n_1:])


class Count_log_p:

	def __init__(self, n, shape, n_1, n_2, m_1, m_2):
		self.n_1 = n_1
		self.count_1 = Count_p(n, shape, n_2, m_2)
		self.count_2 = Count_p(n - n_1, shape[:-m_1], n_2, m_2)

	def output(self, num):
		ans = self.count_1.output(num)
		if ans == 0:
			return 0
		else:
			ans = 1. * ans / self.count_2.output(num[:-self.n_1])
			return ans

	def add(self, num):
		self.count_1.add(num)
		self.count_2.add(num[:-self.n_1])


class Count_div_p:

	def __init__(self, n, shape, n_1, n_2, m_1, m_2):
		self.n_1 = n_1
		self.count_1 = Count_p(n - n_1, shape[:-m_1], n_2, m_2)
		self.count_2 = Count_p(n, shape, n_2, m_2)

	def output(self, num):
		ans = self.count_2.output(num)
		if (ans == 0):
			return 1
		ans = 1. * self.count_1.output(num[:-self.n_1]) / ans
		return ans

	def add(self, num):
		self.count_1.add(num[:-self.n_1])
		self.count_2.add(num)


class Test_Count_div_p:

	def __init__(self, n, shape, n_1, n_2, m_1, m_2):
		self.n_1 = n_1
		self.count_1 = Count_p(n - n_1, shape[:-m_1], n_2, m_2)
		self.count_2 = Count_p(n, shape, n_2, m_2)

	def output(self, num):
		ans_1 = self.count_1.output(num[:-self.n_1])
		ans_2 = self.count_2.output(num)
		if ans_2 == 0:
			return 1, ans_1, ans_2
		ans = 1. * ans_1 / ans_2
		return ans, ans_1, ans_2

	def add(self, num):
		self.count_1.add(num[:-self.n_1])
		self.count_2.add(num)


class Hash_core:

	def __init__(self):
		self.mo1 = 1007
		self.mo2 = 1000000000000007
		self.buffer_size = 100000007
		self.buffer = {}

	def change(self, data):
		ans = 0
		for x in data:
			ans = (ans * self.mo1 + x + self.mo2) % self.mo2
		res = ans % self.buffer_size
		return res, ans

	def update(self, data):
		index, data = self.change(data)
		if data not in self.buffer:
			self.buffer[data] = {index: 0}
		else:
			self.buffer[data][index] += 1

	def output(self, data):
		index, data = self.change(data)
		if data not in self.buffer:
			return 0
		return self.buffer[data][index]


class Hash:

	def __init__(self, n, shape, len=2):
		self.n = n
		self.shape = shape
		self.f = Hash_core()
		self.len = len

	def get(self, num):
		index = []
		for i in range(self.n):
			if isinstance(num[i], np.ndarray):
				for j in range(self.len):
					index.append(num[i][j])
			else:
				index.append(num[i])
		return index

	def output(self, num):
		index = self.get(num)
		return self.f.output(index)

	def add(self, num):
		index = self.get(num)
		self.f.update(index)


class Hash_p:

	def __init__(self, n, shape, n_1, m_1, len=2):
		self.n_1 = n_1
		self.count_1 = Hash(n, shape, len=len)
		self.count_2 = Hash(n - n_1, shape[m_1:], len=len)

	def output(self, num):
		ans = self.count_1.output(num)
		if ans == 0:
			return 0
		else:
			res = self.count_2.output(num[self.n_1:])
			if res == 0:
				print(num)
				print(ans)
			ans = 1. * ans / res
			return ans

	def add(self, num):
		self.count_1.add(num)
		self.count_2.add(num[self.n_1:])


class Hash_div_p:

	def __init__(self, n, shape, n_1, n_2, m_1, m_2, len=2):
		self.n_1 = n_1
		self.count_1 = Hash_p(n - n_1, shape[:-m_1], n_2, m_2, len=len)
		self.count_2 = Hash_p(n, shape, n_2, m_2, len=len)

	def output(self, num):
		ans = self.count_2.output(num)
		if ans == 0:
			return 1
		ans = 1. * self.count_1.output(num[:-self.n_1]) / ans
		return ans

	def add(self, num):
		self.count_1.add(num[:-self.n_1])
		self.count_2.add(num)


def add_shape(n, size):
	shape = []
	for i in range(n):
		shape += [size]
	return shape


def mk(x, y):
	return [np.array([x, y])]


def mk1(x):
	return [x]


class Key_points:

	def __init__(self, size, a_size, arg, is_print):

		self.arg = arg
		self.is_print = is_print

		if (self.is_print):
			self.figure_path = self.arg.save_path + 'sub-goals/'
			if os.path.exists(self.figure_path):
				shutil.rmtree(self.figure_path)
			os.makedirs(self.figure_path)
		# print(self.figure_path)

		self.size = size
		self.a_size = a_size
		self.range = 3

		log_shape = \
			add_shape(2, self.range) + add_shape(2, size) + add_shape(1, a_size) + add_shape(2, size) + add_shape(1,
			                                                                                                      a_size)

		self.log_1 = Count_log_p(5, log_shape, 2, 1, 3, 2)
		self.log_2 = Count_log_p(5, log_shape, 2, 1, 3, 2)

		p_shape = \
			add_shape(2, size) + add_shape(1, a_size) + add_shape(2, self.range) + add_shape(2, size) + add_shape(1,
			                                                                                                      a_size)

		self.p_1 = Count_p(5, p_shape, 1, 2)
		self.p_2 = Count_p(5, p_shape, 1, 2)

		other_p_shape = \
			add_shape(2, size) + add_shape(1, a_size) + add_shape(2, self.range) + add_shape(2, size) + add_shape(1,
			                                                                                                      a_size)

		self.other_p_1 = Count_p(5, other_p_shape, 1, 2)
		self.other_p_2 = Count_p(5, other_p_shape, 1, 2)

		check_p_shape = add_shape(2, self.range) + add_shape(2, size) + add_shape(1, a_size)

		self.check_p_1 = Count(3, check_p_shape)
		self.check_p_2 = Count(3, check_p_shape)

	def calc_del(self, s_t, next_s_t):
		return next_s_t - s_t + 1

	def update(self, state, next_state):

		s_t, a_t = state
		s_t_1, s_t_2 = s_t
		a_t_1, a_t_2 = a_t

		next_s_t = next_state[0]
		next_s_t_1, next_s_t_2 = next_s_t

		del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
		del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

		num_1 = [del_s_t_2, s_t_2, a_t_2, s_t_1, a_t_1]
		num_2 = [del_s_t_1, s_t_1, a_t_1, s_t_2, a_t_2]

		self.log_1.add(num_1)
		self.log_2.add(num_2)

		p_num_1 = [s_t_1, a_t_1, del_s_t_2, s_t_2, a_t_2]
		p_num_2 = [s_t_2, a_t_2, del_s_t_1, s_t_1, a_t_1]

		self.p_1.add(p_num_1)
		self.p_2.add(p_num_2)

		other_p_num_1 = [s_t_2, a_t_2, del_s_t_2, s_t_1, a_t_1]
		other_p_num_2 = [s_t_1, a_t_1, del_s_t_1, s_t_2, a_t_2]

		self.other_p_1.add(other_p_num_1)
		self.other_p_2.add(other_p_num_2)

		check_p_num_1 = [del_s_t_1, s_t_1, a_t_1]
		check_p_num_2 = [del_s_t_2, s_t_2, a_t_2]

		self.check_p_1.add(check_p_num_1)
		self.check_p_2.add(check_p_num_2)

	def output(self, state, next_state, agent_index):

		s_t, a_t = state
		s_t_1, s_t_2 = s_t
		a_t_1, a_t_2 = a_t

		next_s_t = next_state[0]
		next_s_t_1, next_s_t_2 = next_s_t

		del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
		del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

		num_1 = [del_s_t_2, s_t_2, a_t_2, s_t_1, a_t_1]
		num_2 = [del_s_t_1, s_t_1, a_t_1, s_t_2, a_t_2]

		if (self.arg.symmetry):
			p_log = self.log_1.output(num_1)

			if p_log != 0:
				p_log = math.log(p_log)
			# if (p_log != 0):
			#	print(p_log)

			p_log_2 = self.log_2.output(num_2)

			if p_log_2 != 0:
				p_log_2 = math.log(p_log_2)
			# if (p_log_2 != 0):
			#	print(p_log_2)

			return p_log + p_log_2

		if (agent_index == 0):
			p_log = self.log_1.output(num_1)

			if p_log != 0:
				p_log = math.log(p_log)

			return p_log
		else:
			p_log_2 = self.log_2.output(num_2)

			if p_log_2 != 0:
				p_log_2 = math.log(p_log_2)

			return p_log_2

	# return p_log + p_log_2

	def show(self, e):

		if (self.is_print == 0):
			return
		print("start showing, round %d: " % e)

		ans_1 = np.zeros([self.size, self.size])
		ans_2 = np.zeros([self.size, self.size])
		res_1 = np.zeros([self.size, self.size])
		res_2 = np.zeros([self.size, self.size])

		for i1 in range(self.size):
			for j1 in range(self.size):
				for k1 in range(self.a_size):
					for i3 in range(i1 - 1, i1 + 2):
						for j3 in range(j1 - 1, j1 + 2):
							check_p_num_2 = mk(i3 - i1 + 1, j3 - j1 + 1) + mk(i1, j1) + mk1(k1)
							if (np.squeeze(self.check_p_2.output(check_p_num_2)) > 0):
								for i2 in range(self.size):
									for j2 in range(self.size):
										for k2 in range(self.a_size):
											p_num_1 = mk(i2, j2) + mk1(k2) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
											          mk(i1, j1) + mk1(k1)
											prob = self.p_1.output(p_num_1)

											num_1 = mk(i3 - i1 + 1, j3 - j1 + 1) + mk(i1, j1) + mk1(k1) + \
											        mk(i2, j2) + mk1(k2)
											p_log = self.log_1.output(num_1)

											p_num_2 = mk(i1, j1) + mk1(k1) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
											          mk(i2, j2) + mk1(k2)
											prob_2 = self.other_p_1.output(p_num_2)

											if (p_log != 0 and p_log != 1):
												ans_1[i2][j2] += prob * math.log(p_log)
												res_2[i1][j1] += prob_2 * math.log(p_log)

							check_p_num_1 = mk(i3 - i1 + 1, j3 - j1 + 1) + mk(i1, j1) + mk1(k1)
							if (np.squeeze(self.check_p_1.output(check_p_num_1)) > 0):
								for i2 in range(self.size):
									for j2 in range(self.size):
										for k2 in range(self.a_size):
											p_num_2 = mk(i2, j2) + mk1(k2) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
											          mk(i1, j1) + mk1(k1)
											prob = self.p_2.output(p_num_2)

											num_2 = mk(i3 - i1 + 1, j3 - j1 + 1) + mk(i1, j1) + mk1(k1) + \
											        mk(i2, j2) + mk1(k2)
											p_log = self.log_2.output(num_2)

											p_num_2 = mk(i1, j1) + mk1(k1) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
											          mk(i2, j2) + mk1(k2)
											prob_2 = self.other_p_2.output(p_num_2)

											if p_log != 0 and p_log != 1:
												ans_2[i2][j2] += prob * math.log(p_log)
												res_1[i1][j1] += prob_2 * math.log(p_log)

		print("start drawing round %d" % e)

		figure = plt.figure(figsize=(16, 10))
		ax1 = figure.add_subplot(2, 2, 1)
		ax2 = figure.add_subplot(2, 2, 3)
		ax3 = figure.add_subplot(2, 2, 2)
		ax4 = figure.add_subplot(2, 2, 4)

		ax1.imshow(np.log(ans_1 + 1))
		ax2.imshow(np.log(ans_2 + 1))
		ax3.imshow(np.log(res_1 + 1))
		ax4.imshow(np.log(res_2 + 1))

		figure.savefig('%s/%i.png' % (self.figure_path, e))
		plt.close(figure)


class Dec:
	def __init__(self, size, n_agent, args):
		self.env_n_dim = args.env_n_dim
		self.size = size
		self.n_agent = n_agent
		self.args = args
		self.visited = [np.zeros([args.size, args.size, 2]) for _ in range(self.n_agent)]
		self.visited_old = [np.zeros([args.size, args.size, 2]) for _ in range(self.n_agent)]

		if self.env_n_dim == 1:
			self.show_visited = np.zeros([args.size, args.size])
			self.show_visited_old = np.zeros([args.size, args.size])

	def update(self, state, is_door_open):
		# For heat map
		for i in range(self.n_agent):
			self.visited[i][state[i][0]][state[i][1]][int(is_door_open)] += 1

		if self.env_n_dim == 1:
			self.show_visited[state[0][0]][state[1][0]] += 1

	def output(self, state, i):
		return 1. / np.sqrt(np.sum(self.visited[i][state[i][0]][state[i][1]]) + 1)

	def show(self, path, e):
		if self.env_n_dim == 2:
			figure = plt.figure(figsize=(16, 10))

			ax1 = figure.add_subplot(2, 6, 1)
			ax2 = figure.add_subplot(2, 6, 2)
			ax3 = figure.add_subplot(2, 6, 3)
			ax4 = figure.add_subplot(2, 6, 4)
			ax5 = figure.add_subplot(2, 6, 5)
			ax6 = figure.add_subplot(2, 6, 6)
			ax7 = figure.add_subplot(2, 6, 7)
			ax8 = figure.add_subplot(2, 6, 8)
			ax9 = figure.add_subplot(2, 6, 9)
			ax10 = figure.add_subplot(2, 6, 10)
			ax11 = figure.add_subplot(2, 6, 11)
			ax12 = figure.add_subplot(2, 6, 12)

			ax1.imshow(np.log(self.visited[0][:, :, 0] + 1))
			ax2.imshow(np.log(self.visited[0][:, :, 0] - self.visited_old[0][:, :, 0] + 1))
			ax3.imshow(np.log(self.visited[0][:, :, 1] + 1))
			ax4.imshow(np.log((self.visited[0][:, :, 1] - self.visited_old[0][:, :, 1] + 1)))
			ax5.imshow(np.log(np.sum(self.visited[0], axis=2) + 1))
			ax6.imshow(np.log(np.sum(self.visited[0], axis=2) - np.sum(self.visited_old[0], axis=2) + 1))

			ax7.imshow(np.log(self.visited[1][:, :, 0] + 1))
			ax8.imshow(np.log(self.visited[1][:, :, 0] - self.visited_old[1][:, :, 0] + 1))
			ax9.imshow(np.log(self.visited[1][:, :, 1] + 1))
			ax10.imshow(np.log((self.visited[1][:, :, 1] - self.visited_old[1][:, :, 1] + 1)))
			ax11.imshow(np.log(np.sum(self.visited[1], axis=2) + 1))
			ax12.imshow(np.log(np.sum(self.visited[1], axis=2) - np.sum(self.visited_old[1], axis=2) + 1))

			figure.savefig('%s/%i.png' % (path, e))
			plt.close(figure)

			self.visited_old = [v.copy() for v in self.visited]
		elif self.env_n_dim == 1:
			figure = plt.figure(figsize=(16, 10))

			ax1 = figure.add_subplot(2, 1, 1)
			ax2 = figure.add_subplot(2, 1, 2)

			ax1.imshow(np.log(self.show_visited + 1))
			ax2.imshow(np.log(self.show_visited - self.show_visited_old + 1))

			figure.savefig('%s/%i.png' % (path, e))
			plt.close(figure)

			self.visited_old = [v.copy() for v in self.visited]
			self.show_visited_old = self.show_visited.copy()


class Cen:
	def __init__(self, size, n_agent, args):
		self.env_n_dim = args.env_n_dim
		self.size = size
		self.n_agent = n_agent
		self.args = args
		self.visited = np.zeros([args.size, args.size, args.size, args.size])

	def update(self, state):
		self.visited[state[0][0]][state[0][1]][state[1][0]][state[1][1]] += 1

	def output(self, state):
		return 1. / np.sqrt(self.visited[state[0][0]][state[0][1]][state[1][0]][state[1][1]] + 1)

	def show(self, path, e):
		pass


class C_points:

	def __init__(self, size, a_size, arg, is_print):

		self.arg = arg
		self.is_print = is_print

		if (self.is_print):
			self.figure_path = self.arg.save_path + 'sub-goals/'
			if os.path.exists(self.figure_path):
				shutil.rmtree(self.figure_path)
			os.makedirs(self.figure_path)
		# print(self.figure_path)

		self.size = size
		self.a_size = a_size
		self.range = 3

		p_c_shape = add_shape(2, self.range) + add_shape(2, size) + add_shape(1, a_size)

		self.p_c_1 = Count_p(3, p_c_shape, 1, 2)
		self.p_c_2 = Count_p(3, p_c_shape, 1, 2)

		p_shape = add_shape(2, size) + add_shape(1, a_size) + add_shape(2, self.range) + \
		          add_shape(2, size) + add_shape(1, a_size)

		self.p_1 = Count_p(5, p_shape, 1, 2)
		self.p_2 = Count_p(5, p_shape, 1, 2)

		other_p_shape = add_shape(2, size) + add_shape(1, a_size) + add_shape(2, self.range) + \
		                add_shape(2, size) + add_shape(1, a_size)

		self.other_p_1 = Count_p(5, other_p_shape, 1, 2)
		self.other_p_2 = Count_p(5, other_p_shape, 1, 2)

	def calc_del(self, s_t, next_s_t):
		return next_s_t - s_t + 1

	def update(self, state, next_state):

		s_t, a_t = state
		s_t_1, s_t_2 = s_t
		a_t_1, a_t_2 = a_t

		next_s_t = next_state[0]
		next_s_t_1, next_s_t_2 = next_s_t

		del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
		del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

		c_num_1 = [del_s_t_2, s_t_2, a_t_2]
		c_num_2 = [del_s_t_1, s_t_1, a_t_1]

		self.p_c_1.add(c_num_1)
		self.p_c_2.add(c_num_2)

		p_num_1 = [s_t_1, a_t_1, del_s_t_2, s_t_2, a_t_2]
		p_num_2 = [s_t_2, a_t_2, del_s_t_1, s_t_1, a_t_1]

		self.p_1.add(p_num_1)
		self.p_2.add(p_num_2)

		other_p_num_1 = [s_t_2, a_t_2, del_s_t_2, s_t_1, a_t_1]
		other_p_num_2 = [s_t_1, a_t_1, del_s_t_1, s_t_2, a_t_2]

		self.other_p_1.add(other_p_num_1)
		self.other_p_2.add(other_p_num_2)

	def output(self, state, next_state, agent_index):

		s_t, a_t = state
		s_t_1, s_t_2 = s_t
		a_t_1, a_t_2 = a_t

		next_s_t = next_state[0]
		next_s_t_1, next_s_t_2 = next_s_t

		del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
		del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

		c_num_1 = [del_s_t_2, s_t_2, a_t_2]
		c_num_2 = [del_s_t_1, s_t_1, a_t_1]

		if (self.arg.symmetry):
			p_1 = -self.p_c_1.output(c_num_1)
			p_2 = -self.p_c_2.output(c_num_2)
			return p_1 + p_2

		if (agent_index == 0):
			p_1 = self.p_c_1.output(c_num_1)
			return p_1
		else:
			p_2 = self.p_c_2.output(c_num_2)
			return p_2

	def show(self, e):

		if (self.is_print == 0):
			return
		print("start showing, round %d: " % e)

		ans_1 = np.zeros([self.size, self.size])
		ans_2 = np.zeros([self.size, self.size])
		res_1 = np.zeros([self.size, self.size])
		res_2 = np.zeros([self.size, self.size])

		for i1 in range(self.size):
			for j1 in range(self.size):
				for k1 in range(self.a_size):
					for i3 in range(i1 - 1, i1 + 2):
						for j3 in range(j1 - 1, j1 + 2):
							c_num = mk(i3 - i1 + 1, j3 - j1 + 1) + mk(i1, j1) + mk1(k1)
							p_1 = 1 - self.p_c_1.output(c_num)
							p_2 = 1 - self.p_c_2.output(c_num)
							if (p_1 < 1):
								for i2 in range(self.size):
									for j2 in range(self.size):
										for k2 in range(self.a_size):
											p_num_1 = mk(i2, j2) + mk1(k2) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
											          mk(i1, j1) + mk1(k1)
											prob = self.p_1.output(p_num_1)

											p_num_2 = mk(i1, j1) + mk1(k1) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
											          mk(i2, j2) + mk1(k2)
											prob_2 = self.other_p_1.output(p_num_2)

											ans_1[i2][j2] += prob * p_1
											res_2[i1][j1] += prob_2 * p_1

							if (p_2 < 1):
								for i2 in range(self.size):
									for j2 in range(self.size):
										for k2 in range(self.a_size):
											p_num_1 = mk(i2, j2) + mk1(k2) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
											          mk(i1, j1) + mk1(k1)
											prob = self.p_2.output(p_num_1)

											p_num_2 = mk(i1, j1) + mk1(k1) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
											          mk(i2, j2) + mk1(k2)
											prob_2 = self.other_p_2.output(p_num_2)

											ans_2[i2][j2] += prob * p_2
											res_1[i1][j1] += prob_2 * p_2

		print("start drawing round %d" % e)

		figure = plt.figure(figsize=(16, 10))
		ax1 = figure.add_subplot(2, 2, 1)
		ax2 = figure.add_subplot(2, 2, 3)
		ax3 = figure.add_subplot(2, 2, 2)
		ax4 = figure.add_subplot(2, 2, 4)

		ax1.imshow(np.log(ans_1 + 1))
		ax2.imshow(np.log(ans_2 + 1))
		ax3.imshow(np.log(res_1 + 1))
		ax4.imshow(np.log(res_2 + 1))

		figure.savefig('%s/%i.png' % (self.figure_path, e))
		plt.close(figure)


class Island_C_points:

	def __init__(self, size, a_size, arg, is_print):

		self.arg = arg
		self.is_print = is_print

		if (self.is_print):
			self.figure_path = self.arg.save_path + 'sub-goals/'
			if os.path.exists(self.figure_path):
				shutil.rmtree(self.figure_path)
			os.makedirs(self.figure_path)
		# print(self.figure_path)

		self.size = size
		self.a_size = a_size
		self.range = 5

		p_c_shape = add_shape(4, self.range) + add_shape(4, size) + add_shape(1, a_size)

		self.p_c_1 = Count_p(3, p_c_shape, 1, 4, len=4)
		self.p_c_2 = Count_p(3, p_c_shape, 1, 4, len=4)

	def calc_del(self, s_t, next_s_t):
		return np.minimum(np.maximum(next_s_t - s_t, -2), 2) + 2

	def update(self, state, next_state):

		s_t, a_t = state
		s_t_1, s_t_2 = s_t
		a_t_1, a_t_2 = a_t

		next_s_t = next_state[0]
		next_s_t_1, next_s_t_2 = next_s_t

		del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
		del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

		c_num_1 = [del_s_t_2, s_t_2, a_t_2]
		c_num_2 = [del_s_t_1, s_t_1, a_t_1]

		self.p_c_1.add(c_num_1)
		self.p_c_2.add(c_num_2)

	def output(self, state, next_state, agent_index):

		s_t, a_t = state
		s_t_1, s_t_2 = s_t
		a_t_1, a_t_2 = a_t

		next_s_t = next_state[0]
		next_s_t_1, next_s_t_2 = next_s_t

		del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
		del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

		c_num_1 = [del_s_t_2, s_t_2, a_t_2]
		c_num_2 = [del_s_t_1, s_t_1, a_t_1]

		if (self.arg.symmetry):
			p_1 = -self.p_c_1.output(c_num_1)
			p_2 = -self.p_c_2.output(c_num_2)
			return p_1 + p_2

		if (agent_index == 0):
			p_1 = self.p_c_1.output(c_num_1)
			return p_1
		else:
			p_2 = self.p_c_2.output(c_num_2)
			return p_2

	def show(self, e):

		pass


class Island_Dec:

	def __init__(self, size, size_2, n_agent, args):
		self.env_n_dim = args.env_n_dim
		self.size = size
		self.n_agent = n_agent
		self.args = args

		self.not_run = self.args.s_alg_name == 'noisy' or self.args.s_alg_name == 'cen'

		if self.not_run:
			return

		self.visited = [np.zeros([size, size, 2]) for _ in range(self.n_agent)]
		self.visited_old = [np.zeros([size, size, 2]) for _ in range(self.n_agent)]
		self.visited_dec = [np.zeros([size, size, size_2, size, size]) for _ in range(self.n_agent)]

	def is_under_attack(self, state):
		if state[3] == 0:
			return 0
		s_t = state[:2]
		s_e = state[3: 5]
		del_s_t = s_t - s_e
		if -1 <= del_s_t[0] <= 1 and -1 <= del_s_t[1] <= 1:
			return 1
		else:
			return 0

	def update(self, state):

		if self.not_run:
			return

		# For heat map
		for i in range(self.n_agent):
			t_attack = self.is_under_attack(state[i])
			self.visited[i][state[i][0]][state[i][1]][t_attack] += 1
			self.visited_dec[i][state[i][0]][state[i][1]][state[i][2]][state[i][3]][state[i][4]] += 1

	def output(self, state, i):

		if self.not_run:
			return 0

		return 1. / np.sqrt(self.visited_dec[i][state[i][0]][state[i][1]][state[i][2]]
		                    [state[i][3]][state[i][4]] + 1)

	def show(self, path, e):

		if self.not_run:
			return

		figure = plt.figure(figsize=(16, 10))

		ax1 = figure.add_subplot(2, 6, 1)
		ax2 = figure.add_subplot(2, 6, 2)
		ax3 = figure.add_subplot(2, 6, 3)
		ax4 = figure.add_subplot(2, 6, 4)
		ax5 = figure.add_subplot(2, 6, 5)
		ax6 = figure.add_subplot(2, 6, 6)
		ax7 = figure.add_subplot(2, 6, 7)
		ax8 = figure.add_subplot(2, 6, 8)
		ax9 = figure.add_subplot(2, 6, 9)
		ax10 = figure.add_subplot(2, 6, 10)
		ax11 = figure.add_subplot(2, 6, 11)
		ax12 = figure.add_subplot(2, 6, 12)

		ax1.imshow(np.log(self.visited[0][:, :, 0] + 1))
		ax2.imshow(np.log(self.visited[0][:, :, 0] - self.visited_old[0][:, :, 0] + 1))
		ax3.imshow(np.log(self.visited[0][:, :, 1] + 1))
		ax4.imshow(np.log((self.visited[0][:, :, 1] - self.visited_old[0][:, :, 1] + 1)))
		ax5.imshow(np.log(np.sum(self.visited[0], axis=2) + 1))
		ax6.imshow(np.log(np.sum(self.visited[0], axis=2) - np.sum(self.visited_old[0], axis=2) + 1))

		ax7.imshow(np.log(self.visited[1][:, :, 0] + 1))
		ax8.imshow(np.log(self.visited[1][:, :, 0] - self.visited_old[1][:, :, 0] + 1))
		ax9.imshow(np.log(self.visited[1][:, :, 1] + 1))
		ax10.imshow(np.log((self.visited[1][:, :, 1] - self.visited_old[1][:, :, 1] + 1)))
		ax11.imshow(np.log(np.sum(self.visited[1], axis=2) + 1))
		ax12.imshow(np.log(np.sum(self.visited[1], axis=2) - np.sum(self.visited_old[1], axis=2) + 1))

		figure.savefig('%s/%i.png' % (path, e))
		plt.close(figure)

		self.visited_old = [v.copy() for v in self.visited]


class Island_Cen:
	def __init__(self, size, size_2, n_agent, args):
		self.env_n_dim = args.env_n_dim
		self.size = size
		self.n_agent = n_agent
		self.args = args
		self.visited = np.zeros([size, size, size, size, size, size])

	def update(self, state):
		self.visited[state[0][0]][state[0][1]][state[0][3]][state[0][4]][state[1][0]][state[1][1]] += 1

	def output(self, state):
		return 1. / np.sqrt(self.visited[state[0][0]][state[0][1]][state[0][3]][state[0][4]]
		                    [state[1][0]][state[1][1]] + 1)

	def show(self, path, e):
		pass


class x_Island_Cen:
	def __init__(self, size, size_2, n_agent, args):

		self.env_n_dim = args.env_n_dim
		self.size = size
		self.n_agent = n_agent
		self.args = args

		self.not_run = self.args.s_alg_name == 'noisy' or self.args.s_alg_name == 'coor_t' or \
		               self.args.s_alg_name == 'coor_r_tv' or self.args.s_alg_name == 'dec' \
		               or self.args.s_alg_name == 'coor_r' or self.args.s_alg_name == 'coor_v'

		if self.not_run:
			return

		self.visited = [[np.zeros([size, size, size, size, size, size]) for i in range(self.n_agent)]
		                for j in range(self.n_agent)]

	def update(self, state):

		if self.not_run:
			return

		for i in range(self.n_agent):
			for j in range(self.n_agent):
				if i != j:
					self.visited[i][j][state[i][0]][state[i][1]][state[i][3]][state[i][4]][state[j][0]][state[j][1]] \
						+= 1

	def output(self, state, i):

		if self.not_run:
			return 0

		res = 0
		for j in range(self.n_agent):
			if i != j:
				res += 1. / np.sqrt(self.visited[i][j][state[i][0]][state[i][1]][state[i][3]][state[i][4]]
				                    [state[j][0]][state[j][1]] + 1)
		res /= self.n_agent - 1
		return res

	def show(self, path, e):
		pass


class Pushball_Dec:

	def __init__(self, size, n_agent, args):
		self.env_n_dim = args.env_n_dim
		self.size = size
		self.n_agent = n_agent
		self.args = args
		self.visited = [np.zeros([size, size]) for _ in range(self.n_agent)]
		self.visited_old = [np.zeros([size, size]) for _ in range(self.n_agent)]
		self.visited_dec = [np.zeros([size, size, size, size]) for _ in range(self.n_agent)]

	def update(self, state):
		# For heat map
		for i in range(self.n_agent):
			self.visited[i][state[i][0]][state[i][1]] += 1
			self.visited_dec[i][state[i][0]][state[i][1]][state[i][2]][state[i][3]] += 1

	def output(self, state, i):
		return 1. / np.sqrt(self.visited_dec[i][state[i][0]][state[i][1]][state[i][2]][state[i][3]] + 1)

	def show(self, path, e):
		if self.env_n_dim == 2:
			figure = plt.figure(figsize=(16, 10))

			ax1 = figure.add_subplot(2, 2, 1)
			ax2 = figure.add_subplot(2, 2, 2)
			ax3 = figure.add_subplot(2, 2, 3)
			ax4 = figure.add_subplot(2, 2, 4)

			ax1.imshow(np.log(self.visited[0] + 1))
			ax2.imshow(np.log(self.visited[0] - self.visited_old[0] + 1))

			ax3.imshow(np.log(self.visited[1] + 1))
			ax4.imshow(np.log(self.visited[1] - self.visited_old[1] + 1))

			figure.savefig('%s/%i.png' % (path, e))
			plt.close(figure)

			self.visited_old = [v.copy() for v in self.visited]


class Pushball_Cen:
	def __init__(self, size, n_agent, args):
		self.env_n_dim = args.env_n_dim
		self.size = size
		self.n_agent = n_agent
		self.args = args
		self.visited = np.zeros([size, size, size, size, size, size])

	def update(self, state):
		self.visited[state[0][0]][state[0][1]][state[0][2]][state[0][3]][state[1][0]][state[1][1]] += 1

	def output(self, state):
		return 1. / np.sqrt(self.visited[state[0][0]][state[0][1]][state[0][2]][state[0][3]]
		                    [state[1][0]][state[1][1]] + 1)

	def show(self, path, e):
		pass


class Appro_C_points:

	def __init__(self, size, a_size, arg, is_print):

		self.arg = arg
		self.is_print = is_print and not self.arg.s_data_gather

		self.not_run = self.arg.s_alg_name == 'noisy' or self.arg.s_alg_name == 'dec' or \
		               self.arg.s_alg_name == 'cen'
		if self.not_run:
			return

		if (self.is_print):
			self.figure_path = self.arg.save_path + 'sub-goals/'
			if os.path.exists(self.figure_path):
				shutil.rmtree(self.figure_path)
			os.makedirs(self.figure_path)
		# print(self.figure_path)

		self.size = size
		self.a_size = a_size
		self.range = 3

		p_c_shape = \
			add_shape(2, self.range) + add_shape(2, size) + \
			add_shape(1, a_size) + add_shape(2, size) + add_shape(1, a_size)

		self.p_c_1 = Count_div_p(5, p_c_shape, 2, 1, 3, 2)
		self.p_c_2 = Count_div_p(5, p_c_shape, 2, 1, 3, 2)

		if self.is_print:
			p_shape = \
				add_shape(2, size) + add_shape(1, a_size) + \
				add_shape(2, self.range) + add_shape(2, size) + add_shape(1, a_size)

			self.p_1 = Count_p(5, p_shape, 1, 2)
			self.p_2 = Count_p(5, p_shape, 1, 2)

			other_p_shape = \
				add_shape(2, size) + add_shape(1, a_size) + \
				add_shape(2, self.range) + add_shape(2, size) + add_shape(1, a_size)

			self.other_p_1 = Count_p(5, other_p_shape, 1, 2)
			self.other_p_2 = Count_p(5, other_p_shape, 1, 2)

			check_p_shape = add_shape(2, self.range) + add_shape(2, size) + add_shape(1, a_size)

			self.check_p_1 = Count(3, check_p_shape)
			self.check_p_2 = Count(3, check_p_shape)

	def calc_del(self, s_t, next_s_t):
		return next_s_t - s_t + 1

	def update(self, state, next_state):

		if self.not_run:
			return

		s_t, a_t = state
		s_t_1, s_t_2 = s_t
		a_t_1, a_t_2 = a_t

		next_s_t = next_state[0]
		next_s_t_1, next_s_t_2 = next_s_t

		del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
		del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

		c_num_1 = [del_s_t_2, s_t_2, a_t_2, s_t_1, a_t_1]
		c_num_2 = [del_s_t_1, s_t_1, a_t_1, s_t_2, a_t_2]

		self.p_c_1.add(c_num_1)
		self.p_c_2.add(c_num_2)

		if self.is_print:
			p_num_1 = [s_t_1, a_t_1, del_s_t_2, s_t_2, a_t_2]
			p_num_2 = [s_t_2, a_t_2, del_s_t_1, s_t_1, a_t_1]

			self.p_1.add(p_num_1)
			self.p_2.add(p_num_2)

			other_p_num_1 = [s_t_2, a_t_2, del_s_t_2, s_t_1, a_t_1]
			other_p_num_2 = [s_t_1, a_t_1, del_s_t_1, s_t_2, a_t_2]

			self.other_p_1.add(other_p_num_1)
			self.other_p_2.add(other_p_num_2)

			check_p_num_1 = [del_s_t_1, s_t_1, a_t_1]
			check_p_num_2 = [del_s_t_2, s_t_2, a_t_2]

			self.check_p_1.add(check_p_num_1)
			self.check_p_2.add(check_p_num_2)

	def output(self, state, next_state, agent_index):

		if self.not_run:
			return 0

		s_t, a_t = state
		s_t_1, s_t_2 = s_t
		a_t_1, a_t_2 = a_t

		next_s_t = next_state[0]
		next_s_t_1, next_s_t_2 = next_s_t

		del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
		del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

		c_num_1 = [del_s_t_2, s_t_2, a_t_2, s_t_1, a_t_1]
		c_num_2 = [del_s_t_1, s_t_1, a_t_1, s_t_2, a_t_2]

		if (self.arg.symmetry):
			p_1 = 1 - self.p_c_1.output(c_num_1)
			p_2 = 1 - self.p_c_2.output(c_num_2)
			return p_1 + p_2

		if (agent_index == 0):
			p_1 = 1 - self.p_c_1.output(c_num_1)
			return p_1
		else:
			p_2 = 1 - self.p_c_2.output(c_num_2)
			return p_2

	def show(self, e):

		if not self.is_print or self.not_run:
			return

		print("start showing, round %d: " % e)

		ans_1 = np.zeros([self.size, self.size])
		ans_2 = np.zeros([self.size, self.size])
		res_1 = np.zeros([self.size, self.size])
		res_2 = np.zeros([self.size, self.size])

		for i1 in range(self.size):
			for j1 in range(self.size):
				for k1 in range(self.a_size):
					for i3 in range(i1 - 1, i1 + 2):
						for j3 in range(j1 - 1, j1 + 2):
							check_p_num_2 = mk(i3 - i1 + 1, j3 - j1 + 1) + mk(i1, j1) + mk1(k1)
							if self.check_p_2.output(check_p_num_2) > 0:
								for i2 in range(self.size):
									for j2 in range(self.size):
										for k2 in range(self.a_size):
											p_num_1 = mk(i2, j2) + mk1(k2) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
											          mk(i1, j1) + mk1(k1)
											prob = self.p_1.output(p_num_1)

											num_1 = mk(i3 - i1 + 1, j3 - j1 + 1) + mk(i1, j1) + mk1(k1) + \
											        mk(i2, j2) + mk1(k2)
											p_1 = 1 - self.p_c_1.output(num_1)

											p_num_2 = mk(i1, j1) + mk1(k1) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
											          mk(i2, j2) + mk1(k2)
											prob_2 = self.other_p_1.output(p_num_2)

											ans_1[i2][j2] += prob * p_1
											res_2[i1][j1] += prob_2 * p_1

							check_p_num_1 = mk(i3 - i1 + 1, j3 - j1 + 1) + mk(i1, j1) + mk1(k1)
							if self.check_p_1.output(check_p_num_1) > 0:
								for i2 in range(self.size):
									for j2 in range(self.size):
										for k2 in range(self.a_size):
											p_num_2 = mk(i2, j2) + mk1(k2) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
											          mk(i1, j1) + mk1(k1)
											prob = self.p_2.output(p_num_2)

											num_2 = mk(i3 - i1 + 1, j3 - j1 + 1) + mk(i1, j1) + mk1(k1) + \
											        mk(i2, j2) + mk1(k2)
											p_2 = 1 - self.p_c_2.output(num_2)

											p_num_2 = mk(i1, j1) + mk1(k1) + mk(i3 - i1 + 1, j3 - j1 + 1) + \
											          mk(i2, j2) + mk1(k2)
											prob_2 = self.other_p_2.output(p_num_2)

											ans_2[i2][j2] += prob * p_2
											res_1[i1][j1] += prob_2 * p_2

		print("start drawing round %d" % e)

		figure = plt.figure(figsize=(16, 10))
		ax1 = figure.add_subplot(2, 2, 1)
		ax2 = figure.add_subplot(2, 2, 3)
		ax3 = figure.add_subplot(2, 2, 2)
		ax4 = figure.add_subplot(2, 2, 4)

		ax1.imshow(np.log(ans_1 + 1))
		ax2.imshow(np.log(ans_2 + 1))
		ax3.imshow(np.log(res_1 + 1))
		ax4.imshow(np.log(res_2 + 1))

		figure.savefig('%s/%i.png' % (self.figure_path, e))
		plt.close(figure)


class Pushball_Appro_C_points:

	def __init__(self, size, a_size, arg, is_print):

		self.arg = arg
		self.is_print = is_print

		self.not_run = self.arg.s_alg_name == 'noisy' or self.arg.s_alg_name == 'dec' or \
		               self.arg.s_alg_name == 'cen' or self.arg.s_alg_name == 'coor_r' \
		               or self.arg.s_alg_name == 'coor_v'
		if self.not_run:
			return

		if (self.is_print):
			self.figure_path = self.arg.save_path + 'sub-goals/'
			if os.path.exists(self.figure_path):
				shutil.rmtree(self.figure_path)
			os.makedirs(self.figure_path)
		# print(self.figure_path)

		self.size = size
		self.a_size = a_size
		self.range = 3

		p_c_shape = \
			add_shape(2, self.range) + add_shape(2, size) + add_shape(1, a_size) + add_shape(2, size) + \
			add_shape(2, size) + add_shape(1, a_size)

		self.p_c_1 = Count_div_p(6, p_c_shape, 2, 1, 3, 2)
		self.p_c_2 = Count_div_p(6, p_c_shape, 2, 1, 3, 2)

	def calc_del(self, s_t, next_s_t):
		return next_s_t - s_t + 1

	def update(self, state, next_state):

		if self.not_run:
			return

		s_t, a_t = state
		s_t_1, s_t_2 = s_t
		a_t_1, a_t_2 = a_t

		s_e = s_t_1[2:]

		next_s_t = next_state[0]
		next_s_t_1, next_s_t_2 = next_s_t

		del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
		del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

		c_num_1 = [del_s_t_2, s_t_2, a_t_2, s_e, s_t_1, a_t_1]
		c_num_2 = [del_s_t_1, s_t_1, a_t_1, s_e, s_t_2, a_t_2]

		self.p_c_1.add(c_num_1)
		self.p_c_2.add(c_num_2)

	def output(self, state, next_state, agent_index):

		if self.not_run:
			return 0

		s_t, a_t = state
		s_t_1, s_t_2 = s_t
		a_t_1, a_t_2 = a_t

		s_e = s_t_1[2:]

		next_s_t = next_state[0]
		next_s_t_1, next_s_t_2 = next_s_t

		del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
		del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

		c_num_1 = [del_s_t_2, s_t_2, a_t_2, s_e, s_t_1, a_t_1]
		c_num_2 = [del_s_t_1, s_t_1, a_t_1, s_e, s_t_2, a_t_2]

		if (self.arg.symmetry):
			p_1 = 1 - self.p_c_1.output(c_num_1)
			p_2 = 1 - self.p_c_2.output(c_num_2)
			return p_1 + p_2

		if (agent_index == 0):
			p_1 = 1 - self.p_c_1.output(c_num_1)
			return p_1
		else:
			p_2 = 1 - self.p_c_2.output(c_num_2)
			return p_2

	def show(self, e):
		pass


class Pushball_Appro_C_points_2:

	def __init__(self, size, a_size, arg, is_print):

		self.arg = arg
		self.is_print = is_print

		self.not_run = self.arg.s_alg_name == 'noisy' or self.arg.s_alg_name == 'dec' or \
		               self.arg.s_alg_name == 'cen' or self.arg.s_alg_name == 'coor_r' \
		               or self.arg.s_alg_name == 'coor_v'
		if self.not_run:
			return

		if (self.is_print):
			self.figure_path = self.arg.save_path + 'sub-goals/'
			if os.path.exists(self.figure_path):
				shutil.rmtree(self.figure_path)
			os.makedirs(self.figure_path)
		# print(self.figure_path)

		self.size = size
		self.a_size = a_size
		self.pred_state = 5
		self.dis_state = 6

		p_c_shape = add_shape(1, self.pred_state) + add_shape(1, self.dis_state) + add_shape(1, a_size) + \
		            add_shape(1, self.dis_state) + add_shape(1, a_size)

		self.p_c_1 = Count_div_p(5, p_c_shape, 2, 1, 2, 1)
		self.p_c_2 = Count_div_p(5, p_c_shape, 2, 1, 2, 1)

	def calc_pred_state(self, s_t, next_s_t):
		del_s_t = abs(s_t[:2] - next_s_t[:2])
		next_s = 0
		if del_s_t[0] == 0 and del_s_t[1] == 0:
			next_s = 0
		elif del_s_t[0] == 0 and del_s_t[1] == -1:
			next_s = 1
		elif del_s_t[0] == 0 and del_s_t[1] == 1:
			next_s = 2
		elif del_s_t[0] == -1 and del_s_t[1] == 0:
			next_s = 3
		elif del_s_t[0] == 1 and del_s_t[1] == 0:
			next_s = 4
		return next_s

	def calc_dis_state(self, s_t, s_e):
		del_s_t = abs(s_t[:2] - s_e)
		next_s = 0
		if del_s_t[0] == 0 and del_s_t[1] == 0:
			next_s = 0
		elif del_s_t[0] == 0 and del_s_t[1] == -1:
			next_s = 1
		elif del_s_t[0] == 0 and del_s_t[1] == 1:
			next_s = 2
		elif del_s_t[0] == -1 and del_s_t[1] == 0:
			next_s = 3
		elif del_s_t[0] == 1 and del_s_t[1] == 0:
			next_s = 4
		else:
			next_s = 5
		return next_s

	def update(self, state, next_state):

		if self.not_run:
			return

		s_t, a_t = state
		s_t_1, s_t_2 = s_t
		a_t_1, a_t_2 = a_t

		s_e = s_t_1[2:]

		next_s_t = next_state[0]
		next_s_t_1, next_s_t_2 = next_s_t

		del_s_t_1 = self.calc_pred_state(s_t_1, next_s_t_1)
		del_s_t_2 = self.calc_pred_state(s_t_2, next_s_t_2)

		dis_s_t_1 = self.calc_dis_state(s_t_1, s_e)
		dis_s_t_2 = self.calc_dis_state(s_t_2, s_e)

		c_num_1 = [del_s_t_2, dis_s_t_2, a_t_2, dis_s_t_1, a_t_1]
		c_num_2 = [del_s_t_1, dis_s_t_1, a_t_1, dis_s_t_2, a_t_2]

		self.p_c_1.add(c_num_1)
		self.p_c_2.add(c_num_2)

	def output(self, state, next_state, agent_index):

		if self.not_run:
			return 0

		s_t, a_t = state
		s_t_1, s_t_2 = s_t
		a_t_1, a_t_2 = a_t

		s_e = s_t_1[2:]

		next_s_t = next_state[0]
		next_s_t_1, next_s_t_2 = next_s_t

		next_s_t = next_state[0]
		next_s_t_1, next_s_t_2 = next_s_t

		del_s_t_1 = self.calc_pred_state(s_t_1, next_s_t_1)
		del_s_t_2 = self.calc_pred_state(s_t_2, next_s_t_2)

		dis_s_t_1 = self.calc_dis_state(s_t_1, s_e)
		dis_s_t_2 = self.calc_dis_state(s_t_2, s_e)

		c_num_1 = [del_s_t_2, dis_s_t_2, a_t_2, dis_s_t_1, a_t_1]
		c_num_2 = [del_s_t_1, dis_s_t_1, a_t_1, dis_s_t_2, a_t_2]

		if (agent_index == 0):
			p_1 = 1 - self.p_c_1.output(c_num_1)
			return p_1
		else:
			p_2 = 1 - self.p_c_2.output(c_num_2)
			return p_2

	def show(self, e):
		pass


class Island_Appro_C_points:

	def __init__(self, size, a_size, agent_power_size, wolf_power_size, arg, is_print):

		self.arg = arg
		self.is_print = is_print

		self.not_run = self.arg.s_alg_name == 'noisy' or self.arg.s_alg_name == 'cen' or self.arg.s_alg_name == 'dec' \
		               or self.arg.s_alg_name == 'coor_v' or self.arg.s_alg_name == 'coor_r'

		if self.not_run:
			return

		if (self.is_print):
			self.figure_path = self.arg.save_path + 'sub-goals/'
			if os.path.exists(self.figure_path):
				shutil.rmtree(self.figure_path)
			os.makedirs(self.figure_path)
		# print(self.figure_path)

		self.size = size
		self.a_size = a_size
		self.agent_power_size = agent_power_size
		self.wolf_power_size = wolf_power_size
		self.range = 3

		p_c_shape = \
			add_shape(2, self.range) + add_shape(1, 3) + \
			add_shape(2, size) + add_shape(1, agent_power_size) + add_shape(1, a_size) + \
			add_shape(2, size) + add_shape(2, size)

		self.p_c_1 = Count_div_p(7, p_c_shape, 1, 2, 2, 3)
		self.p_c_2 = Count_div_p(7, p_c_shape, 1, 2, 2, 3)

	def calc_del(self, s_t, next_s_t):
		del_s_t = next_s_t - s_t
		del_s_t[:2] += 1
		del_s_t[2] += 2
		return del_s_t

	def update(self, state, next_state):
		if self.not_run:
			return

		s_t, a_t = state
		s_t_1, s_t_2 = s_t
		a_t_1, a_t_2 = a_t

		s_wolf = s_t_1[3:]

		next_s_t = next_state[0]
		next_s_t_1, next_s_t_2 = next_s_t

		del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
		del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

		c_num_1 = [del_s_t_2[:2], del_s_t_2[2], s_t_2[:2], s_t_2[2], a_t_2, s_wolf[:2], s_t_1]
		c_num_2 = [del_s_t_1[:2], del_s_t_1[2], s_t_1[:2], s_t_1[2], a_t_1, s_wolf[:2], s_t_2]

		self.p_c_1.add(c_num_1)
		self.p_c_2.add(c_num_2)

	def output(self, state, next_state, agent_index):
		if self.not_run:
			return 0

		s_t, a_t = state
		s_t_1, s_t_2 = s_t
		a_t_1, a_t_2 = a_t

		s_wolf = s_t_1[3:]

		next_s_t = next_state[0]
		next_s_t_1, next_s_t_2 = next_s_t

		del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
		del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

		c_num_1 = [del_s_t_2[:2], del_s_t_2[2], s_t_2[:2], s_t_2[2], a_t_2, s_wolf[:2], s_t_1]
		c_num_2 = [del_s_t_1[:2], del_s_t_1[2], s_t_1[:2], s_t_1[2], a_t_1, s_wolf[:2], s_t_2]

		if (self.arg.symmetry):
			p_1 = 1 - self.p_c_1.output(c_num_1)
			p_2 = 1 - self.p_c_2.output(c_num_2)
			return p_1 + p_2

		if (agent_index == 0):
			p_1 = 1 - self.p_c_1.output(c_num_1)
			return p_1
		else:
			p_2 = 1 - self.p_c_2.output(c_num_2)
			return p_2

	def show(self, e):
		pass


class Test_Island_Appro_C_points:

	def __init__(self, size, a_size, agent_power_size, wolf_power_size, arg, is_print):

		self.arg = arg
		self.is_print = is_print

		self.not_run = self.arg.s_alg_name == 'noisy' or self.arg.s_alg_name == 'cen'

		if self.not_run:
			return

		if (self.is_print):
			self.figure_path = self.arg.save_path + 'sub-goals-test/'
			if os.path.exists(self.figure_path):
				shutil.rmtree(self.figure_path)
			os.makedirs(self.figure_path)
		# print(self.figure_path)

		self.size = size
		self.a_size = a_size
		self.agent_power_size = agent_power_size
		self.wolf_power_size = wolf_power_size
		self.range = 3

		p_c_shape = \
			add_shape(2, self.range) + add_shape(1, 3) + \
			add_shape(2, size) + add_shape(1, agent_power_size) + add_shape(1, a_size) + \
			add_shape(2, size) + \
			add_shape(2, size)

		self.p_c_1 = Test_Count_div_p(7, p_c_shape, 1, 2, 2, 3)
		self.p_c_2 = Test_Count_div_p(7, p_c_shape, 1, 2, 2, 3)

	def calc_del(self, s_t, next_s_t):
		del_s_t = next_s_t - s_t
		del_s_t[:2] += 1
		del_s_t[2] += 2
		return del_s_t

	def update(self, state, next_state):
		if self.not_run:
			return

		s_t, a_t = state
		s_t_1, s_t_2 = s_t
		a_t_1, a_t_2 = a_t

		s_wolf = s_t_1[3:]

		next_s_t = next_state[0]
		next_s_t_1, next_s_t_2 = next_s_t

		del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
		del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

		c_num_1 = [del_s_t_2[:2], del_s_t_2[2], s_t_2[:2], s_t_2[2], a_t_2, s_wolf, s_t_1]
		c_num_2 = [del_s_t_1[:2], del_s_t_1[2], s_t_1[:2], s_t_1[2], a_t_1, s_wolf, s_t_2]

		self.p_c_1.add(c_num_1)
		self.p_c_2.add(c_num_2)

	def output(self, state, next_state, agent_index):
		if self.not_run:
			return 0

		s_t, a_t = state
		s_t_1, s_t_2 = s_t
		a_t_1, a_t_2 = a_t

		s_wolf = s_t_1[3:]

		next_s_t = next_state[0]
		next_s_t_1, next_s_t_2 = next_s_t

		del_s_t_1 = self.calc_del(s_t_1, next_s_t_1)
		del_s_t_2 = self.calc_del(s_t_2, next_s_t_2)

		c_num_1 = [del_s_t_2[:2], del_s_t_2[2], s_t_2[:2], s_t_2[2], a_t_2, s_wolf, s_t_1]
		c_num_2 = [del_s_t_1[:2], del_s_t_1[2], s_t_1[:2], s_t_1[2], a_t_1, s_wolf, s_t_2]

		if agent_index == 0:
			ans, ans_1, ans_2 = self.p_c_1.output(c_num_1)
			p_1 = 1 - ans
			return p_1, ans_1, ans_2
		else:
			ans, ans_1, ans_2 = self.p_c_2.output(c_num_2)
			p_2 = 1 - ans
			return p_2, ans_1, ans_2

	def show(self, e):
		pass


class VI:

	def __init__(self, goal_range, input_range, s_1, s_2, name, args, lr):
		self.args = args
		self.s_1 = s_1
		self.s_2 = s_2
		self.name = name
		self.lr = lr

		self.input_range = input_range
		self.goal_range = goal_range

		self.graph = tf.Graph()
		with self.graph.as_default():
			self.create_network()

			config = tf.ConfigProto(allow_soft_placement=True)
			config.gpu_options.allow_growth = True

			self.sess = tf.Session(config=config)
			self.sess.run(tf.global_variables_initializer())

	def create_network(self):
		self.data_input = tf.placeholder(tf.float32, shape=[None, self.input_range])
		self.labels = tf.placeholder(tf.int32, shape=[None, ])
		self.num_data = tf.placeholder(tf.float32, shape=[1])

		self.model = tf.keras.Sequential([
			tfp.layers.DenseReparameterization(64, activation=tf.nn.relu),
			tfp.layers.DenseReparameterization(64, activation=tf.nn.relu),
			tfp.layers.DenseReparameterization(self.goal_range),
		])
		self.logits = self.model(self.data_input)
		self.output = tf.nn.softmax(self.logits)

		self.neg_log_likelihood = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=self.labels, logits=self.logits)
		self.log_loss = tf.reduce_mean(self.neg_log_likelihood)
		self.kl = 1. * sum(self.model.losses) / self.num_data
		self.loss = self.log_loss + self.kl

		self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

		'''
		self.max_grad_norm = 0.5
		self.trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
		grads_and_var = self.trainer.compute_gradients(self.loss)
		grads, var = zip(*grads_and_var)

		if self.max_grad_norm is not None:
			# Clip the gradients (normalize)
			grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
		grads_and_var = list(zip(grads, var))
		# zip aggregate each gradient with parameters associated
		# For instance zip(ABCD, xyza) => Ax, By, Cz, Da
		self.grads = grads
		self.var = var
		self.train_op = self.trainer.apply_gradients(grads_and_var)
		'''

	def run(self, labels, feature):
		with self.graph.as_default():
			out, err = self.sess.run([self.output, self.neg_log_likelihood],
			                         feed_dict={self.data_input: feature,
			                                    self.labels: labels})
			res = np.squeeze(np.take_along_axis(out, np.expand_dims(labels, axis=1), axis=1))
		return res, err, out

	def update(self, labels, feature):
		num_data = labels.shape[0]
		with self.graph.as_default():
			_, err = \
				self.sess.run([self.train_op, self.neg_log_likelihood],
				              feed_dict={self.data_input: feature,
				                         self.labels: labels,
				                         self.num_data: np.array([num_data])})
		return err


# if self.name == 'p' and self.s_2 == 1:
#    print(loss)


class Island_VI_Appro_C_points:

	def __init__(self, size, a_size, agent_power_size, wolf_power_size, args, is_print):

		self.args = args
		self.is_print = is_print

		self.not_run = self.args.s_alg_name == 'noisy' or self.args.s_alg_name == 'cen' \
		               or self.args.s_alg_name == 'dec' or self.args.s_alg_name == 'coor_v' \
		               or self.args.s_alg_name == 'coor_r'

		if self.not_run:
			return

		if self.is_print:
			self.figure_path = self.args.save_path + 'sub-goals/'
			if os.path.exists(self.figure_path):
				shutil.rmtree(self.figure_path)
			os.makedirs(self.figure_path)
		# print(self.figure_path)

		self.n_agent = self.args.n_agent
		self.n_wolf = 1
		self.size = size
		self.a_size = a_size
		self.agent_power_size = agent_power_size
		self.wolf_power_size = wolf_power_size

		self.xy_range = 5
		self.harm_range = self.args.x_island_harm_range
		# self.goal_range = 55
		self.goal_range = self.xy_range * self.harm_range
		# self.input_range_p = 251
		self.input_range_p = (self.size * 2 + self.agent_power_size + self.a_size) * (self.n_agent - 1) + \
		                     self.size * 2
		# self.input_range_t = 328
		self.input_range_t = self.input_range_p + self.size * 2 + self.agent_power_size + self.a_size

		print('goal_range', self.goal_range, 'input_range_p', self.input_range_p, 'input_range_t', self.input_range_t)

		self.eye_size = np.eye(self.size)
		self.eye_action = np.eye(self.a_size)
		self.eye_agent_power = np.eye(self.agent_power_size)
		self.eye_wolf_power = np.eye(self.args.x_island_wolf_max_power)

		self.appro_T = self.args.appro_T

		self.batch_size = 2048
		self.collect = 0
		self.collect_label = []
		self.collect_p = []
		self.collect_t = []

		self.model_p = []
		self.model_t = []
		for i in range(self.n_agent):
			for j in range(self.n_agent):
				if i != j:
					self.model_p.append(VI(self.goal_range, self.input_range_p, j, i, 'p', args))
					self.model_t.append(VI(self.goal_range, self.input_range_t, j, i, 't', args))

	def calc_del(self, s_t, next_s_t):
		del_s_t = next_s_t - s_t
		next_s = 0
		if del_s_t[0] == 0 and del_s_t[1] == 0:
			next_s = 0
		elif del_s_t[0] == 0 and del_s_t[1] == -1:
			next_s = 1
		elif del_s_t[0] == 0 and del_s_t[1] == 1:
			next_s = 2
		elif del_s_t[0] == -1 and del_s_t[1] == 0:
			next_s = 3
		elif del_s_t[0] == 1 and del_s_t[1] == 0:
			next_s = 4
		del_s_t[2] += self.harm_range - 1
		next_s += del_s_t[2] * self.xy_range
		return next_s

	def make(self, state, next_state):
		s_t, a_t = state
		next_s_t = next_state[0]

		s_e = s_t[0][3:]

		v_del_i = []
		v_s_t_i = []
		v_a_t_i = []
		for i in range(self.n_agent):
			v_del_i.append(self.calc_del(s_t[i], next_s_t[i]))
			v_state = np.concatenate([self.eye_size[s_t[i][0]], self.eye_size[s_t[i][1]],
			                          self.eye_agent_power[s_t[i][2]]], axis=0)
			v_s_t_i.append(v_state)
			v_a_t_i.append(self.eye_action[a_t[i]])

		v_s_e = np.concatenate([self.eye_size[s_e[0]], self.eye_size[s_e[1]]], axis=0)

		p_num_label = []
		p_num_x_p = []
		p_num_x_t = []
		for i in range(self.n_agent):
			y = v_del_i[i]
			for j in range(self.n_agent):
				if i != j:
					x_p = []
					for k in range(self.n_agent):
						if j != k:
							x_p.append(np.concatenate([v_s_t_i[k], v_a_t_i[k]], axis=0))
					x_p = np.concatenate(x_p + [v_s_e], axis=0)
					x_t = np.concatenate([x_p, v_s_t_i[j], v_a_t_i[j]], axis=0)
					p_num_label.append(y)
					p_num_x_p.append(x_p)
					p_num_x_t.append(x_t)

		return p_num_label, p_num_x_p, p_num_x_t

	def update(self):

		labels = np.concatenate(self.collect_label, axis=1)
		feature_p = np.concatenate(self.collect_p, axis=1)
		feature_t = np.concatenate(self.collect_t, axis=1)

		t_stamp = 0
		for i in range(self.n_agent):
			for j in range(self.n_agent):
				if i != j:
					self.model_p[t_stamp].update(labels[t_stamp], feature_p[t_stamp])
					self.model_t[t_stamp].update(labels[t_stamp], feature_t[t_stamp])
					t_stamp += 1

		self.collect = 0
		self.collect_label = []
		self.collect_p = []
		self.collect_t = []

	def output(self, label, x_p, x_t):

		self.collect_label.append(label)
		self.collect_p.append(x_p)
		self.collect_t.append(x_t)

		num_data = label.shape[1]
		self.collect += num_data
		if self.collect >= self.batch_size:
			self.update()

		coor_rewards = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')
		coor_p = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')
		coor_t = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')

		t_stamp = 0
		for i in range(self.n_agent):
			for j in range(self.n_agent):
				if i != j:
					prob_p = self.model_p[t_stamp].run(label[t_stamp], x_p[t_stamp])
					prob_t = self.model_t[t_stamp].run(label[t_stamp], x_t[t_stamp])
					t_stamp += 1
					coor_p[j][i] = prob_p
					coor_t[j][i] = prob_t
					coor_rewards[j][i] = 1. - np.minimum(prob_p / np.maximum(prob_t, 1e-8), 1. / self.appro_T)

		return coor_rewards, coor_p, coor_t

	def show(self, e):
		pass


class Pass_VI_Appro_C_points:

	def __init__(self, size, a_size, args, is_print):

		self.args = args
		self.is_print = is_print

		self.not_run = self.args.s_alg_name == 'noisy' or self.args.s_alg_name == 'cen' \
		               or self.args.s_alg_name == 'dec' or self.args.s_alg_name == 'coor_v' \
		               or self.args.s_alg_name == 'coor_r'

		if self.not_run:
			return

		if self.is_print:
			self.figure_path = self.args.save_path + 'sub-goals/'
			if os.path.exists(self.figure_path):
				shutil.rmtree(self.figure_path)
			os.makedirs(self.figure_path)
		# print(self.figure_path)

		self.n_agent = self.args.n_agent
		self.size = size
		self.a_size = a_size

		self.xy_range = 5
		# self.goal_range = 5
		self.goal_range = self.xy_range
		# self.input_range_p = 64
		self.input_range_p = (self.size * 2 + self.a_size) * (self.n_agent - 1)
		# self.input_range_t = 128
		self.input_range_t = self.input_range_p + self.size * 2 + self.a_size

		print('goal_range', self.goal_range, 'input_range_p', self.input_range_p, 'input_range_t', self.input_range_t)

		self.eye_size = np.eye(self.size)
		self.eye_action = np.eye(self.a_size)

		self.appro_T = self.args.appro_T

		self.batch_size = 2048
		self.collect = 0
		self.collect_label = []
		self.collect_p = []
		self.collect_t = []

		self.model_p = []
		self.model_t = []
		for i in range(self.n_agent):
			for j in range(self.n_agent):
				if i != j:
					self.model_p.append(VI(self.goal_range, self.input_range_p, j, i, 'p', args))
					self.model_t.append(VI(self.goal_range, self.input_range_t, j, i, 't', args))

	def calc_del(self, s_t, next_s_t):
		del_s_t = next_s_t - s_t
		next_s = 0
		if del_s_t[0] == 0 and del_s_t[1] == 0:
			next_s = 0
		elif del_s_t[0] == 0 and del_s_t[1] == -1:
			next_s = 1
		elif del_s_t[0] == 0 and del_s_t[1] == 1:
			next_s = 2
		elif del_s_t[0] == -1 and del_s_t[1] == 0:
			next_s = 3
		elif del_s_t[0] == 1 and del_s_t[1] == 0:
			next_s = 4
		return next_s

	def make(self, state, next_state):
		s_t, a_t = state
		next_s_t = next_state[0]

		v_del_i = []
		v_s_t_i = []
		v_a_t_i = []
		for i in range(self.n_agent):
			v_del_i.append(self.calc_del(s_t[i], next_s_t[i]))
			v_state = np.concatenate([self.eye_size[s_t[i][0]], self.eye_size[s_t[i][1]]], axis=0)
			v_s_t_i.append(v_state)
			v_a_t_i.append(self.eye_action[a_t[i]])

		p_num_label = []
		p_num_x_p = []
		p_num_x_t = []
		for i in range(self.n_agent):
			y = v_del_i[i]
			for j in range(self.n_agent):
				if i != j:
					x_p = []
					for k in range(self.n_agent):
						if j != k:
							x_p.append(np.concatenate([v_s_t_i[k], v_a_t_i[k]], axis=0))
					x_p = np.concatenate(x_p, axis=0)
					x_t = np.concatenate([x_p, v_s_t_i[j], v_a_t_i[j]], axis=0)
					p_num_label.append(y)
					p_num_x_p.append(x_p)
					p_num_x_t.append(x_t)

		return p_num_label, p_num_x_p, p_num_x_t

	def update(self):

		labels = np.concatenate(self.collect_label, axis=1)
		feature_p = np.concatenate(self.collect_p, axis=1)
		feature_t = np.concatenate(self.collect_t, axis=1)

		t_stamp = 0
		for i in range(self.n_agent):
			for j in range(self.n_agent):
				if i != j:
					self.model_p[t_stamp].update(labels[t_stamp], feature_p[t_stamp])
					self.model_t[t_stamp].update(labels[t_stamp], feature_t[t_stamp])
					t_stamp += 1

		self.collect = 0
		self.collect_label = []
		self.collect_p = []
		self.collect_t = []

	def output(self, label, x_p, x_t):

		self.collect_label.append(label)
		self.collect_p.append(x_p)
		self.collect_t.append(x_t)

		num_data = label.shape[1]
		self.collect += num_data
		if self.collect >= self.batch_size:
			self.update()

		coor_rewards = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')
		coor_p = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')
		coor_t = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')

		t_stamp = 0
		for i in range(self.n_agent):
			for j in range(self.n_agent):
				if i != j:
					prob_p = self.model_p[t_stamp].run(label[t_stamp], x_p[t_stamp])
					prob_t = self.model_t[t_stamp].run(label[t_stamp], x_t[t_stamp])
					t_stamp += 1
					coor_p[j][i] = prob_p
					coor_t[j][i] = prob_t
					coor_rewards[j][i] = 1. - np.minimum(prob_p / np.maximum(prob_t, 1e-8), 1. / self.appro_T)

		return coor_rewards, coor_p, coor_t

	def show(self, e):
		pass


class ReplayBuffer(object):
	def __init__(self, size=50000, sub_size=1):
		"""Create Prioritized Replay buffer.

		Parameters
		----------
		size: int
			Max number of transitions to store in the buffer. When the buffer
			overflows the old memories are dropped.
		"""
		self.size = size
		self.sub_size = sub_size
		self._storage = []
		self._maxsize = int(size)
		self._next_idx = 0

	def __len__(self):
		return len(self._storage)

	def clear(self):
		self._storage = []
		self._next_idx = 0

	def add(self, data):
		data = data.astype(np.bool)
		if self._next_idx >= len(self._storage):
			self._storage.append(data)
		else:
			self._storage[self._next_idx] = data
		self._next_idx = (self._next_idx + 1) % self._maxsize

	def _encode_sample(self, idxes):
		data_list = []
		for i in idxes:
			data = self._storage[i]
			data_list.append(data.astype('float16'))
		return np.concatenate(data_list, axis=1)

	def make_index(self, batch_size):
		return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

	def make_latest_index(self, batch_size):
		idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
		np.random.shuffle(idx)
		return idx

	def sample_index(self, idxes):
		return self._encode_sample(idxes)

	def sample(self, batch_size):
		batch_size = batch_size // self.sub_size
		if batch_size > 0:
			idxes = self.make_index(batch_size)
		else:
			idxes = range(0, len(self._storage))
		return self._encode_sample(idxes)

	def collect(self):
		return self.sample(-1)


class VI_buffer:

	def __init__(self, args, batch_size=2048, t_batch_size=2048, n_update_p=20, n_update_t=20):
		self.args = args
		self.batch_size = batch_size
		self.t_batch_size = t_batch_size
		self.size_p = self.t_batch_size * n_update_p * (self.args.n_agent - 1)
		self.size_t = self.t_batch_size * n_update_t * self.args.n_agent
		self.n_agents = args.n_agent
		self.flag = np.array([0])

		self.buffer_p = []
		self.buffer_t = Experience(memory_size=self.size_t, batch_size=self.batch_size,
		                           sub_size=self.args.num_env, args=args, name='t', flag=self.flag)
		for i in range(self.n_agents):
			self.buffer_p.append(Experience(memory_size=self.size_p, batch_size=self.batch_size,
			                                sub_size=self.args.num_env, args=args, name='p', flag=self.flag))

	def add(self, label, x_p, x_t, err_p, err_t):
		t_stamp = 0
		for i in range(self.n_agents):
			for j in range(self.n_agents):
				if i != j:
					self.buffer_p[i].add((label[t_stamp], x_p[t_stamp]), err_p[t_stamp])
					if i == 0 or i == 1 and j == 0:
						self.buffer_t.add((label[t_stamp], x_t[t_stamp]), err_t[t_stamp])
					t_stamp += 1

	def sample(self, name):
		if name == 'p':
			data, indices = [], []
			for i in range(self.n_agents):
				out, t_indices = self.buffer_p[i].select()
				data.append(out)
				indices.append(t_indices)
		else:
			data, indices = self.buffer_t.select()
		return data, indices

	def priority_update(self, indices, err, name):
		if name == 'p':
			for i in range(self.n_agents):
				self.buffer_p[i].priority_update(indices[i], err[i])
		else:
			self.buffer_t.priority_update(indices, err)


class Fast_Pass_VI_Appro_C_points:

	def __init__(self, size, a_size, args, is_print):

		self.args = args
		self.is_print = is_print

		self.not_run = self.args.s_alg_name == 'noisy' or self.args.s_alg_name == 'cen' \
		               or self.args.s_alg_name == 'dec' or self.args.s_alg_name == 'coor_v' \
		               or self.args.s_alg_name == 'coor_r'

		if self.not_run:
			return

		if self.is_print:
			self.figure_path = self.args.save_path + 'sub-goals/'
			if os.path.exists(self.figure_path):
				shutil.rmtree(self.figure_path)
			os.makedirs(self.figure_path)
		# print(self.figure_path)

		self.alpha_p = self.args.VI_sample_alpha_p
		self.alpha_t = self.args.VI_sample_alpha_t

		self.n_agent = self.args.n_agent
		self.size = size
		self.a_size = a_size

		self.is_one_hot = not self.args.VI_not_one_hot

		self.xy_range = 5
		# self.goal_range = 5
		self.goal_range = self.xy_range
		if self.is_one_hot:
			# self.input_range_p = 64
			self.input_range_p = (self.size * 2 + self.a_size) * (self.n_agent - 1)
			# self.input_range_t = 128
			self.input_range_t = self.input_range_p + self.size * 2 + self.a_size
		else:
			# self.input_range_p = 6
			self.input_range_p = (2 + self.a_size) * (self.n_agent - 1)
			# self.input_range_t = 12
			self.input_range_t = self.input_range_p + 2 + self.a_size

		print('goal_range', self.goal_range, 'input_range_p', self.input_range_p, 'input_range_t', self.input_range_t)

		self.eye_size = np.eye(self.size)
		self.eye_action = np.eye(self.a_size)

		self.appro_T = self.args.appro_T

		self.t_batch_size = 2048
		self.batch_size = self.args.VI_batch_size
		self.wait_updates = 5 * self.t_batch_size
		self.prepare_updates = 50 * self.t_batch_size
		self.t_step = 0

		self.update_weight_p = 4
		self.update_weight_t = 16
		self.update_size_p = self.batch_size // self.update_weight_p
		self.update_step_p = 0
		self.update_size_t = self.batch_size // self.update_weight_t
		self.update_step_t = 0
		self.n_update = 200
		self.buffer = VI_buffer(self.args, batch_size=self.batch_size, t_batch_size=self.t_batch_size,
		                        n_update_p=self.n_update, n_update_t=self.n_update)

		self.model_p = []
		self.model_t = VI(self.goal_range, self.input_range_t, None, None, 't', args, args.VI_lr_t)
		for i in range(self.n_agent):
			self.model_p.append(VI(self.goal_range, self.input_range_p, i, None, 'p', args, args.VI_lr_p))

	def calc_del(self, s_t, next_s_t):
		del_s_t = next_s_t - s_t
		next_s = 0
		if del_s_t[0] == 0 and del_s_t[1] == 0:
			next_s = 0
		elif del_s_t[0] == 0 and del_s_t[1] == -1:
			next_s = 1
		elif del_s_t[0] == 0 and del_s_t[1] == 1:
			next_s = 2
		elif del_s_t[0] == -1 and del_s_t[1] == 0:
			next_s = 3
		elif del_s_t[0] == 1 and del_s_t[1] == 0:
			next_s = 4
		return next_s

	def make(self, state, next_state):
		s_t, a_t = state
		next_s_t = next_state[0]

		v_del_i = []
		v_s_t_i = []
		v_a_t_i = []
		for i in range(self.n_agent):
			v_del_i.append(self.calc_del(s_t[i], next_s_t[i]))
			if self.is_one_hot:
				v_state = np.concatenate([self.eye_size[s_t[i][0]], self.eye_size[s_t[i][1]]], axis=0)
			else:
				v_state = np.array(s_t[i][:2])
			v_s_t_i.append(v_state)
			v_a_t_i.append(self.eye_action[a_t[i]])

		p_num_label = []
		p_num_x_p = []
		p_num_x_t = []
		for i in range(self.n_agent):
			for j in range(self.n_agent):
				if i != j:
					y = v_del_i[j]
					x_p = [v_s_t_i[j], v_a_t_i[j]]
					for k in range(self.n_agent):
						if k != i and k != j:
							x_p.extend([v_s_t_i[k], v_a_t_i[k]])
					x_p = np.concatenate(x_p, axis=0)
					x_t = np.concatenate([x_p, v_s_t_i[i], v_a_t_i[i]], axis=0)
					p_num_label.append(y)
					p_num_x_p.append(x_p)
					p_num_x_t.append(x_t)

		return p_num_label, p_num_x_p, p_num_x_t

	def update_x(self, data, model, alpha):
		labels, features = data
		num_batch = len(labels)
		shape = []
		for item in labels:
			shape.append(item.shape[0])
		labels = np.concatenate(labels, axis=0)
		features = np.concatenate(features, axis=0)
		err = model.update(labels, features)
		res = []
		start = 0
		for i in range(num_batch):
			end = start + shape[i]
			res.append(np.mean(err[start: end] ** alpha))
			start = end
		return res

	def update_single(self, name):
		data, indices = self.buffer.sample(name)
		if name == 'p':
			err = []
			for i in range(self.n_agent):
				t_err = self.update_x(data[i], self.model_p[i], self.alpha_p)
				err.append(t_err)
		else:
			err = self.update_x(data, self.model_t, self.alpha_t)
		self.buffer.priority_update(indices, err, name)

	def update(self):

		if self.t_step <= self.wait_updates:
			return

		if self.update_step_p >= self.update_size_p:
			self.update_single('p')
			self.update_step_p = 0

		if self.update_step_t >= self.update_size_t:
			self.update_single('t')
			self.update_step_t = 0

	def output(self, label, x_p, x_t):

		self.t_step += 1
		num_data = label.shape[1]

		coor_rewards = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')
		coor_p = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')
		coor_t = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')
		coor_rew_show = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')

		err_p, err_t = [], []
		err_p_show = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')
		err_t_show = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')

		# out_p_show = np.zeros((self.n_agent, self.n_agent, num_data, self.goal_range), dtype='float32')
		# out_t_show = np.zeros((self.n_agent, self.n_agent, num_data, self.goal_range), dtype='float32')

		t_stamp = 0
		for i in range(self.n_agent):
			for j in range(self.n_agent):
				if i != j:
					prob_p, t_err_p, t_out_p = self.model_p[i].run(label[t_stamp], x_p[t_stamp])
					err_p.append(np.mean(t_err_p ** self.alpha_p))
					coor_p[i][j] = prob_p
					err_p_show[i][j] = t_err_p

					if i == 0 or i == 1 and j == 0:
						prob_t, t_err_t, t_out_t = self.model_t.run(label[t_stamp], x_t[t_stamp])
					else:
						x = i - 1
						if x == j:
							x -= 1
						prob_t, t_err_t = coor_t[x][j], err_t_show[x][j]
					err_t.append(np.mean(t_err_t ** self.alpha_t))
					coor_t[i][j] = prob_t
					err_t_show[i][j] = t_err_t

					coor_rew = 1. - np.minimum(prob_p / np.maximum(prob_t, 1e-6), 1. / self.appro_T)
					coor_rew_show[i][j] = coor_rew
					coor_rewards[i][j] = coor_rew if self.t_step >= self.prepare_updates else np.zeros(num_data)
					t_stamp += 1
		# out_p_show[j][i] = t_out_p
		# out_t_show[j][i] = t_out_t

		self.buffer.add(label, x_p, x_t, err_p, err_t)

		self.update_step_p += num_data
		self.update_step_t += num_data
		self.update()

		# return coor_rewards, coor_p, coor_t, coor_rew_show, err_p_show, err_t_show, out_p_show, out_t_show
		return coor_rewards, coor_p, coor_t, coor_rew_show, err_p_show, err_t_show

	def show(self, e):
		pass


class Fast_Island_VI_Appro_C_points(Fast_Pass_VI_Appro_C_points):

	def __init__(self, size, a_size, agent_power_size, wolf_power_size, args, is_print):

		Fast_Pass_VI_Appro_C_points.__init__(self, size, a_size, args, is_print)

		self.is_fully_pred = self.args.VI_island_fully_pred
		self.is_test = self.args.VI_island_test
		self.is_test_2 = self.args.VI_island_test_2
		self.is_test_3 = self.args.VI_island_test_3

		self.n_wolf = 1
		self.agent_power_size = agent_power_size
		self.wolf_power_size = wolf_power_size

		self.harm_range = self.args.x_island_harm_range
		# self.goal_range = 55 if self.is_fully_pred else 11
		self.goal_range = self.xy_range * self.harm_range if self.is_fully_pred else self.harm_range
		if self.is_test_3:
			# self.input_range_p = 221
			self.input_range_p = (1 + 1 + self.a_size) * (self.n_agent - 1)
			# self.input_range_t = 288
			self.input_range_t = self.input_range_p + 1 + 1 + self.a_size
		elif self.is_test_2:
			# self.input_range_p = 221
			self.input_range_p = (self.size + 1 + self.a_size) * (self.n_agent - 1)
			# self.input_range_t = 288
			self.input_range_t = self.input_range_p + self.size + 1 + self.a_size
		elif self.is_one_hot:
			if self.is_test:
				# self.input_range_p = 281
				self.input_range_p = (self.size * 3 + 1 + self.a_size) * (self.n_agent - 1) + \
				                     self.size * 2
				# self.input_range_t = 368
				self.input_range_t = self.input_range_p + self.size * 3 + 1 + self.a_size
			else:
				# self.input_range_p = 311
				self.input_range_p = (self.size * 4 + self.agent_power_size + self.a_size) * (self.n_agent - 1) + \
				                     self.size * 2
				# self.input_range_t = 408
				self.input_range_t = self.input_range_p + self.size * 4 + self.agent_power_size + self.a_size
		else:
			if self.is_test:
				self.input_range_p = (4 + self.a_size) * (self.n_agent - 1) + 2
				self.input_range_t = self.input_range_p + 4 + self.a_size
			else:
				# self.input_range_p = 35
				self.input_range_p = (3 + self.a_size) * (self.n_agent - 1) + 2
				# self.input_range_p = 46
				self.input_range_t = self.input_range_p + 3 + self.a_size

		print('goal_range', self.goal_range, 'input_range_p', self.input_range_p, 'input_range_t', self.input_range_t)

		self.eye_state = np.eye(self.size * 2)
		self.eye_agent_power = np.eye(self.agent_power_size)
		self.eye_wolf_power = np.eye(self.args.x_island_wolf_max_power)

		self.t_batch_size = 2048
		self.batch_size = self.args.VI_batch_size
		self.wait_updates = 5 * self.t_batch_size
		self.prepare_updates = 10 * self.t_batch_size
		self.t_step = 0

		self.update_weight_p = 2 if self.args.n_agent == 4 else 4
		self.update_weight_t = 8 if self.args.n_agent == 4 else 16
		self.update_size_p = self.batch_size // self.update_weight_p
		self.update_step_p = 0
		self.update_size_t = self.batch_size // self.update_weight_t
		self.update_step_t = 0
		self.n_update = 50 if self.args.n_agent == 4 else 200
		self.buffer = VI_buffer(self.args, batch_size=self.batch_size, t_batch_size=self.t_batch_size,
		                        n_update_p=self.n_update, n_update_t=self.n_update)

		self.model_p = []
		self.model_t = VI(self.goal_range, self.input_range_t, None, None, 't', args, args.VI_lr_t)
		for i in range(self.n_agent):
			self.model_p.append(VI(self.goal_range, self.input_range_p, i, None, 'p', args, args.VI_lr_p))

	def calc_del(self, s_t, next_s_t):
		del_s_t = next_s_t - s_t
		if self.is_fully_pred:
			next_s = 0
			if del_s_t[0] == 0 and del_s_t[1] == 0:
				next_s = 0
			elif del_s_t[0] == 0 and del_s_t[1] == -1:
				next_s = 1
			elif del_s_t[0] == 0 and del_s_t[1] == 1:
				next_s = 2
			elif del_s_t[0] == -1 and del_s_t[1] == 0:
				next_s = 3
			elif del_s_t[0] == 1 and del_s_t[1] == 0:
				next_s = 4
			next_s += (-del_s_t[2]) * self.xy_range
		else:
			next_s = -del_s_t[2]
		return next_s

	def calc_state(self, s_t, s_e):
		if self.is_test_3:
			del_s_t = abs(s_t[:2] - s_e)
			state = np.concatenate([np.array([max(del_s_t[0], del_s_t[1]) <= 1]),
			                        np.array([s_t[2] == 0])], axis=0)
		elif self.is_test_2:
			del_s_t = abs(s_t[:2] - s_e)
			state = np.concatenate([self.eye_size[max(del_s_t[0], del_s_t[1])],
			                        np.array([s_t[2] == 0])], axis=0)
		elif self.is_one_hot:
			if self.is_test:
				del_s_t = abs(s_t[:2] - s_e)
				state = np.concatenate([self.eye_size[max(del_s_t[0], del_s_t[1])],
				                        self.eye_size[s_t[0]], self.eye_size[s_t[1]],
				                        np.array([s_t[2] == 0])], axis=0)
			else:
				del_s_t = s_t[:2] - s_e + self.size
				state = np.concatenate([self.eye_state[del_s_t[0]], self.eye_state[del_s_t[1]],
				                        self.eye_agent_power[s_t[2]]], axis=0)
		else:
			if self.is_test:
				del_s_t = abs(s_t[:2] - s_e)
				state = np.concatenate([[max(del_s_t[0], del_s_t[1])], s_t[:3]], axis=0)
			else:
				state = np.concatenate([s_t[:2] - s_e, [s_t[3]]], axis=0)
		return state

	def make(self, state, next_state):
		s_t, a_t = state
		next_s_t = next_state[0]

		s_e = s_t[0][3:]
		if self.is_test_2 or self.is_test_3:
			v_s_e = np.array([])
		elif self.is_one_hot:
			v_s_e = np.concatenate([self.eye_size[s_e[0]], self.eye_size[s_e[1]]], axis=0)
		else:
			v_s_e = np.array(s_e[:2])

		v_del_i = []
		v_s_t_i = []
		v_a_t_i = []
		for i in range(self.n_agent):
			v_del_i.append(self.calc_del(s_t[i], next_s_t[i]))
			v_s_t_i.append(self.calc_state(s_t[i], s_e))
			v_a_t_i.append(self.eye_action[a_t[i]])

		p_num_label = []
		p_num_x_p = []
		p_num_x_t = []
		for i in range(self.n_agent):
			for j in range(self.n_agent):
				if i != j:
					y = v_del_i[j]
					x_p = [v_s_t_i[j], v_a_t_i[j], v_s_e]
					for k in range(self.n_agent):
						if k != i and k != j:
							x_p.extend([v_s_t_i[k], v_a_t_i[k]])
					x_p = np.concatenate(x_p, axis=0)
					x_t = np.concatenate([x_p, v_s_t_i[i], v_a_t_i[i]], axis=0)
					p_num_label.append(y)
					p_num_x_p.append(x_p)
					p_num_x_t.append(x_t)

		return p_num_label, p_num_x_p, p_num_x_t


class x_Island_Appro_C_points:

	def __init__(self, size, a_size, agent_max_power, wolf_power_size, args, is_print):

		self.args = args
		self.is_print = is_print

		self.not_run = self.args.s_alg_name == 'noisy' or self.args.s_alg_name == 'cen' \
		               or self.args.s_alg_name == 'dec' or self.args.s_alg_name == 'coor_v' \
		               or self.args.s_alg_name == 'coor_r'

		if self.not_run:
			return

		if (self.is_print):
			self.figure_path = self.args.save_path + 'sub-goals/'
			if os.path.exists(self.figure_path):
				shutil.rmtree(self.figure_path)
			os.makedirs(self.figure_path)
		# print(self.figure_path)

		self.n_agents = self.args.n_agent
		self.size = size
		self.a_size = a_size
		self.harm_range = self.args.x_island_harm_range
		self.pred = self.harm_range
		self.pred_state = 2 * self.harm_range
		self.other_state = 2 * 2

		p_c_shape = add_shape(1, self.pred) + add_shape(1, self.pred_state)
		for i in range(self.n_agents - 1):
			p_c_shape.extend(add_shape(1, self.other_state))
		self.p_c_len = self.n_agents + 1

		self.c_t = Count_p(self.p_c_len, p_c_shape, 1, 1)
		self.c_p = []
		for i in range(self.n_agents):
			self.c_p.append(Count_p(self.p_c_len - 1, p_c_shape[:-1], 1, 1))

	def calc_del(self, s_t, next_s_t):
		del_s_t = next_s_t - s_t
		next_s = -del_s_t[2]
		return next_s

	def calc_pred_state(self, s_t, s_e):
		del_s_t = abs(s_t[:2] - s_e)
		is_kill = int(max(del_s_t[0], del_s_t[1]) <= 1)
		blood = min(self.harm_range - 1, s_t[2])
		state = is_kill * self.harm_range + blood
		return state

	def calc_state(self, s_t, s_e):
		del_s_t = abs(s_t[:2] - s_e)
		is_kill = int(max(del_s_t[0], del_s_t[1]) <= 1)
		is_alive = int(s_t[2] > 0)
		state = is_kill * 2 + is_alive
		return state

	def update(self, state, next_state):
		if self.not_run:
			return

		s_t, a_t = state
		next_s_t = next_state[0]

		s_e = s_t[0][3:]

		for j in range(self.n_agents):
			pred = self.calc_del(s_t[j], next_s_t[j])
			state_t = [pred, self.calc_pred_state(s_t[j], s_e)]
			for k in range(self.n_agents):
				if j != k:
					state_t.append(self.calc_state(s_t[k], s_e))
			self.c_t.add(state_t)

		for i in range(self.n_agents):
			for j in range(self.n_agents):
				if i != j:
					pred = self.calc_del(s_t[j], next_s_t[j])
					state_p = [pred, self.calc_pred_state(s_t[j], s_e)]
					for k in range(self.n_agents):
						if k != i and k != j:
							state_p.append(self.calc_state(s_t[k], s_e))
					self.c_p[i].add(state_p)

	def output(self, state, next_state):
		if self.not_run:
			return

		s_t, a_t = state
		next_s_t = next_state[0]

		s_e = s_t[0][3:]

		c_t = np.zeros((self.n_agents), dtype='float32')
		for j in range(self.n_agents):
			pred = self.calc_del(s_t[j], next_s_t[j])
			state_t = [pred, self.calc_pred_state(s_t[j], s_e)]
			for k in range(self.n_agents):
				if j != k:
					state_t.append(self.calc_state(s_t[k], s_e))
			c_t[j] = self.c_t.output(state_t)

		c_p = np.zeros((self.n_agents, self.n_agents), dtype='float32')
		rew = np.zeros((self.n_agents, self.n_agents), dtype='float32')
		for i in range(self.n_agents):
			for j in range(self.n_agents):
				if i != j:
					pred = self.calc_del(s_t[j], next_s_t[j])
					state_p = [pred, self.calc_pred_state(s_t[j], s_e)]
					for k in range(self.n_agents):
						if k != i and k != j:
							state_p.append(self.calc_state(s_t[k], s_e))
					c_p[i, j] = self.c_p[i].output(state_p)
					rew[i, j] = 1 - 1. * c_p[i, j] / c_t[j]

		return rew, c_t, c_p

	def show(self, e):
		pass


class Visual_coor_rew:

	def __init__(self, size, n_agent, args, num_envs, name, is_print=True):
		self.env_n_dim = args.env_n_dim
		self.size = size
		self.n_agent = n_agent
		self.args = args
		self.num_envs = num_envs

		self.is_print = is_print

		self.visited = [np.zeros([2, args.size, args.size]) for _ in range(self.n_agent)]
		self.visited_old = [np.zeros([2, args.size, args.size]) for _ in range(self.n_agent)]
		self.value = [np.zeros([2, args.size, args.size]) for _ in range(self.n_agent)]
		self.value_old = [np.zeros([2, args.size, args.size]) for _ in range(self.n_agent)]

		self.figure_path = self.args.save_path + 'sub-goals-%s/' % name
		if os.path.exists(self.figure_path):
			shutil.rmtree(self.figure_path)
		os.makedirs(self.figure_path)

		self.e_step = 0

	def update(self, infos, dones, coor_rewards):

		if self.args.s_data_gather or not self.is_print:
			return

		for i in range(self.num_envs):
			pre_state = infos[i]['pre_state']
			if (pre_state != None):
				for j in range(self.n_agent):
					self.value[j][0][pre_state[j][0]][pre_state[j][1]] += coor_rewards[j][i]
					self.visited[j][0][pre_state[j][0]][pre_state[j][1]] += 1
					self.value[1 - j][1][pre_state[1 - j][0]][pre_state[1 - j][1]] += coor_rewards[j][i]
					self.visited[1 - j][1][pre_state[1 - j][0]][pre_state[1 - j][1]] += 1
			if dones[i]:
				self.e_step += 1
				if self.e_step % (10000 * self.args.t_save_rate) == 0:
					self.show(self.e_step)

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

	def show(self, e):

		if self.args.s_data_gather or not self.is_print:
			return

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

		figure.savefig('%s/%i.png' % (self.figure_path, e))
		plt.close(figure)

		self.visited_old = [v.copy() for v in self.visited]
		self.value_old = [v.copy() for v in self.value]


class VI_buffer_old:

	def __init__(self, args, batch_size=2048, t_batch_size=2048, size=100000):
		self.args = args
		self.batch_size = batch_size
		self.t_batch_size = t_batch_size
		self.size = self.t_batch_size * 20
		self.n_agents = args.n_agent
		self.flag = np.array([0])

		self.buffer_p = []
		self.buffer_t = []
		for i in range(self.n_agents):
			for j in range(self.n_agents):
				if i != j:
					self.buffer_p.append(Experience(memory_size=self.size, batch_size=self.batch_size,
					                                sub_size=self.args.num_env, args=args, name='p', flag=self.flag))
					self.buffer_t.append(Experience(memory_size=self.size, batch_size=self.batch_size,
					                                sub_size=self.args.num_env, args=args, name='t', flag=self.flag))

	def add(self, label, x_p, x_t, err_p, err_t):
		t_stamp = 0
		for i in range(self.n_agents):
			for j in range(self.n_agents):
				if i != j:
					self.buffer_p[t_stamp].add((label[t_stamp], x_p[t_stamp]), err_p[t_stamp])
					self.buffer_t[t_stamp].add((label[t_stamp], x_t[t_stamp]), err_t[t_stamp])
					t_stamp += 1

	def sample(self, name):
		data, indices = [], []
		t_stamp = 0
		for i in range(self.n_agents):
			for j in range(self.n_agents):
				if i != j:
					if name == 'p':
						out, t_indices = self.buffer_p[t_stamp].select()
					else:
						out, t_indices = self.buffer_t[t_stamp].select()
					data.append(out)
					indices.append(t_indices)
					t_stamp += 1
		return data, indices

	def priority_update(self, indices, err, name):
		t_stamp = 0
		for i in range(self.n_agents):
			for j in range(self.n_agents):
				if i != j:
					if name == 'p':
						self.buffer_p[t_stamp].priority_update(indices[t_stamp], err[t_stamp])
					else:
						self.buffer_t[t_stamp].priority_update(indices[t_stamp], err[t_stamp])
					t_stamp += 1


class Fast_Pass_VI_Appro_C_points_old:

	def __init__(self, size, a_size, args, is_print):

		self.args = args
		self.is_print = is_print

		self.not_run = self.args.s_alg_name == 'noisy' or self.args.s_alg_name == 'cen' \
		               or self.args.s_alg_name == 'dec' or self.args.s_alg_name == 'coor_v' \
		               or self.args.s_alg_name == 'coor_r'

		if self.not_run:
			return

		if self.is_print:
			self.figure_path = self.args.save_path + 'sub-goals/'
			if os.path.exists(self.figure_path):
				shutil.rmtree(self.figure_path)
			os.makedirs(self.figure_path)
		# print(self.figure_path)

		self.alpha_p = self.args.VI_sample_alpha_p
		self.alpha_t = self.args.VI_sample_alpha_t

		self.n_agent = self.args.n_agent
		self.size = size
		self.a_size = a_size

		self.is_one_hot = not self.args.VI_not_one_hot

		self.xy_range = 5
		# self.goal_range = 5
		self.goal_range = self.xy_range
		if self.is_one_hot:
			# self.input_range_p = 64
			self.input_range_p = (self.size * 2 + self.a_size) * (self.n_agent - 1)
			# self.input_range_t = 128
			self.input_range_t = self.input_range_p + self.size * 2 + self.a_size
		else:
			# self.input_range_p = 6
			self.input_range_p = (2 + self.a_size) * (self.n_agent - 1)
			# self.input_range_t = 12
			self.input_range_t = self.input_range_p + 2 + self.a_size

		print('goal_range', self.goal_range, 'input_range_p', self.input_range_p, 'input_range_t', self.input_range_t)

		self.eye_size = np.eye(self.size)
		self.eye_action = np.eye(self.a_size)

		self.appro_T = self.args.appro_T

		self.t_batch_size = 2048
		self.batch_size = 512
		self.wait_updates = 5 * self.t_batch_size
		self.prepare_updates = 10 * self.t_batch_size
		self.t_step = 0

		self.update_size_p = 512
		self.update_step_p = 0
		self.update_size_t = 128
		self.update_step_t = 0
		self.buffer = VI_buffer(self.args, batch_size=self.batch_size, t_batch_size=self.t_batch_size)

		self.model_p = []
		self.model_t = []
		for i in range(self.n_agent):
			for j in range(self.n_agent):
				if i != j:
					self.model_p.append(VI(self.goal_range, self.input_range_p, j, i, 'p', args, args.VI_lr_p))
					self.model_t.append(VI(self.goal_range, self.input_range_t, j, i, 't', args, args.VI_lr_t))

	def calc_del(self, s_t, next_s_t):
		del_s_t = next_s_t - s_t
		next_s = 0
		if del_s_t[0] == 0 and del_s_t[1] == 0:
			next_s = 0
		elif del_s_t[0] == 0 and del_s_t[1] == -1:
			next_s = 1
		elif del_s_t[0] == 0 and del_s_t[1] == 1:
			next_s = 2
		elif del_s_t[0] == -1 and del_s_t[1] == 0:
			next_s = 3
		elif del_s_t[0] == 1 and del_s_t[1] == 0:
			next_s = 4
		return next_s

	def make(self, state, next_state):
		s_t, a_t = state
		next_s_t = next_state[0]

		v_del_i = []
		v_s_t_i = []
		v_a_t_i = []
		for i in range(self.n_agent):
			v_del_i.append(self.calc_del(s_t[i], next_s_t[i]))
			if self.is_one_hot:
				v_state = np.concatenate([self.eye_size[s_t[i][0]], self.eye_size[s_t[i][1]]], axis=0)
			else:
				v_state = np.array(s_t[i][:2])
			v_s_t_i.append(v_state)
			v_a_t_i.append(self.eye_action[a_t[i]])

		p_num_label = []
		p_num_x_p = []
		p_num_x_t = []
		for i in range(self.n_agent):
			y = v_del_i[i]
			for j in range(self.n_agent):
				if i != j:
					x_p = []
					for k in range(self.n_agent):
						if j != k:
							x_p.append(np.concatenate([v_s_t_i[k], v_a_t_i[k]], axis=0))
					x_p = np.concatenate(x_p, axis=0)
					x_t = np.concatenate([x_p, v_s_t_i[j], v_a_t_i[j]], axis=0)
					p_num_label.append(y)
					p_num_x_p.append(x_p)
					p_num_x_t.append(x_t)

		return p_num_label, p_num_x_p, p_num_x_t

	def update_x(self, data, model, alpha):
		labels, features = data
		num_batch = len(labels)
		shape = []
		for item in labels:
			shape.append(item.shape[0])
		labels = np.concatenate(labels, axis=0)
		features = np.concatenate(features, axis=0)
		err = model.update(labels, features)
		res = []
		start = 0
		for i in range(num_batch):
			end = start + shape[i]
			res.append(np.mean(err[start: end] ** alpha))
			start = end
		return res

	def update_single(self, name):
		data, indices = self.buffer.sample(name)
		err = []
		t_stamp = 0
		for i in range(self.n_agent):
			for j in range(self.n_agent):
				if i != j:
					if name == 'p':
						t_err = self.update_x(data[t_stamp], self.model_p[t_stamp], self.alpha_p)
					else:
						t_err = self.update_x(data[t_stamp], self.model_t[t_stamp], self.alpha_t)
					err.append(t_err)
					t_stamp += 1
		self.buffer.priority_update(indices, err, name)

	def update(self):

		if self.t_step <= self.wait_updates:
			return

		if self.update_step_p >= self.update_size_p:
			self.update_single('p')
			self.update_step_p = 0

		if self.update_step_t >= self.update_size_t:
			self.update_single('t')
			self.update_step_t = 0

	def output(self, label, x_p, x_t):

		self.t_step += 1
		num_data = label.shape[1]

		coor_rewards = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')
		coor_p = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')
		coor_t = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')
		coor_rew_show = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')

		err_p, err_t = [], []
		err_p_show = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')
		err_t_show = np.zeros((self.n_agent, self.n_agent, num_data), dtype='float32')

		# out_p_show = np.zeros((self.n_agent, self.n_agent, num_data, self.goal_range), dtype='float32')
		# out_t_show = np.zeros((self.n_agent, self.n_agent, num_data, self.goal_range), dtype='float32')

		t_stamp = 0
		for i in range(self.n_agent):
			for j in range(self.n_agent):
				if i != j:
					prob_p, t_err_p, t_out_p = self.model_p[t_stamp].run(label[t_stamp], x_p[t_stamp])
					prob_t, t_err_t, t_out_t = self.model_t[t_stamp].run(label[t_stamp], x_t[t_stamp])
					err_p.append(np.mean(t_err_p ** self.alpha_p))
					err_t.append(np.mean(t_err_t ** self.alpha_t))
					coor_rew = 1. - np.minimum(prob_p / np.maximum(prob_t, 1e-6), 1. / self.appro_T)
					coor_rew_show[j][i] = coor_rew
					coor_p[j][i] = prob_p
					coor_t[j][i] = prob_t
					coor_rewards[j][i] = coor_rew if self.t_step >= self.prepare_updates else np.zeros(num_data)
					err_p_show[j][i] = t_err_p
					err_t_show[j][i] = t_err_t
					t_stamp += 1
		# out_p_show[j][i] = t_out_p
		# out_t_show[j][i] = t_out_t

		self.buffer.add(label, x_p, x_t, err_p, err_t)

		self.update_step_p += num_data
		self.update_step_t += num_data
		self.update()

		# return coor_rewards, coor_p, coor_t, coor_rew_show, err_p_show, err_t_show, out_p_show, out_t_show
		return coor_rewards, coor_p, coor_t, coor_rew_show, err_p_show, err_t_show

	def show(self, e):
		pass


class Fast_Island_VI_Appro_C_points_old(Fast_Pass_VI_Appro_C_points):

	def __init__(self, size, a_size, agent_power_size, wolf_power_size, args, is_print):

		Fast_Pass_VI_Appro_C_points.__init__(self, size, a_size, args, is_print)

		self.n_wolf = 1
		self.agent_power_size = agent_power_size
		self.wolf_power_size = wolf_power_size

		self.xy_range = 5
		self.harm_range = self.args.x_island_harm_range
		# self.goal_range = 55
		self.goal_range = self.xy_range * self.harm_range
		if self.is_one_hot:
			# self.input_range_p = 251
			self.input_range_p = (self.size * 2 + self.agent_power_size + self.a_size) * (self.n_agent - 1) + \
			                     self.size * 2
			# self.input_range_t = 328
			self.input_range_t = self.input_range_p + self.size * 2 + self.agent_power_size + self.a_size
		else:
			# self.input_range_p = 29
			self.input_range_p = (3 + self.a_size) * (self.n_agent - 1) + 2
			# self.input_range_p = 38
			self.input_range_t = self.input_range_p + 3 + self.a_size

		print('goal_range', self.goal_range, 'input_range_p', self.input_range_p, 'input_range_t', self.input_range_t)

		self.eye_agent_power = np.eye(self.agent_power_size)
		self.eye_wolf_power = np.eye(self.args.x_island_wolf_max_power)

		self.t_batch_size = 2048
		self.batch_size = 512
		self.wait_updates = 5 * self.t_batch_size
		self.prepare_updates = 10 * self.t_batch_size
		self.t_step = 0

		self.update_size_p = 512
		self.update_step_p = 0
		self.update_size_t = 256
		self.update_step_t = 0
		self.buffer = VI_buffer(self.args, batch_size=self.batch_size, t_batch_size=self.t_batch_size)

		self.model_p = []
		self.model_t = []
		for i in range(self.n_agent):
			for j in range(self.n_agent):
				if i != j:
					self.model_p.append(VI(self.goal_range, self.input_range_p, j, i, 'p', args, args.VI_lr_p))
					self.model_t.append(VI(self.goal_range, self.input_range_t, j, i, 't', args, args.VI_lr_t))

	def calc_del(self, s_t, next_s_t):
		del_s_t = next_s_t - s_t
		next_s = 0
		if del_s_t[0] == 0 and del_s_t[1] == 0:
			next_s = 0
		elif del_s_t[0] == 0 and del_s_t[1] == -1:
			next_s = 1
		elif del_s_t[0] == 0 and del_s_t[1] == 1:
			next_s = 2
		elif del_s_t[0] == -1 and del_s_t[1] == 0:
			next_s = 3
		elif del_s_t[0] == 1 and del_s_t[1] == 0:
			next_s = 4
		del_s_t[2] *= -1
		next_s += del_s_t[2] * self.xy_range
		return next_s

	def make(self, state, next_state):
		s_t, a_t = state
		next_s_t = next_state[0]

		s_e = s_t[0][3:]

		v_del_i = []
		v_s_t_i = []
		v_a_t_i = []
		for i in range(self.n_agent):
			v_del_i.append(self.calc_del(s_t[i], next_s_t[i]))
			if self.is_one_hot:
				v_state = np.concatenate([self.eye_size[s_t[i][0]], self.eye_size[s_t[i][1]],
				                          self.eye_agent_power[s_t[i][2]]], axis=0)
			else:
				v_state = np.array(s_t[i][:3])
			v_s_t_i.append(v_state)
			v_a_t_i.append(self.eye_action[a_t[i]])

		if self.is_one_hot:
			v_s_e = np.concatenate([self.eye_size[s_e[0]], self.eye_size[s_e[1]]], axis=0)
		else:
			v_s_e = np.array(s_e[:2])

		p_num_label = []
		p_num_x_p = []
		p_num_x_t = []
		for i in range(self.n_agent):
			y = v_del_i[i]
			for j in range(self.n_agent):
				if i != j:
					x_p = []
					for k in range(self.n_agent):
						if j != k:
							x_p.append(np.concatenate([v_s_t_i[k], v_a_t_i[k]], axis=0))
					x_p = np.concatenate(x_p + [v_s_e], axis=0)
					x_t = np.concatenate([x_p, v_s_t_i[j], v_a_t_i[j]], axis=0)
					p_num_label.append(y)
					p_num_x_p.append(x_p)
					p_num_x_t.append(x_t)

		return p_num_label, p_num_x_p, p_num_x_t
