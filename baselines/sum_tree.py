#! -*- coding:utf-8 -*-

import sys
import os
import math
import random


class SumTree(object):
	def __init__(self, max_size, name, args):
		self.max_size = max_size
		self.tree_level = math.ceil(math.log(max_size + 1, 2)) + 1
		self.tree_size = 2 ** self.tree_level - 1
		self.tree = [0 for i in range(self.tree_size)]
		self.data = [None for i in range(self.max_size)]
		self.name = name
		self.args = args
		self.size = 0
		self.cursor = 0
		self.alpha = self.args.VI_sample_alpha_p if self.name == 'p' else self.args.VI_sample_alpha_t
		self.prepare()

	def p_calc(self, item, weight):
		item *= weight
		item = item ** (1. / self.alpha)
		item = math.exp(-item)
		return round(item, 5)

	def prepare(self):
		if self.args.env == 'x_pass':
			self.anchor = [0.01, 0.1, 1.]
		elif self.args.env == 'x_island':
			self.anchor = [0.05, 0.25, 0.5]
		self.anchor_value_worst = [self.p_calc(item, self.args.num_env) for item in self.anchor]
		self.anchor_value_mean = [self.p_calc(item, 1) for item in self.anchor]
		self.counts = [0 for _ in self.anchor]
		self.counts_mean = [0. for _ in self.anchor]

	def add(self, contents, value):
		index = self.cursor
		self.cursor = (self.cursor + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

		self.data[index] = contents
		self.val_update(index, value)

	def get_val(self, index):
		tree_index = 2 ** (self.tree_level - 1) - 1 + index
		return self.tree[tree_index]

	def val_update(self, index, value, infos=None):
		tree_index = 2 ** (self.tree_level - 1) - 1 + index
		if infos != None and ((infos['name'][0] == 0) ^ (self.name == 't')) and infos['index'] == 0 and \
				random.randint(0, self.args.VI_print_steps) == 0:
			print(self.name, round(self.tree[0], 5))
			print('mean:', round(self.tree[0] / self.size, 5), self.size)
			for i, item in enumerate(self.anchor):
				print('counts_mean of worst < %.5lf, mean < %.5lf, anchor: %.3lf' %
				      (self.anchor_value_worst[i], self.anchor_value_mean[i], item),
				      round(self.counts_mean[i] / max(self.counts[i], 1), 5), self.counts[i])
			infos['name'][0] ^= 1
		# one of sub_minibatch < 0.7 -> (-math.log(0.7)) / 32 > 0.01
		# one of sub_minibatch < 0.001 -> (-math.log(0.001)) / 32 > 0.2
		for i, item in enumerate(self.anchor):
			if self.tree[tree_index] >= item:
				self.counts[i] -= 1
				self.counts_mean[i] -= self.tree[tree_index]
			if value >= item:
				self.counts[i] += 1
				self.counts_mean[i] += value
		diff = value - self.tree[tree_index]
		self.reconstruct(tree_index, diff)

	def reconstruct(self, tindex, diff):
		self.tree[tindex] += diff
		if not tindex == 0:
			tindex = int((tindex - 1) / 2)
			self.reconstruct(tindex, diff)

	def find(self, value, norm=True):
		pre_value = value
		if norm:
			value *= self.tree[0]
		list = []
		return self._find(value, 0, pre_value, list)

	def _find(self, value, index, r, list):
		if 2 ** (self.tree_level - 1) - 1 <= index:
			if index - (2 ** (self.tree_level - 1) - 1) >= self.size:
				index = (2 ** (self.tree_level - 1) - 1) + random.randint(0, self.size)
				# print('!!!!!')
				# print(index, value, self.tree[0], r)
				# print(list)
				# index = (2 ** (self.tree_level - 1) - 1)
			return self.data[index - (2 ** (self.tree_level - 1) - 1)], self.tree[index], index - (
					2 ** (self.tree_level - 1) - 1)

		left = self.tree[2 * index + 1]
		list.append(left)

		if value <= left:
			return self._find(value, 2 * index + 1, r, list)
		else:
			return self._find(value - left, 2 * (index + 1), r, list)

	def print_tree(self):
		for k in range(1, self.tree_level + 1):
			for j in range(2 ** (k - 1) - 1, 2 ** k - 1):
				print(self.tree[j], end=' ')
			print()

	def filled_size(self):
		return self.size


if __name__ == '__main__':
	s = SumTree(10)
	for i in range(20):
		s.add(2 ** i, i)
	s.print_tree()
	print(s.find(0.5))
