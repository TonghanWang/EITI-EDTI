import numpy as np
import random
import baselines.sum_tree as sum_tree


class Experience(object):
	""" The class represents prioritized experience replay buffer.

	The class has functions: store samples, pick samples with
	probability in proportion to sample's priority, update
	each sample's priority, reset alpha.

	see https://arxiv.org/pdf/1511.05952.pdf .

	"""

	def __init__(self, memory_size, batch_size, args, name, flag, sub_size=1, alpha=1):
		self.tree = sum_tree.SumTree(memory_size, name, args)
		self.memory_size = memory_size
		self.batch_size = batch_size // sub_size
		self.sub_size = sub_size
		self.alpha = alpha
		self.args = args
		self.name = name
		self.flag = flag

	def add(self, data, priority):
		labels, features = data
		features = features.astype(np.int8 if self.args.VI_not_one_hot else np.bool)
		self.tree.add((labels, features), priority ** self.alpha)

	def select(self):

		if self.tree.filled_size() < self.batch_size:
			return None, None, None

		labels, features = [], []
		indices = []
		priorities = []
		for _ in range(self.batch_size):
			r = random.random()
			data, priority, index = self.tree.find(r)
			if data == None:
				print(index, self.tree.cursor, r)
			t_labels, t_feature = data
			labels.append(t_labels)
			features.append(t_feature.astype(np.float16))
			priorities.append(priority)
			indices.append(index)
			self.priority_update([index], [0], infos={'index': _, 'name': self.flag})  # To avoid duplicating

		self.priority_update(indices, priorities)  # Revert priorities

		return (labels, features), indices

	def priority_update(self, indices, priorities, infos=None):
		""" The methods update samples's priority.

		Parameters
		----------
		indices :
			list of sample indices
		"""
		for i, p in zip(indices, priorities):
			self.tree.val_update(i, p ** self.alpha, infos=infos)

	def reset_alpha(self, alpha):
		""" Reset a exponent alpha.

		Parameters
		----------
		alpha : float
		"""
		self.alpha, old_alpha = alpha, self.alpha
		priorities = [self.tree.get_val(i) ** -old_alpha for i in range(self.tree.filled_size())]
		self.priority_update(range(self.tree.filled_size()), priorities)
