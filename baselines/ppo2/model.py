import tensorflow as tf
import functools

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize

try:
	from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
	from mpi4py import MPI
	from baselines.common.mpi_util import sync_from_root
except ImportError:
	MPI = None


class Model(object):
	"""
	We use this object to :
	__init__:
	- Creates the step_model
	- Creates the train_model

	train():
	- Make the training part (feedforward and retropropagation of gradients)

	save/load():
	- Save load the model
	"""

	def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
	             nsteps, ent_coef, vf_coef, max_grad_norm, agent_index, microbatch_size=None):
		self.sess = sess = get_session()

		with tf.variable_scope('ppo2_model_%i_act_and_train' % agent_index, reuse=tf.AUTO_REUSE):
			# CREATE OUR TWO MODELS
			# act_model that is used for sampling
			act_model = policy(nbatch_act, 1, sess)

			# Train model for training
			if microbatch_size is None:
				train_model = policy(nbatch_train, nsteps, sess)
			else:
				train_model = policy(microbatch_size, nsteps, sess)

		with tf.variable_scope('ppo2_model_%i_e' % agent_index, reuse=tf.AUTO_REUSE):
			e_act_model = policy(nbatch_act, 1, sess)

			# Model for 'e'xtrinsic and 'c'uriosity rewards
			if microbatch_size is None:
				e_train_model = policy(nbatch_train, nsteps, sess)
			else:
				e_train_model = policy(microbatch_size, nsteps, sess)

		with tf.variable_scope('ppo2_model_%i_c' % agent_index, reuse=tf.AUTO_REUSE):
			c_act_model = policy(nbatch_act, 1, sess)

			# Model for 'e'xtrinsic and 'c'uriosity rewards
			if microbatch_size is None:
				c_train_model = policy(nbatch_train, nsteps, sess)
			else:
				c_train_model = policy(microbatch_size, nsteps, sess)

		# CREATE THE PLACEHOLDERS
		self.A = A = train_model.pdtype.sample_placeholder([None])
		self.ADV = ADV = tf.placeholder(tf.float32, [None])
		self.R = R = tf.placeholder(tf.float32, [None])
		# Keep track of old actor
		self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
		# Keep track of old critic
		self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
		self.LR = LR = tf.placeholder(tf.float32, [])
		# Cliprange
		self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

		neglogpac = train_model.pd.neglogp(A)

		# Calculate the entropy
		# Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
		entropy = tf.reduce_mean(train_model.pd.entropy())

		# CALCULATE THE LOSS
		# Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

		# Clip the value to reduce variability during Critic training
		# Get the predicted value
		vpred = train_model.vf
		vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
		# Unclipped value
		vf_losses1 = tf.square(vpred - R)
		# Clipped value
		vf_losses2 = tf.square(vpredclipped - R)

		vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

		# Calculate ratio (pi current policy / pi old policy)
		ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

		# Defining Loss = - J is equivalent to max J
		pg_losses = -ADV * ratio

		pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

		# Final PG loss
		pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
		approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
		clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

		# Total loss
		loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

		# UPDATE THE PARAMETERS USING LOSS
		# 1. Get the model parameters
		params = tf.trainable_variables('ppo2_model_%i_act_and_train' % agent_index)
		# 2. Build our trainer
		if MPI is not None:
			self.trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
		else:
			self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
		# 3. Calculate the gradients
		grads_and_var = self.trainer.compute_gradients(loss, params)
		grads, var = zip(*grads_and_var)

		if max_grad_norm is not None:
			# Clip the gradients (normalize)
			grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
		grads_and_var = list(zip(grads, var))
		# zip aggregate each gradient with parameters associated
		# For instance zip(ABCD, xyza) => Ax, By, Cz, Da

		self.grads = grads
		self.var = var
		self._train_op = self.trainer.apply_gradients(grads_and_var)
		self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
		self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]

		self.train_model = train_model
		self.act_model = act_model
		self.step = act_model.step
		self.value = act_model.value
		self.initial_state = act_model.initial_state

		# self.save = functools.partial(save_variables, sess=sess)
		# self.load = functools.partial(load_variables, sess=sess)

		# END OF TRAIN MODEL

		# BEGIN OF E_MODEL

		self.e_A = e_A = e_train_model.pdtype.sample_placeholder([None])
		self.e_ADV = e_ADV = tf.placeholder(tf.float32, [None])
		self.e_R = e_R = tf.placeholder(tf.float32, [None])
		# Keep track of old actor
		self.e_OLDNEGLOGPAC = e_OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
		# Keep track of old critic
		self.e_OLDVPRED = e_OLDVPRED = tf.placeholder(tf.float32, [None])
		self.e_LR = e_LR = tf.placeholder(tf.float32, [])
		# Cliprange
		self.e_CLIPRANGE = e_CLIPRANGE = tf.placeholder(tf.float32, [])

		e_neglogpac = e_train_model.pd.neglogp(e_A)

		# Calculate the entropy
		# Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
		e_entropy = tf.reduce_mean(e_train_model.pd.entropy())

		# CALCULATE THE LOSS
		# Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

		# Clip the value to reduce variability during Critic training
		# Get the predicted value
		e_vpred = e_train_model.vf
		e_vpredclipped = e_OLDVPRED + tf.clip_by_value(e_train_model.vf - e_OLDVPRED, - e_CLIPRANGE, e_CLIPRANGE)
		# Unclipped value
		e_vf_losses1 = tf.square(e_vpred - e_R)
		# Clipped value
		e_vf_losses2 = tf.square(e_vpredclipped - e_R)

		e_vf_loss = .5 * tf.reduce_mean(tf.maximum(e_vf_losses1, e_vf_losses2))

		# Calculate ratio (pi current policy / pi old policy)
		e_ratio = tf.exp(e_OLDNEGLOGPAC - e_neglogpac)

		# Defining Loss = - J is equivalent to max J
		e_pg_losses = -e_ADV * e_ratio

		e_pg_losses2 = -e_ADV * tf.clip_by_value(e_ratio, 1.0 - e_CLIPRANGE, 1.0 + e_CLIPRANGE)

		# Final PG loss
		e_pg_loss = tf.reduce_mean(tf.maximum(e_pg_losses, e_pg_losses2))
		e_approxkl = .5 * tf.reduce_mean(tf.square(e_neglogpac - e_OLDNEGLOGPAC))
		e_clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(e_ratio - 1.0), e_CLIPRANGE)))

		# Total loss
		e_loss = e_vf_loss * vf_coef

		# UPDATE THE PARAMETERS USING LOSS
		# 1. Get the model parameters
		e_params = tf.trainable_variables('ppo2_model_%i_e' % agent_index)
		# 2. Build our trainer
		if MPI is not None:
			self.e_trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=e_LR, epsilon=1e-5)
		else:
			self.e_trainer = tf.train.AdamOptimizer(learning_rate=e_LR, epsilon=1e-5)
		# 3. Calculate the gradients
		e_grads_and_var = self.e_trainer.compute_gradients(e_loss, e_params)
		e_grads, e_var = zip(*e_grads_and_var)

		if max_grad_norm is not None:
			# Clip the gradients (normalize)
			e_grads, _e_grad_norm = tf.clip_by_global_norm(e_grads, max_grad_norm)
		e_grads_and_var = list(zip(e_grads, e_var))
		# zip aggregate each gradient with parameters associated
		# For instance zip(ABCD, xyza) => Ax, By, Cz, Da

		self.e_grads = e_grads
		self.e_var = e_var
		self._e_train_op = self.e_trainer.apply_gradients(e_grads_and_var)
		self.e_loss_names = ['e_policy_loss', 'e_value_loss', 'e_policy_entropy', 'e_approxkl', 'e_clipfrac']
		self.e_stats_list = [e_pg_loss, e_vf_loss, e_entropy, e_approxkl, e_clipfrac]

		self.e_train_model = e_train_model
		self.e_act_model = e_act_model
		self.e_value = e_act_model.value
		self.e_initial_state = e_act_model.initial_state
		self.e_step = e_act_model.step

		# END OF E_MODEL

		# BEGIN OF C_MODEL

		self.c_A = c_A = c_train_model.pdtype.sample_placeholder([None])
		self.c_ADV = c_ADV = tf.placeholder(tf.float32, [None])
		self.c_R = c_R = tf.placeholder(tf.float32, [None])
		# Keep track of old actor
		self.c_OLDNEGLOGPAC = c_OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
		# Keep track of old critic
		self.c_OLDVPRED = c_OLDVPRED = tf.placeholder(tf.float32, [None])
		self.c_LR = c_LR = tf.placeholder(tf.float32, [])
		# Cliprange
		self.c_CLIPRANGE = c_CLIPRANGE = tf.placeholder(tf.float32, [])

		c_neglogpac = c_train_model.pd.neglogp(c_A)

		# Calculate the entropy
		# Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
		c_entropy = tf.reduce_mean(c_train_model.pd.entropy())

		# CALCULATE THE LOSS
		# Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

		# Clip the value to reduce variability during Critic training
		# Get the predicted value
		c_vpred = c_train_model.vf
		c_vpredclipped = c_OLDVPRED + tf.clip_by_value(c_train_model.vf - c_OLDVPRED, - c_CLIPRANGE, c_CLIPRANGE)
		# Unclipped value
		c_vf_losses1 = tf.square(c_vpred - c_R)
		# Clipped value
		c_vf_losses2 = tf.square(c_vpredclipped - c_R)

		c_vf_loss = .5 * tf.reduce_mean(tf.maximum(c_vf_losses1, c_vf_losses2))

		# Calculate ratio (pi current policy / pi old policy)
		c_ratio = tf.exp(c_OLDNEGLOGPAC - c_neglogpac)

		# Defining Loss = - J is equivalent to max J
		c_pg_losses = -c_ADV * c_ratio

		c_pg_losses2 = -c_ADV * tf.clip_by_value(c_ratio, 1.0 - c_CLIPRANGE, 1.0 + c_CLIPRANGE)

		# Final PG loss
		c_pg_loss = tf.reduce_mean(tf.maximum(c_pg_losses, c_pg_losses2))
		c_approxkl = .5 * tf.reduce_mean(tf.square(c_neglogpac - c_OLDNEGLOGPAC))
		c_clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(c_ratio - 1.0), c_CLIPRANGE)))

		# Total loss
		c_loss = c_vf_loss * vf_coef

		# UPDATE THE PARAMETERS USING LOSS
		# 1. Get the model parameters
		c_params = tf.trainable_variables('ppo2_model_%i_c' % agent_index)
		# 2. Build our trainer
		if MPI is not None:
			self.c_trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=c_LR, epsilon=1e-5)
		else:
			self.c_trainer = tf.train.AdamOptimizer(learning_rate=c_LR, epsilon=1e-5)
		# 3. Calculate the gradients
		c_grads_and_var = self.c_trainer.compute_gradients(c_loss, c_params)
		c_grads, c_var = zip(*c_grads_and_var)

		if max_grad_norm is not None:
			# Clip the gradients (normalize)
			c_grads, _c_grad_norm = tf.clip_by_global_norm(c_grads, max_grad_norm)
		c_grads_and_var = list(zip(c_grads, c_var))
		# zip aggregate each gradient with parameters associated
		# For instance zip(ABCD, xyza) => Ax, By, Cz, Da

		self.c_grads = c_grads
		self.c_var = c_var
		self._c_train_op = self.c_trainer.apply_gradients(c_grads_and_var)
		self.c_loss_names = ['c_policy_loss', 'c_valuc_loss', 'c_policy_entropy', 'c_approxkl', 'c_clipfrac']
		self.c_stats_list = [c_pg_loss, c_vf_loss, c_entropy, c_approxkl, c_clipfrac]

		self.c_train_model = c_train_model
		self.c_act_model = c_act_model
		self.c_value = c_act_model.value
		self.c_initial_state = c_act_model.initial_state
		self.c_step = c_act_model.step

		self.save = functools.partial(save_variables, sess=sess)
		self.load = functools.partial(load_variables, sess=sess)

		# END OF C_MODEL

		initialize()
		global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
		if MPI is not None:
			sync_from_root(sess, global_variables)  # pylint: disable=E1101

	def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
		# Here we calculate advantage A(s,a) = R + yV(s') - V(s)
		# Returns = R + yV(s')
		advs = returns - values

		# Normalize the advantages
		advs = (advs - advs.mean()) / (advs.std() + 1e-8)

		td_map = {
			self.train_model.X: obs,
			self.A: actions,
			self.ADV: advs,
			self.R: returns,
			self.LR: lr,
			self.CLIPRANGE: cliprange,
			self.OLDNEGLOGPAC: neglogpacs,
			self.OLDVPRED: values
		}
		if states is not None:
			td_map[self.train_model.S] = states
			td_map[self.train_model.M] = masks

		return self.sess.run(
			self.stats_list + [self._train_op],
			td_map
		)[:-1]

	def e_train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
		# Here we calculate advantage A(s,a) = R + yV(s') - V(s)
		# Returns = R + yV(s')
		advs = returns - values

		# Normalize the advantages
		advs = (advs - advs.mean()) / (advs.std() + 1e-8)

		td_map = {
			self.e_train_model.X: obs,
			self.e_A: actions,
			self.e_ADV: advs,
			self.e_R: returns,
			self.e_LR: lr,
			self.e_CLIPRANGE: cliprange,
			self.e_OLDNEGLOGPAC: neglogpacs,
			self.e_OLDVPRED: values
		}
		if states is not None:
			td_map[self.e_train_model.S] = states
			td_map[self.e_train_model.M] = masks

		return self.sess.run(
			self.e_stats_list + [self._e_train_op],
			td_map
		)[:-1]

	def c_train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
		# Here we calculate advantage A(s,a) = R + yV(s') - V(s)
		# Returns = R + yV(s')
		advs = returns - values

		# Normalize the advantages
		advs = (advs - advs.mean()) / (advs.std() + 1e-8)

		td_map = {
			self.c_train_model.X: obs,
			self.c_A: actions,
			self.c_ADV: advs,
			self.c_R: returns,
			self.c_LR: lr,
			self.c_CLIPRANGE: cliprange,
			self.c_OLDNEGLOGPAC: neglogpacs,
			self.c_OLDVPRED: values
		}
		if states is not None:
			td_map[self.c_train_model.S] = states
			td_map[self.c_train_model.M] = masks

		return self.sess.run(
			self.c_stats_list + [self._c_train_op],
			td_map
		)[:-1]
