import os
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.policies import build_policy

try:
	from mpi4py import MPI
except ImportError:
	MPI = None
from baselines.ppo2.runner import Runner


def constfn(val):
	def f(_):
		return val

	return f


def learn(*, network, env, total_timesteps, eval_env=None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=1, noptepochs=4, cliprange=0.2,
          save_interval=1000, load_path=None, model_fn=None, **network_kwargs):
	'''
	Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

	Parameters:
	----------

	network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
									  specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
									  tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
									  neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
									  See common/models.py/lstm for more details on using recurrent nets in policies

	env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
									  The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


	nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
									  nenv is number of environment copies simulated in parallel)

	total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

	ent_coef: float                   policy entropy coefficient in the optimization objective

	lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
									  training and 0 is the end of the training.

	vf_coef: float                    value function loss coefficient in the optimization objective

	max_grad_norm: float or None      gradient norm clipping coefficient

	gamma: float                      discounting factor

	lam: float                        advantage estimation discounting factor (lambda in the paper)

	log_interval: int                 number of timesteps between logging events

	nminibatches: int                 number of training minibatches per update. For recurrent policies,
									  should be smaller or equal than number of environments run in parallel.

	noptepochs: int                   number of training epochs per update

	cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
									  and 0 is the end of the training

	save_interval: int                number of timesteps between saving events

	load_path: str                    path to load the model from

	**network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
									  For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



	'''

	nsteps = env.args.nsteps

	set_global_seeds(seed)

	if isinstance(lr, float):
		lr = constfn(lr)
	else:
		assert callable(lr)
	if isinstance(cliprange, float):
		cliprange = constfn(cliprange)
	else:
		assert callable(cliprange)
	total_timesteps = int(total_timesteps)

	# Get state_space and action_space
	ob_space = env.observation_space
	ac_space = env.action_space

	# Get the nb of env
	nenvs = env.num_envs

	# Calculate the batch_size
	nbatch = nenvs * nsteps
	nbatch_train = nbatch // nminibatches

	models = []
	policy = []

	for agent_i in range(env.spec):
		policy.append(build_policy(env, network, **network_kwargs))

		# Instantiate the model object (that creates act_model and train_model)
		if model_fn is None:
			from baselines.ppo2.model import Model
			model_fn = Model

		model = model_fn(policy=policy[agent_i], ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
		                 nbatch_train=nbatch_train,
		                 nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
		                 max_grad_norm=max_grad_norm, agent_index=agent_i)

		if load_path is not None:
			model.load(load_path + ('checkpoints-%i/' % agent_i) + env.args.s_load_num)
			print('successfully load agent-%d' % agent_i)
		# Instantiate the runner object

		models.append(model)

	# ###
	runner = Runner(env=env, model_n=models, nsteps=nsteps, gamma=gamma, lam=lam)
	if eval_env is not None:
		eval_runner = Runner(env=eval_env, model_n=models, nsteps=nsteps, gamma=gamma, lam=lam)

	epinfobuf = deque(maxlen=100)
	all_rewards_buf_0 = deque(maxlen=100)
	all_rewards_buf_1 = deque(maxlen=100)
	d_rewards_buf_0 = deque(maxlen=100)
	ce_rewards_buf_0 = deque(maxlen=100)
	c_rewards_buf_show_0 = deque(maxlen=100)
	c_rewards_buf_t_0 = deque(maxlen=100)
	c_rewards_buf_r_0 = deque(maxlen=100)
	c_rewards_buf_v_0 = deque(maxlen=100)
	ext_rewards_buf_v_0 = deque(maxlen=100)
	int_rewards_buf_v_0 = deque(maxlen=100)
	c_rewards_buf_tv_0 = deque(maxlen=100)
	ext_rewards_buf_tv_0 = deque(maxlen=100)
	int_rewards_buf_tv_0 = deque(maxlen=100)
	c_rewards_buf_all_0 = deque(maxlen=100)
	p_rewards_buf = deque(maxlen=100)
	d_rewards_buf_1 = deque(maxlen=100)
	ce_rewards_buf_1 = deque(maxlen=100)
	c_rewards_buf_show_1 = deque(maxlen=100)
	c_rewards_buf_t_1 = deque(maxlen=100)
	c_rewards_buf_r_1 = deque(maxlen=100)
	c_rewards_buf_v_1 = deque(maxlen=100)
	ext_rewards_buf_v_1 = deque(maxlen=100)
	int_rewards_buf_v_1 = deque(maxlen=100)
	c_rewards_buf_tv_1 = deque(maxlen=100)
	ext_rewards_buf_tv_1 = deque(maxlen=100)
	int_rewards_buf_tv_1 = deque(maxlen=100)
	c_rewards_buf_all_1 = deque(maxlen=100)
	if eval_env is not None:
		eval_epinfobuf = deque(maxlen=100)

	# Start total timer
	tfirststart = time.perf_counter()

	nupdates = total_timesteps // nbatch
	print(total_timesteps)
	print(nupdates)
	for update in range(1, nupdates + 1):
		assert nbatch % nminibatches == 0
		# Start timer
		tstart = time.perf_counter()
		frac = 1.0 - (update - 1.0) / nupdates
		# Calculate the learning rate
		lrnow = lr(frac)
		# Calculate the cliprange
		cliprangenow = cliprange(frac)

		# Get minibatch
		obs_n, returns_n, masks_n, actions_n, values_n, neglogpacs_n, e_returns_n, e_values_n, e_neglogpacs_n, \
		c_returns_n, c_values_n, c_neglogpacs_n, \
		states_n, e_states_n, c_states_n, epinfos, all_rewards, d_rewards, c_rewards_show, c_rewards_t, c_rewards_r, \
		c_rewards_v, c_rewards_tv, c_rewards_all, p_rewards, ce_rewards, \
		ext_rewards_tv_n, int_rewards_tv_n, ext_rewards_v_n, int_rewards_v_n = runner.run()  # pylint: disable=E0632

		if eval_env is not None:
			eval_obs_n, eval_returns_n, eval_masks_n, eval_actions_n, eval_values_n, eval_neglogpacs_n, eval_states_n, \
			eval_epinfos = eval_runner.run()  # pylint: disable=E0632

		num_env = p_rewards.shape[1]
		epinfobuf.append(1. * np.sum(epinfos) / (np.sum(masks_n[0]) + num_env))
		all_rewards_buf_0.append(1. * np.sum(all_rewards[:, 0]) / (np.sum(masks_n[0]) + num_env))
		d_rewards_buf_0.append(1. * np.sum(d_rewards[:, 0]) / (np.sum(masks_n[0]) + num_env))
		ce_rewards_buf_0.append(1. * np.sum(ce_rewards[:, 0]) / (np.sum(masks_n[0]) + num_env))
		c_rewards_buf_show_0.append(1. * np.sum(c_rewards_show[:, 0]) / (np.sum(masks_n[0]) + num_env))
		c_rewards_buf_r_0.append(1. * np.sum(c_rewards_r[:, 0]) / (np.sum(masks_n[0]) + num_env))
		c_rewards_buf_v_0.append(1. * np.sum(c_rewards_v[:, 0]) / (np.sum(masks_n[0]) + num_env))
		ext_rewards_buf_v_0.append(1. * np.sum(ext_rewards_v_n[:, 0]) / (np.sum(masks_n[0]) + num_env))
		int_rewards_buf_v_0.append(1. * np.sum(int_rewards_v_n[:, 0]) / (np.sum(masks_n[0]) + num_env))
		c_rewards_buf_t_0.append(1. * np.sum(c_rewards_t[:, 0]) / (np.sum(masks_n[0]) + num_env))
		c_rewards_buf_tv_0.append(1. * np.sum(c_rewards_tv[:, 0]) / (np.sum(masks_n[0]) + num_env))
		ext_rewards_buf_tv_0.append(1. * np.sum(ext_rewards_tv_n[:, 0]) / (np.sum(masks_n[0]) + num_env))
		int_rewards_buf_tv_0.append(1. * np.sum(int_rewards_tv_n[:, 0]) / (np.sum(masks_n[0]) + num_env))
		c_rewards_buf_all_0.append(1. * np.sum(c_rewards_all[:, 0]) / (np.sum(masks_n[0]) + num_env))
		all_rewards_buf_1.append(1. * np.sum(all_rewards[:, 1]) / (np.sum(masks_n[0]) + num_env))
		d_rewards_buf_1.append(1. * np.sum(d_rewards[:, 1]) / (np.sum(masks_n[0]) + num_env))
		ce_rewards_buf_1.append(1. * np.sum(ce_rewards[:, 1]) / (np.sum(masks_n[0]) + num_env))
		c_rewards_buf_show_1.append(1. * np.sum(c_rewards_show[:, 1]) / (np.sum(masks_n[0]) + num_env))
		c_rewards_buf_r_1.append(1. * np.sum(c_rewards_r[:, 1]) / (np.sum(masks_n[0]) + num_env))
		c_rewards_buf_v_1.append(1. * np.sum(c_rewards_v[:, 1]) / (np.sum(masks_n[0]) + num_env))
		ext_rewards_buf_v_1.append(1. * np.sum(ext_rewards_v_n[:, 1]) / (np.sum(masks_n[0]) + num_env))
		int_rewards_buf_v_1.append(1. * np.sum(int_rewards_v_n[:, 1]) / (np.sum(masks_n[0]) + num_env))
		c_rewards_buf_t_1.append(1. * np.sum(c_rewards_t[:, 1]) / (np.sum(masks_n[0]) + num_env))
		c_rewards_buf_tv_1.append(1. * np.sum(c_rewards_tv[:, 1]) / (np.sum(masks_n[0]) + num_env))
		ext_rewards_buf_tv_1.append(1. * np.sum(ext_rewards_tv_n[:, 1]) / (np.sum(masks_n[0]) + num_env))
		int_rewards_buf_tv_1.append(1. * np.sum(int_rewards_tv_n[:, 1]) / (np.sum(masks_n[0]) + num_env))
		c_rewards_buf_all_1.append(1. * np.sum(c_rewards_all[:, 1]) / (np.sum(masks_n[0]) + num_env))

		p_rewards_buf.append(-1. * np.sum(p_rewards) / (np.sum(masks_n[0]) + num_env))

		# Here what we're going to do is for each minibatch calculate the loss and append it.
		mblossvals_n = [[] for _ in range(env.spec)]
		mb_e_lossvals_n = [[] for _ in range(env.spec)]
		mb_c_lossvals_n = [[] for _ in range(env.spec)]

		if states_n[0] is None:  # nonrecurrent version
			# Index of each element of batch_size
			# Create the indices array
			for agent_i in range(env.spec):
				inds = np.arange(nbatch)
				for _ in range(noptepochs):
					# Randomize the indexes
					np.random.shuffle(inds)
					# 0 to batch_size with batch_train_size step
					for start in range(0, nbatch, nbatch_train):
						end = start + nbatch_train
						mbinds = inds[start:end]

						# ########### TRAIN MODEL
						slices = (arr[mbinds] for arr in (obs_n[agent_i], returns_n[agent_i], masks_n[agent_i],
						                                  actions_n[agent_i], values_n[agent_i], neglogpacs_n[agent_i]))
						mblossvals_n[agent_i].append(models[agent_i].train(lrnow, cliprangenow, *slices))

						# ########## TRAIN E_MODEL
						e_slices = (arr[mbinds] for arr in (obs_n[agent_i], e_returns_n[agent_i], masks_n[agent_i],
						                                     actions_n[agent_i], e_values_n[agent_i],
						                                     e_neglogpacs_n[agent_i]))
						if env.args.s_alg_name == 'noisy' or env.args.s_alg_name == 'cen' or \
								env.args.s_alg_name == 'dec':
							mb_e_lossvals_n[agent_i].append(0)
						else:
							mb_e_lossvals_n[agent_i].append(models[agent_i].e_train(lrnow, cliprangenow, *e_slices))

						# ########## TRAIN C_MODEL
						c_slices = (arr[mbinds] for arr in
						             (obs_n[agent_i], c_returns_n[agent_i], masks_n[agent_i],
						              actions_n[agent_i], c_values_n[agent_i],
						              c_neglogpacs_n[agent_i]))
						if env.args.s_alg_name == 'noisy' or env.args.s_alg_name == 'cen' or \
								env.args.s_alg_name == 'dec':
							mb_c_lossvals_n[agent_i].append(0)
						else:
							mb_c_lossvals_n[agent_i].append(
								models[agent_i].c_train(lrnow, cliprangenow, *c_slices))

		else:  # recurrent version
			assert nenvs % nminibatches == 0
			envsperbatch = nenvs // nminibatches

			for agent_i in range(env.spec):
				envinds = np.arange(nenvs)
				flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
				for _ in range(noptepochs):
					np.random.shuffle(envinds)
					for start in range(0, nenvs, envsperbatch):
						end = start + envsperbatch
						mbenvinds = envinds[start:end]
						mbflatinds = flatinds[mbenvinds].ravel()
						slices = (arr[mbflatinds] for arr in (obs_n[agent_i], returns_n[agent_i], masks_n[agent_i],
						                                      actions_n[agent_i], values_n[agent_i],
						                                      neglogpacs_n[agent_i]))
						mbstates = states_n[agent_i][mbenvinds]
						mblossvals_n[agent_i].append(models[agent_i].train(lrnow, cliprangenow, *slices, mbstates))

						e_slices = (arr[mbflatinds] for arr in
						             (obs_n[agent_i], e_returns_n[agent_i], masks_n[agent_i],
						              actions_n[agent_i], e_values_n[agent_i],
						              e_neglogpacs_n[agent_i]))
						e_mbstates = e_states_n[agent_i][mbenvinds]
						if env.args.s_alg_name == 'noisy' or env.args.s_alg_name == 'cen' or \
								env.args.s_alg_name == 'dec':
							mb_e_lossvals_n[agent_i].append(0)
						else:
							mb_e_lossvals_n[agent_i].append(
								models[agent_i].e_train(lrnow, cliprangenow, *e_slices, e_mbstates))

						c_slices = (arr[mbflatinds] for arr in
						             (obs_n[agent_i], c_returns_n[agent_i], masks_n[agent_i],
						              actions_n[agent_i], c_values_n[agent_i],
						              c_neglogpacs_n[agent_i]))
						c_mbstates = c_states_n[agent_i][mbenvinds]
						if env.args.s_alg_name == 'noisy' or env.args.s_alg_name == 'cen' or \
								env.args.s_alg_name == 'dec':
							mb_c_lossvals_n[agent_i].append(0)
						else:
							mb_c_lossvals_n[agent_i].append(
								models[agent_i].c_train(lrnow, cliprangenow, *c_slices, c_mbstates))

		# Feedforward --> get losses --> update
		lossvals_n = [np.mean(mblossvals_n[agent_i], axis=0) for agent_i in range(env.spec)]
		e_lossvals_n = [np.mean(mb_e_lossvals_n[agent_i], axis=0) for agent_i in range(env.spec)]
		c_lossvals_n = [np.mean(mb_c_lossvals_n[agent_i], axis=0) for agent_i in range(env.spec)]

		# End timer
		tnow = time.perf_counter()
		# Calculate the fps (frame per second)
		fps = int(nbatch / (tnow - tstart))

		if update % log_interval == 0 or update == 1:
			# Calculates if value function is a good predicator of the returns (ev > 1)
			# or if it's just worse than predicting nothing (ev =< 0)
			logger.logkv("serial_timesteps", update * nsteps)
			logger.logkv("nupdates", update)
			logger.logkv("total_timesteps", update * nbatch)
			logger.logkv("fps", fps)
			logger.logkv('time_elapsed', tnow - tfirststart)
			logger.logkv('eprewmean', safemean([epinfo for epinfo in epinfobuf]))
			logger.logkv('ep_all_rewmean_0', safemean([all_rewards for all_rewards in all_rewards_buf_0]))
			logger.logkv('ep_all_rewmean_1', safemean([all_rewards for all_rewards in all_rewards_buf_1]))
			logger.logkv('ep_dec_rewmean_0', safemean([all_rewards for all_rewards in d_rewards_buf_0]))
			logger.logkv('ep_cen_rewmean_0', safemean([all_rewards for all_rewards in ce_rewards_buf_0]))
			logger.logkv('ep_coor_rewmean_show_0', safemean([all_rewards for all_rewards in c_rewards_buf_show_0]))
			logger.logkv('ep_coor_rewmean_r_0', safemean([all_rewards for all_rewards in c_rewards_buf_r_0]))
			logger.logkv('ep_coor_rewmean_v_0', safemean([all_rewards for all_rewards in c_rewards_buf_v_0]))
			logger.logkv('ep_coor_rewmean_v_ext_0', safemean([all_rewards for all_rewards in ext_rewards_buf_v_0]))
			logger.logkv('ep_coor_rewmean_v_int_0', safemean([all_rewards for all_rewards in int_rewards_buf_v_0]))
			logger.logkv('ep_coor_rewmean_t_0', safemean([all_rewards for all_rewards in c_rewards_buf_t_0]))
			logger.logkv('ep_coor_rewmean_tv_0', safemean([all_rewards for all_rewards in c_rewards_buf_tv_0]))
			logger.logkv('ep_coor_rewmean_tv_ext_0', safemean([all_rewards for all_rewards in ext_rewards_buf_tv_0]))
			logger.logkv('ep_coor_rewmean_tv_int_0', safemean([all_rewards for all_rewards in int_rewards_buf_tv_0]))
			logger.logkv('ep_coor_rewmean_all_0', safemean([all_rewards for all_rewards in c_rewards_buf_all_0]))
			logger.logkv('ep_dec_rewmean_1', safemean([all_rewards for all_rewards in d_rewards_buf_1]))
			logger.logkv('ep_cen_rewmean_1', safemean([all_rewards for all_rewards in ce_rewards_buf_1]))
			logger.logkv('ep_coor_rewmean_show_1', safemean([all_rewards for all_rewards in c_rewards_buf_show_1]))
			logger.logkv('ep_coor_rewmean_r_1', safemean([all_rewards for all_rewards in c_rewards_buf_r_1]))
			logger.logkv('ep_coor_rewmean_v_1', safemean([all_rewards for all_rewards in c_rewards_buf_v_1]))
			logger.logkv('ep_coor_rewmean_v_ext_1', safemean([all_rewards for all_rewards in ext_rewards_buf_v_1]))
			logger.logkv('ep_coor_rewmean_v_int_1', safemean([all_rewards for all_rewards in int_rewards_buf_v_1]))
			logger.logkv('ep_coor_rewmean_t_1', safemean([all_rewards for all_rewards in c_rewards_buf_t_1]))
			logger.logkv('ep_coor_rewmean_tv_1', safemean([all_rewards for all_rewards in c_rewards_buf_tv_1]))
			logger.logkv('ep_coor_rewmean_tv_ext_1', safemean([all_rewards for all_rewards in ext_rewards_buf_tv_1]))
			logger.logkv('ep_coor_rewmean_tv_int_1', safemean([all_rewards for all_rewards in int_rewards_buf_tv_1]))
			logger.logkv('ep_coor_rewmean_all_1', safemean([all_rewards for all_rewards in c_rewards_buf_all_1]))
			logger.logkv('ep_penalty_rewmean', safemean([all_rewards for all_rewards in p_rewards_buf]))

			# logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))

			for agent_i in range(env.spec):
				ev = explained_variance(values_n[agent_i], returns_n[agent_i])
				logger.logkv("explained_variance-%i" % agent_i, float(ev))

				if eval_env is not None:
					logger.logkv('eval_eprewmean', safemean([epinfo['r']
					                                         for epinfo in eval_epinfobuf]))
					logger.logkv('eval_eplenmean', safemean([epinfo['l']
					                                         for epinfo in eval_epinfobuf]))
				for (lossval, lossname) in zip(lossvals_n[agent_i], models[agent_i].loss_names):
					logger.logkv(lossname + ('-%i' % agent_i), lossval)

				if env.args.s_alg_name == 'noisy' or env.args.s_alg_name == 'cen' or \
						env.args.s_alg_name == 'dec':
					pass
				else:
					for (lossval, lossname) in zip(e_lossvals_n[agent_i], models[agent_i].loss_names):
						logger.logkv(lossname + ('-e-%i' % agent_i), lossval)
					for (lossval, lossname) in zip(c_lossvals_n[agent_i], models[agent_i].loss_names):
						logger.logkv(lossname + ('-c-%i' % agent_i), lossval)

				if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
					logger.dumpkvs()
		if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and (
				MPI is None or MPI.COMM_WORLD.Get_rank() == 0):
			for i_m, m in enumerate(models):
				checkdir = osp.join(logger.get_dir(), 'checkpoints-%i' % i_m)
				os.makedirs(checkdir, exist_ok=True)
				savepath = osp.join(checkdir, '%.5i' % update)
				print('Saving to', savepath)
				m.save(savepath)
	return models


# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
	return np.nan if len(xs) == 0 else np.mean(xs)
