"""
Helpers for scripts like run_atari.py.
"""

import os

try:
	from mpi4py import MPI
except ImportError:
	MPI = None

import gym
from gym.wrappers import FlattenDictWrapper
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, SubprocVecEnv_Pass, SubprocVecEnv_ThreePass, \
	SubprocVecEnv_Leftward, SubprocVecEnv_Island, SubprocVecEnv_PushBall, SubprocVecEnv_x_Island, SubprocVecEnv_x_Pass
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common import retro_wrappers
from baselines.pass_environment import Pass
from baselines.three_pass_environment import ThreePass
from baselines.leftward_environment import Leftward
from baselines.island_environment import Island
from baselines.pushball_environment import PushBall
from baselines.x_island_environment import x_Island
from baselines.x_pass_environment import x_Pass
from baselines.test_island_environment import test_Island


def make_pushball_env(env_id, env_type, num_env, seed, args, subrank=0, wrapper_kwargs=None, start_index=0,
                      reward_scale=1.0):
	env = PushBall(args, subrank)
	env.initialization(args)
	return env


def make_m_pushball_env(env_id, env_type, num_env, seed, args, wrapper_kwargs=None, start_index=0, reward_scale=1.0,
                        flatten_dict_observations=True,
                        gamestate=None):
	wrapper_kwargs = wrapper_kwargs or {}
	mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
	seed = seed + 10000 * mpi_rank if seed is not None else None
	logger_dir = logger.get_dir()

	def make_thunk(rank):
		return lambda: make_pushball_env(
			env_id=env_id,
			env_type=env_type,
			subrank=rank,
			num_env=1,
			seed=seed,
			args=args,
			reward_scale=reward_scale,
			wrapper_kwargs=wrapper_kwargs
		)

	return SubprocVecEnv_PushBall([make_thunk(i + start_index) for i in range(num_env)], args)


def make_x_island_env(env_id, env_type, num_env, seed, args, subrank=0, wrapper_kwargs=None, start_index=0,
                    reward_scale=1.0):
	#env = test_Island(args, subrank)
	env = x_Island(args, subrank)
	env.initialization(args)
	return env


def make_m_x_island_env(env_id, env_type, num_env, seed, args, wrapper_kwargs=None, start_index=0, reward_scale=1.0,
                      flatten_dict_observations=True,
                      gamestate=None):
	wrapper_kwargs = wrapper_kwargs or {}
	mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
	seed = seed + 10000 * mpi_rank if seed is not None else None
	logger_dir = logger.get_dir()

	def make_thunk(rank):
		return lambda: make_x_island_env(
			env_id=env_id,
			env_type=env_type,
			subrank=rank,
			num_env=1,
			seed=seed,
			args=args,
			reward_scale=reward_scale,
			wrapper_kwargs=wrapper_kwargs
		)

	return SubprocVecEnv_x_Island([make_thunk(i + start_index) for i in range(num_env)], args)


def make_island_env(env_id, env_type, num_env, seed, args, subrank=0, wrapper_kwargs=None, start_index=0,
                    reward_scale=1.0):
	env = Island(args, subrank)
	env.initialization(args)

	return env


def make_m_island_env(env_id, env_type, num_env, seed, args, wrapper_kwargs=None, start_index=0, reward_scale=1.0,
                      flatten_dict_observations=True,
                      gamestate=None):
	wrapper_kwargs = wrapper_kwargs or {}
	mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
	seed = seed + 10000 * mpi_rank if seed is not None else None
	logger_dir = logger.get_dir()

	def make_thunk(rank):
		return lambda: make_island_env(
			env_id=env_id,
			env_type=env_type,
			subrank=rank,
			num_env=1,
			seed=seed,
			args=args,
			reward_scale=reward_scale,
			wrapper_kwargs=wrapper_kwargs
		)

	return SubprocVecEnv_Island([make_thunk(i + start_index) for i in range(num_env)], args)


def make_three_pass_env(env_id, env_type, num_env, seed, args, subrank=0, wrapper_kwargs=None, start_index=0,
                        reward_scale=1.0):
	env = ThreePass(args, subrank)
	env.initialization(args)

	return env


def make_m_three_pass_env(env_id, env_type, num_env, seed, args, wrapper_kwargs=None, start_index=0, reward_scale=1.0,
                          flatten_dict_observations=True,
                          gamestate=None):
	wrapper_kwargs = wrapper_kwargs or {}
	mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
	seed = seed + 10000 * mpi_rank if seed is not None else None
	logger_dir = logger.get_dir()

	def make_thunk(rank):
		return lambda: make_three_pass_env(
			env_id=env_id,
			env_type=env_type,
			subrank=rank,
			num_env=1,
			seed=seed,
			args=args,
			reward_scale=reward_scale,
			wrapper_kwargs=wrapper_kwargs
		)

	return SubprocVecEnv_ThreePass([make_thunk(i + start_index) for i in range(num_env)], args)


def make_pass_env(env_id, env_type, num_env, seed, args, subrank=0, wrapper_kwargs=None, start_index=0,
                  reward_scale=1.0):
	env = Pass(args, subrank)
	env.initialization(args)

	return env


def make_multi_pass_env(env_id, env_type, num_env, seed, args, wrapper_kwargs=None, start_index=0, reward_scale=1.0,
                        flatten_dict_observations=True,
                        gamestate=None):
	wrapper_kwargs = wrapper_kwargs or {}
	mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
	seed = seed + 10000 * mpi_rank if seed is not None else None
	logger_dir = logger.get_dir()

	def make_thunk(rank):
		return lambda: make_pass_env(
			env_id=env_id,
			env_type=env_type,
			subrank=rank,
			num_env=1,
			seed=seed,
			args=args,
			reward_scale=reward_scale,
			wrapper_kwargs=wrapper_kwargs
		)

	return SubprocVecEnv_Pass([make_thunk(i + start_index) for i in range(num_env)], args)


def make_leftward_env(env_id, env_type, num_env, seed, args, subrank=0, wrapper_kwargs=None, start_index=0,
                      reward_scale=1.0):
	env = Leftward(args, subrank)
	env.initialization(args)

	return env


def make_m_leftward_env(env_id, env_type, num_env, seed, args, wrapper_kwargs=None, start_index=0, reward_scale=1.0,
                        flatten_dict_observations=True,
                        gamestate=None):
	wrapper_kwargs = wrapper_kwargs or {}
	mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
	seed = seed + 10000 * mpi_rank if seed is not None else None
	logger_dir = logger.get_dir()

	def make_thunk(rank):
		return lambda: make_leftward_env(
			env_id=env_id,
			env_type=env_type,
			subrank=rank,
			num_env=1,
			seed=seed,
			args=args,
			reward_scale=reward_scale,
			wrapper_kwargs=wrapper_kwargs
		)

	return SubprocVecEnv_Leftward([make_thunk(i + start_index) for i in range(num_env)], args)

def make_x_pass_env(env_id, env_type, num_env, seed, args, wrapper_kwargs=None, start_index=0, reward_scale=1.0,
                        flatten_dict_observations=True,
                        gamestate=None):
	wrapper_kwargs = wrapper_kwargs or {}
	mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
	seed = seed + 10000 * mpi_rank if seed is not None else None
	logger_dir = logger.get_dir()

	def make_thunk(rank):
		return lambda: make_pass_env(
			env_id=env_id,
			env_type=env_type,
			subrank=rank,
			num_env=1,
			seed=seed,
			args=args,
			reward_scale=reward_scale,
			wrapper_kwargs=wrapper_kwargs
		)

	return SubprocVecEnv_x_Pass([make_thunk(i + start_index) for i in range(num_env)], args)


def make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None):
	"""
	Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
	"""
	wrapper_kwargs = wrapper_kwargs or {}
	mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
	seed = seed + 10000 * mpi_rank if seed is not None else None
	logger_dir = logger.get_dir()

	def make_thunk(rank):
		return lambda: make_env(
			env_id=env_id,
			env_type=env_type,
			mpi_rank=mpi_rank,
			subrank=rank,
			seed=seed,
			reward_scale=reward_scale,
			gamestate=gamestate,
			flatten_dict_observations=flatten_dict_observations,
			wrapper_kwargs=wrapper_kwargs,
			logger_dir=logger_dir
		)

	set_global_seeds(seed)
	if num_env > 1:
		return SubprocVecEnv([make_thunk(i + start_index) for i in range(num_env)])
	else:
		return DummyVecEnv([make_thunk(start_index)])


def make_env(env_id, env_type, mpi_rank=0, subrank=0, seed=None, reward_scale=1.0, gamestate=None,
             flatten_dict_observations=True, wrapper_kwargs=None, logger_dir=None):
	wrapper_kwargs = wrapper_kwargs or {}
	if env_type == 'atari':
		env = make_atari(env_id)
	elif env_type == 'retro':
		import retro
		gamestate = gamestate or retro.State.DEFAULT
		env = retro_wrappers.make_retro(game=env_id, max_episode_steps=10000,
		                                use_restricted_actions=retro.Actions.DISCRETE, state=gamestate)
	else:
		env = gym.make(env_id)

	if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
		keys = env.observation_space.spaces.keys()
		env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))

	env.seed(seed + subrank if seed is not None else None)
	env = Monitor(env,
	              logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
	              allow_early_resets=True)

	if env_type == 'atari':
		env = wrap_deepmind(env, **wrapper_kwargs)
	elif env_type == 'retro':
		if 'frame_stack' not in wrapper_kwargs:
			wrapper_kwargs['frame_stack'] = 1
		env = retro_wrappers.wrap_deepmind_retro(env, **wrapper_kwargs)

	if reward_scale != 1:
		env = retro_wrappers.RewardScaler(env, reward_scale)

	return env


def make_mujoco_env(env_id, seed, reward_scale=1.0):
	"""
	Create a wrapped, monitored gym.Env for MuJoCo.
	"""
	rank = MPI.COMM_WORLD.Get_rank()
	myseed = seed + 1000 * rank if seed is not None else None
	set_global_seeds(myseed)
	env = gym.make(env_id)
	logger_path = None if logger.get_dir() is None else os.path.join(logger.get_dir(), str(rank))
	env = Monitor(env, logger_path, allow_early_resets=True)
	env.seed(seed)
	if reward_scale != 1.0:
		from baselines.common.retro_wrappers import RewardScaler
		env = RewardScaler(env, reward_scale)
	return env


def make_robotics_env(env_id, seed, rank=0):
	"""
	Create a wrapped, monitored gym.Env for MuJoCo.
	"""
	set_global_seeds(seed)
	env = gym.make(env_id)
	env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
	env = Monitor(
		env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
		info_keywords=('is_success',))
	env.seed(seed)
	return env


def arg_parser():
	"""
	Create an empty argparse.ArgumentParser.
	"""
	import argparse
	return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def atari_arg_parser():
	"""
	Create an argparse.ArgumentParser for run_atari.py.
	"""
	print('Obsolete - use common_arg_parser instead')
	return common_arg_parser()


def mujoco_arg_parser():
	print('Obsolete - use common_arg_parser instead')
	return common_arg_parser()


def common_arg_parser():
	"""
	Create an argparse.ArgumentParser for run_mujoco.py.
	"""
	parser = arg_parser()
	parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
	parser.add_argument('--env_type',
	                    help='type of environment, used when the environment type cannot be automatically determined',
	                    type=str)
	parser.add_argument('--seed', help='RNG seed', type=int, default=None)
	parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
	parser.add_argument('--num_timesteps', type=float, default=1e6),
	parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
	parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
	parser.add_argument('--num_env',
	                    help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco',
	                    default=1, type=int)
	parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
	parser.add_argument('--save_path', help='Path to save trained model to',
	                    default='../../results/PPO/try_1/Random_start/', type=str)
	parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
	parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
	parser.add_argument('--play', default=False, action='store_true')
	parser.add_argument('--nsteps', default=2048, type=int)
	parser.add_argument('--size', default=30, type=int)
	parser.add_argument('--n_action', default=4, type=int)
	parser.add_argument('--n_agent', default=2, type=int)
	parser.add_argument('--episode_length', default=300, type=int)
	parser.add_argument('--doi', default=8, type=int, help='door_open_interval')
	parser.add_argument('--penalty', default=0.0, type=float)
	parser.add_argument('--gamma_dec', default=0.0, type=float)
	parser.add_argument('--gamma_cen', default=0.0, type=float)
	parser.add_argument('--fix_start', default=False, action='store_true')

	parser.add_argument('--gamma_coor_r_e', default=0.0, type=float)
	parser.add_argument('--gamma_coor_tv_e', default=0.0, type=float)
	parser.add_argument('--gamma_coor_r_c', default=0.0, type=float)
	parser.add_argument('--gamma_coor_tv_c', default=0.0, type=float)
	parser.add_argument('--gamma_coor_v_e', default=0.0, type=float)
	parser.add_argument('--gamma_coor_v_c', default=0.0, type=float)
	parser.add_argument('--gamma_coor_r', default=0.0, type=float)
	parser.add_argument('--gamma_coor_tv', default=0.0, type=float)
	parser.add_argument('--gamma_coor_t', default=0.0, type=float)
	parser.add_argument('--gamma_coor_v', default=0.0, type=float)

	parser.add_argument('--symmetry', default=False, action='store_true')
	parser.add_argument('--simple_env', default=False, action='store_true')
	parser.add_argument('--r', default=False, action='store_true')
	parser.add_argument('--t', default=False, action='store_true')
	parser.add_argument('--tv', default=False, action='store_true')
	parser.add_argument('--v', default=False, action='store_true')
	parser.add_argument('--r_tv', default=False, action='store_true')
	parser.add_argument('--env_n_dim', default=2, type=int)
	parser.add_argument('--t_save_rate', default=1, type=int)
	parser.add_argument('--s_data_gather', default=False, action='store_true')
	parser.add_argument('--s_data_path', default='/data1/wjh/code/results/data/', type=str)
	parser.add_argument('--s_try_num', default=0, type=int)
	parser.add_argument('--s_alg_name', default='', type=str)
	parser.add_argument('--s_load_num', default='', type=str)

	parser.add_argument('--pushball_random_l', default=4, type=int)
	parser.add_argument('--pushball_random_r', default=12, type=int)

	parser.add_argument('--island_partial_obs', default=False, action='store_true')
	parser.add_argument('--island_agent_max_power', default=11, type=int)
	parser.add_argument('--island_wolf_max_power', default=9, type=int)
	parser.add_argument('--island_wolf_recover_time', default=5, type=int)
	parser.add_argument('--i_num_landmark', default=2, type=int)
	#parser.add_argument('--x_island_agent_max_power', default=11, type=int)
	#parser.add_argument('--x_island_wolf_max_power', default=10, type=int)
	parser.add_argument('--x_island_agent_max_power', default=51, type=int)
	parser.add_argument('--x_island_wolf_max_power', default=21, type=int)
	parser.add_argument('--x_island_wolf_recover_time', default=5, type=int)
	#parser.add_argument('--x_island_harm_range', default=3, type=int)
	parser.add_argument('--x_island_harm_range', default=11, type=int)
	parser.add_argument('--x_num_landmark', default=2, type=int)
	parser.add_argument('--x_wolf_rew', default=600, type=int)
	parser.add_argument('--x_landmark_rew', default=10, type=int)
	parser.add_argument('--not_view_landmark', default=False, action='store_true')
	parser.add_argument('--VI_lr_p', default=0.0001, type=float)
	parser.add_argument('--VI_lr_t', default=0.01, type=float)
	parser.add_argument('--VI_sample_alpha_p', default=1., type=float)
	parser.add_argument('--VI_sample_alpha_t', default=1., type=float)
	parser.add_argument('--appro_T', default=0.5, type=float)
	parser.add_argument('--VI_not_one_hot', default=False, action='store_true')
	parser.add_argument('--VI_print_steps', default=1000, type=int)
	parser.add_argument('--VI_batch_size', default=512, type=int)
	parser.add_argument('--VI_island_fully_pred', default=False, action='store_true')
	parser.add_argument('--VI_island_test', default=False, action='store_true')
	parser.add_argument('--VI_island_test_2', default=False, action='store_true')
	parser.add_argument('--VI_island_test_3', default=False, action='store_true')
	return parser


def robotics_arg_parser():
	"""
	Create an argparse.ArgumentParser for run_mujoco.py.
	"""
	parser = arg_parser()
	parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v0')
	parser.add_argument('--seed', help='RNG seed', type=int, default=None)
	parser.add_argument('--num-timesteps', type=int, default=int(1e6))
	return parser


def parse_unknown_args(args):
	"""
	Parse arguments not consumed by arg parser into a dicitonary
	"""
	retval = {}
	preceded_by_key = False
	for arg in args:
		if arg.startswith('--'):
			if '=' in arg:
				key = arg.split('=')[0][2:]
				value = arg.split('=')[1]
				retval[key] = value
			else:
				key = arg[2:]
				preceded_by_key = True
		elif preceded_by_key:
			retval[key] = arg
			preceded_by_key = False

	return retval
