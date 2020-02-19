
# Influence-Based Multi-Agent Exploration

## Note
 This codebase accompanies the paper [Influence-Based Multi-Agent Exploration](https://openreview.net/forum?id=BJgy96EYvr&noteId=BJgy96EYvr), 
 and is based on the PPO2 implementation provided by [OpenAI Baselines](https://github.com/openai/baselines) codebase.

## Run an experiment 

In the folder `baselines/`, run the following command to train EDIT on the task *pass*.

```shell
python run.py
--alg=ppo2
--env=pass
--network=mlp
--num_timesteps=6e8
--ent_coef=0.1
--num_hidden=32
--num_layers=3
--value_network=copy
--save_path=$SAVE_PATH
--num_env=32
--gamma_dec=10.
--gamma_cen=0
--gamma_coor_tv_e=0.1
--gamma_coor_tv_c=1.
--gamma_coor_t=0
--tv
--s_data_gather
--s_alg_name=coor_tv
--s_data_path=$DATA_PATH
--s_try_num=1
```

Here, `gamma_dec` (or `gamma_cen`) is &eta; in the paper, `gamma_coor_tv_e` is &beta;<sub>ext</sub>, `gamma_coor_tv_c` is &beta;<sub>int</sub>, and `gamma_coor_t` is &beta;<sub>T</sub>. (See Table 2 on page 20 of the paper for the specific values.)

To run EITI, specify the option `--t`. 

To tun EDTI, specify the option `--r_tv`.

# Requirements
- TensorFlow >= 1.4.0
- TensorFlow < 2.0.0
- gym == 0.13.0
- numpy
- cv2
- imp
- mpi4py
- scipy
- matplotlib
