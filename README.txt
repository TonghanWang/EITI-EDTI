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


# Training
In the folder `baselines/`, the following command shows how to train EDIT.

$ python run.py

And the arguments of Pass-Door are

--alg=ppo2
--env=pass
--network=mlp
--num_timesteps=6e8
--ent_coef=0.1
--num_hidden=32
--num_layers=3
--value_network=copy
--save_path=../../tmp/results/EDIT/Pass/size_30/Coor_tv/try_1/
--fix_start
--doi=0
--penalty=0
--size=30
--episode_length=300
--num_env=32
--n_agent=2
--n_action=4
--gamma_dec=10.
--gamma_cen=0
--gamma_coor_r_e=0.1
--gamma_coor_tv_e=0.1
--gamma_coor_r_c=0.1
--gamma_coor_tv_c=1.
--gamma_coor_r=1.
--gamma_coor_tv=1.
--gamma_coor_t=0
--tv
--s_data_gather
--s_alg_name=coor_tv
--s_data_path=../../tmp/data/EDIT/
--s_try_num=1