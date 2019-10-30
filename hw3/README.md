## CS285 HW3

### Question 1
```
python run_hw3_dqn.py --env_name PongNoFrameskip-v4 --exp_name q1
```
To plot, run
```
python plot_1.py
```

### Question 2
```
python run_hw3_dqn.py --env_name LunarLander-v2 --exp_name q2_dqn_1 --seed 1

python run_hw3_dqn.py --env_name LunarLander-v2 --exp_name q2_dqn_2 --seed 2

python run_hw3_dqn.py --env_name LunarLander-v2 --exp_name q2_dqn_3 --seed 3

python run_hw3_dqn.py --env_name LunarLander-v2 --exp_name q2_doubledqn_1 --double_q --seed 1

python run_hw3_dqn.py --env_name LunarLander-v2 --exp_name q2_doubledqn_2 --double_q --seed 2

python run_hw3_dqn.py --env_name LunarLander-v2 --exp_name q2_doubledqn_3 --double_q --seed 3
```
To plot, run
```
python plot_2.py
```

### Question 3
```
python run_hw3_dqn.py --env_name PongNoFrameskip-v4 --exp_name q3_hparam1 --batch_size 16

python run_hw3_dqn.py --env_name PongNoFrameskip-v4 --exp_name q3_hparam2 --batch_size 64

python run_hw3_dqn.py --env_name PongNoFrameskip-v4 --exp_name q3_hparam3 --batch_size 128
```
To plot, run
```
python plot_3.py
```

### Question 4
```
python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name 1_1 -ntu 1 -ngsptu 1

python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name 100_1 -ntu 100 -ngsptu 1

python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name 1_100 -ntu 1 -ngsptu 100

python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name 10_10 -ntu 10 -ngsptu 10
```
To plot, run
```
python plot_4.py
```

### Question 5
```
python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name 10_10 -ntu 10 -ngsptu 10

python run_hw3_actor_critic.py --env_name HalfCheetah-v2--ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name 10_10 -ntu 10 -ngsptu 10
```
To plot, run
```
python plot_5.py
```