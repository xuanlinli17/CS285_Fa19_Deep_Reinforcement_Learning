# CS285 hw2

## Question 3

```
python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 1000 -dsa --exp_name sb_no_rtg_dsa

python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa --exp_name sb_rtg_dsa

python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name sb_rtg_na

python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 5000 -dsa --exp_name lb_no_rtg_dsa

python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa --exp_name lb_rtg_dsa

python run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 5000 -rtg--exp_name lb_rtg_na
```

To plot the two figures, run:
```
python cs285/plot_3.py
```

## Question 4

```
python run_hw2_policy_gradient.py --env_name  InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 100 -lr 0.008 -rtg --exp_name ip_b100_r8e-3
```

To plot, run:
```
python cs285/plot_4.py
```

## Question 6

```
python run_hw2_policy_gradient.py --env_name LunarLanderContinuous-v2 --ep_len 1000 --discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 -rtg       --nn_baseline --exp_name ll_b40000_r0.005 --eval_batch_size 5000
```

To plot, run:
```
python cs285/plot_6.py
```

## Question 7

```
python run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.02 --video_log_freq -1 --reward_to_go --nn_baseline --eval_batch_size 5000 --exp_name hc_b30000_lr2e-2_nnbaseline

python run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.02 --exp_name hc_b30000_r2e-2

python run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.02 -rtg --exp_name hc_b30000_r2e-2

python run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.02 --nn_baseline --exp_name hc_b30000_r2e-2

python run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.02 -rtg --nn_baseline --exp_name hc_b30000_r2e-2
```

To plot the two figures, run:
```
python cs285/plot_7.py
```

## Bonus

```
 python run_hw2_policy_gradient.py --env_name Walker2d-v2 --ep_len 1000 --discount 0.99 --lambda 0.95 -n 100 -l 2 -s 64 -b 10000 -lr 0.005 -rtg --nn_baseline --exp_name walker_gae_b10000_r0.005_eval5000 --eval_batch_size 5000 --use_gae=True
 
  python run_hw2_policy_gradient.py --env_name Walker2d-v2 --ep_len 1000 --discount 0.99 -n 100 -l 2 -s 64 -b 10000 -lr 0.005 -rtg --nn_baseline --exp_name walker_nogae_b10000_r0.005_eval5000 --eval_batch_size 5000
```

To plot, run:
```
python cs285/plot_bonus.py
```
