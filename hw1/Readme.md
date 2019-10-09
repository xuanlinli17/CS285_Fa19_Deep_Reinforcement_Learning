For problem 1(b), run 
```python
python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/HalfCheetah.pkl --env_name HalfCheetah-v2 --exp_name test_bc_hcheetah --n_iter 1 --expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl --batch_size=1000 --eval_batch_size=5000
```
```python
python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Humanoid.pkl --env_name Humanoid-v2 --exp_name test_bc_humanoid --n_iter 1 --expert_data cs285/expert_data/expert_data_Humanoid-v2.pkl --batch_size=1000 --eval_batch_size=5000
```
For problem 1(c), run
```python
python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/HalfCheetah.pkl --env_name HalfCheetah-v2 --exp_name test_bc_hcheetah --n_iter 1 --expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl --video_log_freq -1 --num_agent_train_steps_per_iter {num_iters} --batch_size=1000 --eval_batch_size=5000
```
For problem 2, run
```python
python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Humanoid.pkl --env_name Humanoid-v2 --exp_name test_dg_humanoid --n_iter 20 --expert_data cs285/expert_data/expert_data_Humanoid-v2.pkl --video_log_freq -1 --do_dagger --batch_size=1000 --eval_batch_size=5000
```