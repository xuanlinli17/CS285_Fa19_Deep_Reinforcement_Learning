import numpy as np
from matplotlib import pyplot as plt

a = np.loadtxt("run_mb_obstacles_singleiteration_obstacles-cs285-v0_02-11-2019_00-18-22-tag-Train_AverageReturn.csv", skiprows=1, delimiter=",")
train_avg = a[-1]
a = np.loadtxt("run_mb_obstacles_singleiteration_obstacles-cs285-v0_02-11-2019_00-18-22-tag-Eval_AverageReturn.csv", skiprows=1, delimiter=",")
eval_avg = a[-1]
plt.plot([0], [train_avg], 'o', label="train_avg")
plt.plot([0], [eval_avg], 'o', label="eval_avg")
plt.title("Train vs eval average for Obstacles environment")
plt.legend()
plt.savefig("q2.png")
plt.close()