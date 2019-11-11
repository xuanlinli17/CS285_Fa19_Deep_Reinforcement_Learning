import numpy as np
from matplotlib import pyplot as plt

a = np.loadtxt("run_mb_obstacles_obstacles-cs285-v0_02-11-2019_01-00-16-tag-Eval_AverageReturn.csv", skiprows=1, delimiter=",")
plt.plot(a[:,1], a[:,2])
plt.title("Obstacles")
plt.savefig("q31.png")
plt.close()
a = np.loadtxt("run_mb_reacher_reacher-cs285-v0_02-11-2019_07-24-03-tag-Eval_AverageReturn.csv", skiprows=1, delimiter=",")
plt.plot(a[:,1], a[:,2])
plt.title("Reacher")
plt.savefig("q32.png")
plt.close()
a = np.loadtxt("run_mb_cheetah_cheetah-cs285-v0_02-11-2019_07-25-23-tag-Eval_AverageReturn.csv", skiprows=1, delimiter=",")
plt.plot(a[:,1], a[:,2])
plt.title("Cheetah")
plt.savefig("q33.png")
plt.close()