import numpy as np
from matplotlib import pyplot as plt

a = np.loadtxt("run_dqn_q1_PongNoFrameskip-v4_17-10-2019_08-17-28-tag-Train_AverageReturn.csv", skiprows=1, delimiter=",")
a = a[a[:,1]<3e6]
plt.plot(a[:,1], a[:,2])
plt.title("Q1 PONG DQN")
plt.savefig("q1.png")
plt.close()