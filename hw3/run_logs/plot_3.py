import numpy as np
from matplotlib import pyplot as plt

a = np.loadtxt("run_dqn_q3_hparam1_PongNoFrameskip-v4_17-10-2019_19-50-46-tag-Train_AverageReturn.csv", skiprows=1, delimiter=",")
a = a[a[:,1]<3e6]
plt.plot(a[:,1], a[:,2], label="batch=16")

a = np.loadtxt("run_dqn_q1_PongNoFrameskip-v4_17-10-2019_08-17-28-tag-Train_AverageReturn.csv", skiprows=1, delimiter=",")
a = a[a[:,1]<3e6]
plt.plot(a[:,1], a[:,2], label="batch=32")

a = np.loadtxt("run_dqn_q3_hparam2_PongNoFrameskip-v4_17-10-2019_19-51-06-tag-Train_AverageReturn.csv", skiprows=1, delimiter=",")
a = a[a[:,1]<3e6]
plt.plot(a[:,1], a[:,2], label="batch=64")

a = np.loadtxt("run_dqn_q3_hparam3_PongNoFrameskip-v4_18-10-2019_17-21-31-tag-Train_AverageReturn.csv", skiprows=1, delimiter=",")
a = a[a[:,1]<3e6]
plt.plot(a[:,1], a[:,2], label="batch=128")

plt.title("Q3 PONG DQN")
plt.legend()
plt.savefig("q3.png")
plt.close()