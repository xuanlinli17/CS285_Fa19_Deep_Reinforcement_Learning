import numpy as np
from matplotlib import pyplot as plt

a = np.loadtxt("run_ac_1_1_CartPole-v0_17-10-2019_18-53-17-tag-Train_AverageReturn.csv", skiprows=1, delimiter=",")
a = a[a[:,1]<3e6]
plt.plot(a[:,1], a[:,2], label="1_1")

a = np.loadtxt("run_ac_1_100_CartPole-v0_17-10-2019_18-55-51-tag-Train_AverageReturn.csv", skiprows=1, delimiter=",")
a = a[a[:,1]<3e6]
plt.plot(a[:,1], a[:,2], label="1_100")

a = np.loadtxt("run_ac_100_1_CartPole-v0_17-10-2019_18-51-13-tag-Train_AverageReturn.csv", skiprows=1, delimiter=",")
a = a[a[:,1]<3e6]
plt.plot(a[:,1], a[:,2], label="100_1")

a = np.loadtxt("run_ac_10_10_CartPole-v0_17-10-2019_18-47-59-tag-Train_AverageReturn.csv", skiprows=1, delimiter=",")
a = a[a[:,1]<3e6]
plt.plot(a[:,1], a[:,2], label="10_10")

plt.title("Q4 Cartpole")
plt.legend()
plt.savefig("q4.png")
plt.close()