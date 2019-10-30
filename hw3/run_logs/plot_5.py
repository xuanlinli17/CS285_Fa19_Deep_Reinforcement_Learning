import numpy as np
from matplotlib import pyplot as plt

a = np.loadtxt("run_logs_ac_10_10_InvertedPendulum-v2_20-10-2019_16-16-35-tag-Train_AverageReturn.csv", skiprows=1, delimiter=",")
a = a[a[:,1]<3e6]
plt.plot(a[:,1], a[:,2])
plt.title("InvertedPendulum_10_10")
plt.savefig("q5_1.png")
plt.close()

a = np.loadtxt("run_ac_10_10_HalfCheetah-v2_17-10-2019_19-14-33-tag-Train_AverageReturn.csv", skiprows=1, delimiter=",")
a = a[a[:,1]<3e6]
plt.plot(a[:,1], a[:,2])
plt.title("HalfCheetah_10_10")
plt.savefig("q5_2.png")
plt.close()