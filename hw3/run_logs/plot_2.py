import numpy as np
from matplotlib import pyplot as plt

dat = []
def adddata(path):
    a = np.loadtxt(path, skiprows=1, delimiter=",")
    xaxis = np.asarray(a[:,1])
    dat.append(a[:,2])
    return xaxis

xaxis = adddata("run_dqn_q2_dqn_1_LunarLander-v2_20-10-2019_14-27-13-tag-Train_AverageReturn.csv")
adddata("run_dqn_q2_dqn_2_LunarLander-v2_20-10-2019_17-19-20-tag-Train_AverageReturn.csv")
adddata("run_dqn_q2_dqn_3_LunarLander-v2_20-10-2019_16-05-55-tag-Train_AverageReturn.csv")
plt.plot(xaxis, np.mean(dat, axis=0), label="DQN")

dat = []
adddata("run_dqn_double_q_q2_doubledqn_1_LunarLander-v2_20-10-2019_14-53-19-tag-Train_AverageReturn.csv")
adddata("run_dqn_double_q_q2_doubledqn_2_LunarLander-v2_20-10-2019_15-42-19-tag-Train_AverageReturn.csv")
adddata("run_dqn_double_q_q2_doubledqn_3_LunarLander-v2_20-10-2019_16-28-39-tag-Train_AverageReturn.csv")
plt.plot(xaxis, np.mean(dat, axis=0), label="Double DQN")
plt.legend()
plt.title("Q2 LunarLander")
plt.savefig("q2.png")
plt.close()

dat = []
xaxis = adddata("run_dqn_q2_dqn_1_LunarLander-v2_20-10-2019_14-27-13-tag-Train_AverageReturn.csv")
adddata("run_dqn_q2_dqn_2_LunarLander-v2_20-10-2019_17-19-20-tag-Train_AverageReturn.csv")
adddata("run_dqn_q2_dqn_3_LunarLander-v2_20-10-2019_16-05-55-tag-Train_AverageReturn.csv")
plt.plot(xaxis, dat[0])
plt.plot(xaxis, dat[1])
plt.plot(xaxis, dat[2])
plt.title("DQN trials")
plt.savefig("q2_1.png")
plt.close()

dat = []
adddata("run_dqn_double_q_q2_doubledqn_1_LunarLander-v2_20-10-2019_14-53-19-tag-Train_AverageReturn.csv")
adddata("run_dqn_double_q_q2_doubledqn_2_LunarLander-v2_20-10-2019_15-42-19-tag-Train_AverageReturn.csv")
adddata("run_dqn_double_q_q2_doubledqn_3_LunarLander-v2_20-10-2019_16-28-39-tag-Train_AverageReturn.csv")
plt.plot(xaxis, dat[0])
plt.plot(xaxis, dat[1])
plt.plot(xaxis, dat[2])
plt.title("DOUBLE_DQN trials")
plt.savefig("q2_2.png")
plt.close()