import numpy as np
from matplotlib import pyplot as plt

def main():
    a = np.loadtxt("run-pg_sb_no_rtg_dsa_CartPole-v0_28-09-2019_17-47-22-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="no_rtg_dsa")
    a = np.loadtxt("run-pg_sb_rtg_dsa_CartPole-v0_28-09-2019_17-50-02-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="rtg_dsa")
    a = np.loadtxt("run-pg_sb_rtg_na_CartPole-v0_28-09-2019_18-40-38-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="rtg_na")
    plt.legend()
    plt.savefig('3_small.png')
    plt.close()
    a = np.loadtxt("run-pg_lb_no_rtg_dsa_CartPole-v0_28-09-2019_18-44-09-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="no_rtg_dsa")
    a = np.loadtxt("run-pg_lb_rtg_dsa_CartPole-v0_28-09-2019_18-49-13-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="rtg_dsa")
    a = np.loadtxt("run-pg_lb_rtg_na_CartPole-v0_28-09-2019_18-54-16-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="rtg_na")
    plt.legend()
    plt.savefig('3_large.png')
    plt.close()
    
if __name__ == '__main__':
    main()