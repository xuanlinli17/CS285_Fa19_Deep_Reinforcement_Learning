import numpy as np
from matplotlib import pyplot as plt

def main():
    a = np.loadtxt("run_pg_walker_gae_b10000_r0.005_eval5000_Walker2d-v2_30-09-2019_15-57-50-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="gae")
    a = np.loadtxt("run_pg_walker_nogae_b10000_r0.005_eval5000_Walker2d-v2_30-09-2019_16-06-42-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="baseline")
    plt.legend()
    plt.savefig('bonus.png')
    plt.close()
    
    a = np.loadtxt("run_pg_walker_gae_b10000_r0.005_eval5000_Walker2d-v2_30-09-2019_15-57-50-tag-TimeSinceStart.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="gae")
    a = np.loadtxt("run_pg_walker_nogae_b10000_r0.005_eval5000_Walker2d-v2_30-09-2019_16-06-42-tag-TimeSinceStart.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="baseline")
    plt.legend()
    plt.savefig('bonus_time.png')
    plt.close()
    
if __name__ == '__main__':
    main()