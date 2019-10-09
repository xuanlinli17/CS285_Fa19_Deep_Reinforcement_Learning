import numpy as np
from matplotlib import pyplot as plt

def main():
    a = np.loadtxt("run_pg_ll_b40000_r0.005_eval5000_LunarLanderContinuous-v2_29-09-2019_07-40-18-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2])
    plt.savefig('6.png')
    plt.close()
    
if __name__ == '__main__':
    main()