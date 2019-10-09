import numpy as np
from matplotlib import pyplot as plt

def main():
    a = np.loadtxt("prob4-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2])
    plt.savefig('4.png')
    plt.close()
    
if __name__ == '__main__':
    main()