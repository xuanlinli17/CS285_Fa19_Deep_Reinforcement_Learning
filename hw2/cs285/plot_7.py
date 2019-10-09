import numpy as np
from matplotlib import pyplot as plt

def main():
    a = np.loadtxt("run_pg_hc_b10000_lr5e-3_eval5000_nnbaseline_HalfCheetah-v2_29-09-2019_07-47-23-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="lr_5e-3_b_10k")
    a = np.loadtxt("run_pg_hc_b30000_lr5e-3_eval5000_nnbaseline_HalfCheetah-v2_29-09-2019_07-51-54-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="lr_5e-3_b_30k")
    a = np.loadtxt("run_pg_hc_b50000_lr5e-3_eval5000_nnbaseline_HalfCheetah-v2_29-09-2019_07-53-36-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="lr_5e-3_b_50k")
    a = np.loadtxt("run_pg_hc_b10000_lr1e-2_eval5000_nnbaseline_HalfCheetah-v2_29-09-2019_08-00-37-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="lr_1e-2_b_10k")
    a = np.loadtxt("run_pg_hc_b30000_lr1e-2_eval5000_nnbaseline_HalfCheetah-v2_29-09-2019_08-01-51-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="lr_1e-2_b_30k")
    a = np.loadtxt("run_pg_hc_b50000_lr1e-2_eval5000_nnbaseline_HalfCheetah-v2_29-09-2019_08-02-37-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="lr_1e-2_b_50k")
    a = np.loadtxt("run_pg_hc_b10000_lr2e-2_eval5000_nnbaseline_HalfCheetah-v2_29-09-2019_08-03-25-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="lr_2e-2_b_10k")
    a = np.loadtxt("run_pg_hc_b30000_lr2e-2_eval5000_nnbaseline_HalfCheetah-v2_29-09-2019_08-04-10-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="lr_2e-2_b_30k")
    a = np.loadtxt("run_pg_hc_b50000_lr2e-2_eval5000_nnbaseline_HalfCheetah-v2_29-09-2019_08-04-51-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="lr_2e-2_b_50k")
    plt.legend()
    plt.savefig('7_1.png')
    plt.close()
    
    
    a = np.loadtxt("run_pg_hc_b30000_r2e-2_nortgnobase_HalfCheetah-v2_29-09-2019_19-33-58-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="no_rtg_no_baseline")
    a = np.loadtxt("run_pg_hc_b30000_r2e-2_nortgbase_HalfCheetah-v2_29-09-2019_19-42-05-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="no_rtg_baseline")
    a = np.loadtxt("run_pg_hc_b30000_r2e-2_rtgnobase_HalfCheetah-v2_29-09-2019_19-34-13-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="rtg_no_baseline")
    a = np.loadtxt("run_pg_hc_b30000_r2e-2_rtgbase_HalfCheetah-v2_29-09-2019_19-34-37-tag-Eval_AverageReturn.csv",delimiter=",",skiprows=1)
    plt.plot(a[:,1], a[:,2], label="rtg_baseline")    
    plt.legend()
    plt.savefig('7_2.png')
    plt.close()
    
if __name__ == '__main__':
    main()