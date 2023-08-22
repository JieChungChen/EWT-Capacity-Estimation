from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os


def open_mat():
    filepath = "c1/V4.mat"
    f = loadmat(filepath)
    c = f['I_full_vec_M1_NMC25degC']
    v = f['V_full_vec_M1_NMC25degC']
    step = f['Step_Index_full_vec_M1_NMC25degC']
    plt.plot(c, c='black')
    plt.show()
    plt.close()


def main():
    open_mat()


if __name__=="__main__":
    main()