

import numpy as np

def jac2R(j0, j1, Nsample, l=0.1):
    R = np.zeros((Nsample, 6))
    for i in range(Nsample):
        R[i, 0] = -l * np.sin(j0[i]) - l * np.sin(j0[i] + j1[i])
        R[i, 1] = -l * np.sin(j0[i] + j1[i])
        R[i, 2] = l * np.cos(j0[i]) + l * np.cos(j0[i] + j1[i])
        R[i, 3] = l * np.cos(j0[i] + j1[i])
        R[i, 4] = 1
        R[i, 5] = 1
    return R

def jac3R(j0, j1, j2, Nsample, l=0.1):
    R = np.zeros((Nsample, 3, 3))
    for i in range(Nsample):
        R[i][0, 0] = -l * np.sin(j0[i]) - l * np.sin(j0[i] + j1[i]) - l * np.sin(j0[i] + j1[i] + j2[i])
        R[i][0, 1] = -l * np.sin(j0[i] + j1[i]) - l * np.sin(j0[i] + j1[i] + j2[i])
        R[i][0, 2] = -l * np.sin(j0[i] + j1[i] + j2[i])
        R[i][1, 0] = l * np.cos(j0[i]) + l * np.cos(j0[i] + j1[i]) + l * np.cos(j0[i] + j1[i] + j2[i])
        R[i][1, 1] = l * np.cos(j0[i] + j1[i]) + l * np.cos(j0[i] + j1[i] + j2[i])
        R[i][1, 2] = l * np.cos(j0[i] + j1[i] + j2[i])
        R[i][2, 0] = 1
        R[i][2, 1] = 1
        R[i][2, 2] = 1
    return R