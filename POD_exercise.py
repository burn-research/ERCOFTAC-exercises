#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:03:48 2022

@author: nct00021
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/gpfs/projects/nct00/nct00021/Data' # Path to the dataset 
dt = 0.00025
tmax = 0.25
timesteps = np.arange(0., tmax, step=dt) # Number of timesteps 

# We load the first timestep to get the spatial coordinates
df_f10_0 = pd.read_csv(f'{path}/output_f10_t0', sep='\s+')
xyz = np.array(df_f10_0[['x[m](2)', 'y[m](3)', 'z[m](4)']])

n = xyz.shape[0] # Number of rows (cells)

X_f10 = np.load(f'{path}/X_f10_CO2.npy')
m = X_f10.shape[1] # Number of columns (timesteps)


#plt.scatter(xyz[:,0], xyz[:,2], c=X_f10[:,0])
#plt.show()

U, S, Vt = np.linalg.svd(X_f10, full_matrices=False)
V = Vt.T

np.save('./U.npy', U)
np.save('./V.npy', V)
np.save('./S.npy', S)

q = 16
A = np.diag(S)[:, :q] @ V[:,:q]

observation = X_f10[:,-1]
a = A[:,-1]
reconstruction = U[:,:q] @ a

plt.scatter(observation, reconstruction)
plt.show()
 


