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

U = np.load('./U.npy')
V = np.load('./V.npy')
S = np.load('./S.npy')

#plt.scatter(xyz[:,0], xyz[:,2], c=U[:,6])
#plt.show()

#plt.plot(V[:,6])
#plt.show()

X_f10 = np.load(f'{path}/X_f10_CO2.npy')

q = 4
A = np.diag(S[:q]) @ V[:,:q].T

observation = X_f10[:,-1]
a = A[:,-1]
reconstruction = U[:,:q] @ a

plt.scatter(observation, reconstruction)
plt.show()

fig, ax = plt.subplots()
ax.scatter(xyz[:,0], xyz[:,2], c=observation)
ax.scatter(-xyz[:,0], xyz[:,2], c=reconstruction)
plt.show()



