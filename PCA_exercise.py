import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/gpfs/projects/nct00/nct00021/Data'

df = pd.read_csv(f'{path}/output_f10_t0', sep='\s+')

xyz = np.array(df[['x[m](2)', 'y[m](3)', 'z[m](4)']]) # x, y, z coordinates

# We select the features that we want to analyse
features = ['T[k](11)', 'H2(12)', 'H(13)', 'O(14)', 'O2(15)', 'OH(16)', 'H2O(17)', 
            'HO2(18)', 'H2O2(19)', 'CH2(22)', 'CH3(24)', 
            'CH4(25)', 'CO(26)', 'CO2(27)', 'CH2O(29)', 
            'CH3OH(32)', 'C2H(33)', 'C2H2(34)', 'C2H3(35)', 'C2H4(36)', 
            'C2H5(37)', 'C2H6(38)', 'HCCO(39)', 'CH2CO(40)', 'N2(42)', 
            'C3H7(44)', 'C3H8(45)', 'CH2CHO(46)', 'CH3CHO(47)']

feature = 'O2(15)'
f = np.array(df[feature])

#fig, ax = plt.subplots()
#ax.scatter(xyz[:,0], xyz[:,2], c=f)
#plt.show()

X = np.array(df[features])
n = X.shape[0]
m = X.shape[1]

# Scale the dataset

X0 = np.zeros_like(X)

mean = np.mean(X, axis=0)
#s = np.std(X, axis=0)
s = np.max(X, axis=0) - np.min(X, axis=0)

for i in range(m):
	X0[:,i] = (X[:,i] - mean[i])/s[i]

S = 1/(n-1)* X0.T @ X0

l, A = np.linalg.eig(S)

#plt.plot(l)
#plt.show()

#plt.bar(np.linspace(1, m, m), A[:,0])
#plt.show()

Z = X0 @ A

#plt.scatter(xyz[:,0], xyz[:,2], c=Z[:,5])
#plt.show()

q = 10

X0_r = Z[:,:q]@A[:,:q].T

plt.scatter(X0.flatten(), X0_r.flatten())
plt.show()














