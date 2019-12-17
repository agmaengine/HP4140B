# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:35:52 2019

@author: user
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%%
def pdIVplot(iv):
    plt.plot(iv['V(V)'].to_numpy(),iv['I(A)'].to_numpy()*1e6)

#%% loads data
namelist = ['20191217_1820','20191217_1833']

iv=[]
iv.append(pd.read_csv('../../DataLogs/'+namelist.pop()+'.csv'))
fig = plt.figure()
pdIVplot(iv[0])

for name in namelist:
    iv.append(pd.read_csv('../../DataLogs/'+name+'.csv'))
    pdIVplot(iv[-1])
    
x = np.arange(-200,200)
plt.plot(x,np.zeros(len(x)))
plt.xlabel('V (V)')
plt.ylabel('I (uA)')
plt.grid()
#plt.plot(V,I)
#%%
iv = iv.sort_values(['V(V)'])
V = iv['V(V)'].to_numpy()
I = iv['I(mA)'].to_numpy()
plt.plot(V,I)