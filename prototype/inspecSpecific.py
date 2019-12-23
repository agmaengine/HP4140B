# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:35:52 2019

@author: Chatchai Sirithipvanich
"""


import matplotlib.pyplot as plt
import pandas as pd
from utils import plot_2d_df
from utils import get_system_directory

#%% initial parameters
DataDir = 'D:\OneDrive\MinWork\DataLogs'
namelist = ['20191217_2015','20191217_2030','20191217_2054']

#%% loads data
path = get_system_directory(DataDir)
iv=[]
iv.append(pd.read_csv(path+namelist.pop()+'.csv'))
fig = plt.figure()
plot_2d_df(iv[0])

for name in namelist:
    iv.append(pd.read_csv(path+name+'.csv'))
    plot_2d_df(iv[-1],yfactor=1e6)
    
#x = np.arange(-200,200)
#plt.plot(x,np.zeros(len(x)))
plt.xlabel('V (V)')
plt.ylabel('I (uA)')
plt.grid()
#plt.plot(V,I)
#%%
iv = iv.sort_values(['V(V)'])
V = iv['V(V)'].to_numpy()
I = iv['I(mA)'].to_numpy()
plt.plot(V,I)