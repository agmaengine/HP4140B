# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:23:48 2019

@author: user
"""
#%% import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%%
def pdIVplot(iv):
    plt.plot(iv['V(V)'].to_numpy(),iv['I(mA)'].to_numpy()*1e6)

#%% loads data
Exp = pd.read_excel('../../LangmuirExp/2019-12-13/TraceMeta.xlsx',index_col=0)

# sorted respected to Vr with the first at the end to pop it out
# when loop initiate
namelist = Exp['IV_Filename'][[4,1,2,3,5]].to_list()

Vr=-100
iv = pd.read_csv('../../DataLogs/'+namelist.pop()+'.csv')
iv['V(V)']=iv['V(V)']+Vr
fig = plt.figure()
pdIVplot(iv)
Vr=Vr+50

for name in namelist:
    Inter = pd.read_csv('../../DataLogs/'+name+'.csv')
    Inter['V(V)']=Inter['V(V)']+Vr
    pdIVplot(Inter)
    iv = iv.append(Inter)
    Vr = Vr + 50

x = np.arange(-200,200)
plt.plot(x,np.zeros(len(x)))
plt.xlabel('V (V)')
plt.ylabel('I (nA)')
plt.grid()
#plt.plot(V,I)
#%%
iv = iv.sort_values(['V(V)'])
V = iv['V(V)'].to_numpy()
I = iv['I(mA)'].to_numpy()
plt.plot(V,I)