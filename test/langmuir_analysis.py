# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 12:53:39 2019

@author: Chatchai
"""
#%%
import os
import HP4140B
import matplotlib.pyplot as plt
from utils import load_data
# from utils import plot_2d_df
#%%
'''
data_path = './HP4140B/data'
filename_list = os.listdir(data_path)
iv_list = load_data(data_path,filename_list,plot=False)
title = filename_list[:]

for iv in iv_list:
    plt.cla()
    plt.title(title.pop(0))
    plot_2d_df(iv)
    plt.pause(0.5)
    #input("Press Enter to continue...")
'''  

#%%
data_path = './HP4140B/data'
filename_list = os.listdir(data_path)
iv_list = load_data(data_path,filename_list,plot=False)
title = filename_list[:]

# testing
# iv_list = iv_list[:5]
# title = title[:5]

fig = plt.figure(figsize = (20,6))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
for iv in iv_list:
    
    lm = HP4140B.langmuir(iv,1.6e-3,6e-3,18)
    
    plt.sca(ax1)
    lm.smoothening(alg = 'bw',mutate = True)
    
    plt.sca(ax2)
    plt.title(title.pop(0))
    lm.diagnostics(method='classic')
    #plt.tight_layout()
    plt.draw()
    plt.waitforbuttonpress(timeout = 10)
    plt.cla()
    plt.sca(ax1)
    plt.cla()

plt.close()