# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:10:16 2019

@author: Chatchai Sirithipvanich
"""
import matplotlib.pyplot as _plt
import pandas as _pd
import os as _os

def get_system_directory(path):
    '''
    

    Parameters
    ----------
    path : string
        directory path either real path or relative path.

    Returns
    -------
    path : string
        system directory path.

    '''
    if _os.sys.platform == 'win32':
        path = _os.path.realpath(path)+'\\'
    else:
        path = _os.path.realpath(path)+'/'
    return path

def plot_2d_df(xy,xfactor = 1,yfactor = 1,scale = False):
    '''
    

    Parameters
    ----------
    xy : pandas.DataFrame
        pandas dataframe carry two columns of data.
    xfactor : float, optional
        muliplicative factor of the 1st column. The default is 1.
    yfactor : float, optional
        multplicative factor of the 2nd column. The default is 1.
    scale : BOOL, optional
        scale y axis to [0,1]. The default is False.

    Returns
    -------
    None.

    '''
    #x_max = xy.iloc[:,0].max()
    #x_min = xy.iloc[:,0].min()
    y_max = xy.iloc[:,1].max()
    y_min = xy.iloc[:,1].min()
    #xscale = x_max - x_min
    yscale = y_max - y_min
    if scale:
        #xfactor = 1/xscale
        yfactor = 1/yscale
    _plt.plot(xy.iloc[:,0].to_numpy()*xfactor,xy.iloc[:,1].to_numpy()*yfactor)
    
def load_data(path,namelist,plot=True):
    '''
    only csv is supported

    Parameters
    ----------
    path : string
        real or relative directory path of the data.
    namelist : string or list
        files name or list of file's names.
    plot : TYPE, optional
        trigger whether plot the loaded data. The default is True.

    Returns
    -------
    iv : pandas.DataFrame
        The dataframe of loaded data

    '''
    path = get_system_directory(path)
    iv=[]
    _plt.figure()
    for name in namelist:
        iv.append(_pd.read_csv(path+name+'.csv'))
        
    if plot:
        for df in iv:
            plot_2d_df(df,yfactor=1e6)
            
        _plt.xlabel('V (V)')
        _plt.ylabel('I (uA)')
        _plt.legend(namelist)
        _plt.grid()
    return iv