# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:28:05 2019

@author: Chatchai Sirithipvanich
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import scipy as sp
from utils import plot_2d_df


class langmuir:
    '''cores functions'''
    def __init__(self,rawiv,prober,probel,Z):
        #avg the i of the same v
        self.iv = rawiv.groupby('V(V)').mean().reset_index()
        self.prober = prober
        self.probel = probel
        self.Z = Z
    
    def _butter_worth_smooth(self,a,n):    
        l = len(a)-n
        inter = np.zeros(l)
        for i in range(l):
            inter[i] = np.sum(a[i:i+n])/n
        return inter
    
    def _lowess_smooth(self,xy,frac = 0.1):
        xy_np = xy.to_numpy()
        xy_np = lowess(xy_np[:,1],xy_np[:,0],frac)
        return xy_np
    
    def _smoothxy(self,xy, **kwargs):
        alg = kwargs.get('alg', 'bw')
        if  alg == 'bw':
            n = kwargs.get('n', 1)
            cols = xy.columns
            i = []
            for col in cols:
                a = xy[col].to_numpy()
                i.append(self._butter_worth_smooth(a,n))
            i = np.array(i).T
            i = pd.DataFrame(i)
            i = i.dropna()
            
        elif alg == 'lowess':
            frac = kwargs.get('frac', 0.05)
            i = self._lowess_smooth(xy, frac)
            i = pd.DataFrame(i)

        return i
            
    def _get_floating_potential(self,iv):
        # iv is the pandas data frame with column = ['V(V)', 'I(A)']
        vf = np.zeros([2,2])
        vf[0,:] = iv[iv.iloc[:,1] < 0].tail(1).to_numpy()
        vf[1,:] = iv[iv.iloc[:,1] > 0].head(1).to_numpy()
        return (vf[0,0]-vf[1,0])/(vf[0,1]-vf[1,1])*(-vf[0,1])+vf[0,0]
    
    def _avg_differential(self,xy):
        x = xy.iloc[:,0].to_numpy()
        y = xy.iloc[:,1].to_numpy()
        dx = np.diff(x)
        dy = np.diff(y)
        dyx = dy/dx
        dyx = np.array([x[:len(x)-1],dyx]).T
        return pd.DataFrame(dyx)
    
    def _get_scale(self,xy):
        x_max = xy.iloc[:,0].max()
        x_min = xy.iloc[:,0].min()
        y_max = xy.iloc[:,1].max()
        y_min = xy.iloc[:,1].min()
        xscale = x_max - x_min
        yscale = y_max - y_min
        return (xscale , yscale)
     
    '''algorithm 1'''
    def _classical_analysis(self,iv,prober,probel,Z,plot = True):
        '''
            determine the regions separators
            # first point
            # determined by second ddiv < 1e-3
            # second point
            # determined by the maximum of div
            the first region used for determining the Ii0
            the middle region used for determining the temperatures and Ie0
            the vp must be determined first before determine Ie0
            the vp is the intersection of fitted graph between region 2 and 3
            
            electron density
            ie0 = n*e^1.5*Ap(1/4)(8*(k*Te/e)/pi/me)^0.5
            ne = ie0*(pi/me/8*(k*Te/e))^0.5 /(e^1.5 *Ap(1/4))
            ii0 = -0.61*e^1.5*Ap*((k*Te/e)/Z*mi)^0.5
            ni = ii0*(Z*mp/(k*Te/e))^0.5/(-0.61*e^1.5*Ap)
        '''
        # constants
        mp = 1.67262158e-27 #SI kg
        me = 9.10938188e-31 #SI kg
        e = 1.60217733e-19 #SI C
        # determining the region separators
        ivxs, ivys = self._get_scale(iv)
        div = self._avg_differential(iv)
        ddiv = self._avg_differential(div)
        iv1 = ddiv[0][ddiv[1]/ivys-1e-2 > 0].head(1).index.values
        iv2 = ddiv[0][ddiv[1]==ddiv[1].max()].index.values
        iv1 = int(iv1)+1
        iv2 = int(iv2)+1
        # region separator obtained
        # fit the linear to the first region
        # prepare data
        v = iv.iloc[:,0].to_numpy()
        i = iv.iloc[:,1].to_numpy()
        a = np.vstack((v[:iv1],np.ones(iv1)))
        # fit the data and get the first region parameters
        delta1, ii0 = np.linalg.lstsq(a.T,i[:iv1],rcond=None)[0]
        # the fitted linear 
        ii = delta1*v + ii0
        # inspection
        
        # fit the exponetial to the second region
        # prepare data
        ie = i - ii
        log_ie = np.log(ie)
        offset = 0
        a = np.vstack((v[iv1+offset:iv2],np.ones(len(v[iv1+offset:iv2]))))
        # fit the data and get the second region parameters
        Teinverse, c1 = np.linalg.lstsq(a.T,log_ie[iv1+offset:iv2],rcond=None)[0]
        # get the fitted curves
        #log_iefit = Teinverse*v + c1
        iefit = np.exp(Teinverse*v + c1)
        
        # fit the linear to the third region
        #a = np.vstack( (v[iv2:], np.ones(len(v[iv2:])) ) )
        #delta2, c2 = np.linalg.lstsq(a.T,ie[iv2:],rcond=None)[0]
        #iesat = delta2*v + c2
        
        # poly fit the third region
        p = np.polyfit(ie[iv2+10:],v[iv2+10:],deg=2)
        ie0 = p[1]/p[0]/2
        vp = p[2]-ie0**2/p[0]
        vesat = p[0]*(ie**2)+ p[1]*ie +p[2]
        #plt.plot(v[iv2:],ie[iv2:])
        
        # parameters
        Te = 1/Teinverse # eV
        vf = self._get_floating_potential(iv) # V
        Ap = 2*np.pi*prober*probel + 2*np.pi*prober**2 # m-2
        ne = ie0*4/(Ap*e**1.5)*(np.pi*me/8/Te)**0.5 # m-3
        ni = ii0*(Z*mp/Te)**0.5/(-0.61*e**1.5*Ap) # m-3
        
        # inspection plot and print processed values
        if plot:
            plt.plot(v,i*1e6)
            plt.plot(v[:iv1],ii[:iv1]*1e6)
            plt.plot(v[iv1:iv2],(iefit[iv1:iv2]+ii[iv1:iv2])*1e6)
            plt.plot(vesat[iv2-10:],(ie[iv2-10:]+ii[iv2-10:])*1e6)
            plt.ylabel('I (uA)')
            plt.xlabel('V (V)')
            plt.title('fitted model')
            plt.grid()
            print('floating potential = %.3f V'%vf)
            print('plasma potential = %.3f V'%vp)
            print('ion saturated current = %.3f uA'%(abs(ii0)*1e6))
            print('electron saturate current = %.3f uA'%(abs(ie0)*1e6))
            print('electron temperature = %.3f eV'%Te)
            print('electron density = %.3E cm-3'%(ne*1e-6))
            print('ion density = %.3E cm-3'%(ni*1e-6))
            
        return Te,ne,ni,ii0,ie0,vp,vf
    
    '''algorithm 2'''
    def _nonlinear_analysis(self,iv,prober,probel,Z,plot = True):
        '''
            determine the regions separators
            # first point
            # determined by second ddiv < 1e-3
            # second point
            # determined by the maximum of div
            the first region used for determining the Ii0
            the middle region used for determining the temperatures and Ie0
            the vp must be determined first before determine Ie0
            the vp is the intersection of fitted graph between region 2 and 3
            
            electron density
            ie0 = n*e^1.5*Ap(1/4)(8*(k*Te/e)/pi/me)^0.5
            ne = ie0*(pi/me/8*(k*Te/e))^0.5 /(e^1.5 *Ap(1/4))
            ii0 = -0.61*e^1.5*Ap*((k*Te/e)/Z*mi)^0.5
            ni = ii0*(Z*mp/(k*Te/e))^0.5/(-0.61*e^1.5*Ap)
        '''
        # constants
        mp = 1.67262158e-27 #SI kg
        me = 9.10938188e-31 #SI kg
        e = 1.60217733e-19 #SI C
        # determining the region separators
        ivxs, ivys = self._get_scale(iv)
        div = self._avg_differential(iv)
        ddiv = self._avg_differential(div)
        iv1 = ddiv[0][ddiv[1]/ivys-1e-2 > 0].head(1).index.values
        iv2 = ddiv[0][ddiv[1]==ddiv[1].max()].index.values
        iv1 = int(iv1)+1
        iv2 = int(iv2)+1
        # prepare data
        v = iv.iloc[:,0].to_numpy()
        i = iv.iloc[:,1].to_numpy()
        def fit_function1(x,v):
            ii = x[3]+x[4]*v
            # v < vp
            i1 = x[0]*np.exp((v[v<x[1]]-x[1])/x[2])
            # v > vp
            i2 = x[5]*np.sqrt(v[v>=x[1]]-x[1]-x[6])+x[0]-x[7]
            i = np.hstack((i1,i2)) + ii
            return i
        
        def fit_function2(x,v):
            # v < vp
            return x[0]*np.exp((v-x[1])/x[2])+x[3]+x[4]*v
        
        def ie_residue1(x):
            return i-fit_function1(x,v)
        
        def ie_residue2(x):
            return i[:iv2]-fit_function2(x,v[:iv2])
        
        Te, ne, ni, ii0, ie0, vp, vf = \
            self._classical_analysis(iv,prober,probel,Z,plot=False)
        #x1 =[ie0,vp,Te,ii0,0,1e-3,0,0]
        x2 =[ie0,vp,Te,ii0,0] 
        residue = sp.optimize.least_squares(ie_residue2,x2)
        
        # parameters
        ie0 = residue.x[0]
        vp = residue.x[1]
        Te = residue.x[2]
        ii0 = residue.x[3]
        Ap = 2*np.pi*prober*probel + 2*np.pi*prober**2 # m-2
        ne = ie0*4/(Ap*e**1.5)*(np.pi*me/8/Te)**0.5 # m-3
        ni = ii0*(Z*mp/Te)**0.5/(-0.61*e**1.5*Ap) # m-3
        
        if plot:
            plt.plot(v,i)
            plt.plot(v[:iv2],fit_function2(residue.x,v[:iv2]))
            print('floating potential = %.3f V'%vf)
            print('plasma potential = %.3f V'%vp)
            print('ion saturated current = %.3f uA'%(abs(ii0)*1e6))
            print('electron saturate current = %.3f uA'%(abs(ie0)*1e6))
            print('electron temperature = %.3f eV'%Te)
            print('electron density = %.3E cm-3'%(ne*1e-6))
            print('ion density = %.3E cm-3'%(ni*1e-6))
            
        return Te,ne,ni,ii0,ie0,vp,vf

    '''wraper'''
    def smoothening(self,xy, **kwargs):
        
        '''
        

        Parameters
        ----------
        xy : iv dataframe
            DESCRIPTION.
        sw : string
            'bw' butterworth filter 
                additional options:
                    n           : number of data to be averaged
                    iteration   : number of iterations to be perform filtering
            'lowess' lowess smoothener
                additional options:
                    frac        : allow value (0,1) used for tuning the
                                smoothening model
        plot : BOOL
            visualized the smoothened data and its 1st and 2nd derivatives in
            yscale plot
        Returns
        -------
        i : iv dataframe
            the smoothened iv.

        '''
        xy = self.iv
        sw = kwargs.get('alg', None)
        if sw == None:
            plt_title = 'not smoothen'
            i = xy
        elif sw == 'bw':
            nn = kwargs.get('n', 2)
            loop = kwargs.get('iteration', 1)
            plt_title = 'butterworth smoothen iv n=%d consecutively for %d times'%(nn,loop)
            i = self._smoothxy(xy, alg = sw, n = nn)
            for l in range(loop-1):
                i = self._smoothxy(i,alg = sw, n= nn)
                print(l)
        elif sw == 'lowess':
            fracc = kwargs.get('frac',0.05)
            plt_title = 'lowess smoothen iv frac=%.2f'%fracc
            i = self._smoothxy(xy, alg = sw, frac = fracc)
        #plot
        plot = kwargs.get('plot',True)
        if plot:
            plt.figure()
            plt.title(plt_title)
            plot_2d_df(i, scale =True)
            dyx = self._avg_differential(i)
            plot_2d_df(dyx, scale = True)
            ddyx = self._avg_differential(dyx)
            plot_2d_df(ddyx, scale = True)
        return i
    def diagnostics(self,method = 'nonlinear',plot = True):
        '''
        

        Parameters
        ----------
        method : string, optional
            choose the analysis methods, there are 2 options including
            conventional method and nolinear fitting method. 
            The default is 'nonlinear'.
        plot : BOOL, optional
            visualized the fitted model to the (smoothend) iv. The default is True.

        Returns
        -------
        Te : float
            Electron Temperature.
        ne : float
            Electron Density.
        ni : float
            Ion Density.
        ii0 : float
            Ion Saturated Current.
        ie0 : float
            Electron Saturated Current.
        vp : float
            Plasma Potential.
        vf : float
            Floating Potential.

        '''
        if method == 'nonlinear':
            Te,ne,ni,ii0,ie0,vp,vf =self.nonlinear_analysis(self.iv, \
                    self.prober, self.probel, self.Z, plot = plot)
        elif method == 'classic':
            Te,ne,ni,ii0,ie0,vp,vf =self.classical_analysis(self.iv, \
                    self.prober, self.probel, self.Z, plot = plot)
            self.Te = Te
            self.ne = ne
            self.ni = ni
            self.ii0 = ii0
            self.ie0 = ie0
            self.vp = vp
            self.vf = vf
            return (Te,ne,ni,ii0,ie0,vp,vf)
        