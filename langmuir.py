# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:28:05 2019

@author: Chatchai Sirithipvanich
"""

import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
from scipy import optimize as _optimize
from utils import plot_2d_df as _plot_2d_df
from scipy.signal import find_peaks_cwt as _find_peaks_cwt


class langmuir:
    '''cores functions'''
    def __init__(self,rawiv,prober,probel,Z):
        #avg the i of the same v
        self._iv = rawiv.groupby('V(V)').mean().reset_index()
        self._prober = prober
        self._probel = probel
        self._Z = Z
        
        self._iv_around_vf()
        
        
    def _butter_worth_smooth(self,a,n):    
        l = len(a)-n
        inter = _np.zeros(l)
        for i in range(l):
            inter[i] = _np.sum(a[i:i+n])/n
        return inter
    
    def _lowess_smooth(self,xy,frac = 0.1):
        xy_np = xy.to_numpy()
        xy_np = _lowess(xy_np[:,1],xy_np[:,0],frac)
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
            i = _np.array(i).T
            i = _pd.DataFrame(i)
            i = i.dropna()
            
        elif alg == 'lowess':
            frac = kwargs.get('frac', 0.05)
            i = self._lowess_smooth(xy, frac)
            i = _pd.DataFrame(i)

        return i
            
    def _get_floating_potential(self):
        # iv is the pandas data frame with column = ['V(V)', 'I(A)']
        iv = self._iv
        vf = _np.zeros([2,2])
        vf[0,:] = iv[iv.iloc[:,1] < 0].tail(1).to_numpy()
        vf[1,:] = iv[iv.iloc[:,1] > 0].head(1).to_numpy()
        return (vf[0,0]-vf[1,0])/(vf[0,1]-vf[1,1])*(-vf[0,1])+vf[0,0]
    
    def _estimate_plasma_potential(self):
        div = self._div
        v = div[0].to_numpy()
        di = div[1].to_numpy()
        index_peaks = _find_peaks_cwt(di, _np.arange(1,5))
        _plot_2d_df(div,yfactor = 1e6)
        print(index_peaks)
        _plt.plot(v[index_peaks],di[index_peaks]*1e6,'or')
        return (v[index_peaks[0]],di[index_peaks[0]])

    def _avg_differential(self,xy):
        x = xy.iloc[:,0].to_numpy()
        y = xy.iloc[:,1].to_numpy()
        dx = _np.diff(x)
        dy = _np.diff(y)
        dyx = dy/dx
        dyx = _np.array([x[:len(x)-1],dyx]).T
        return _pd.DataFrame(dyx)
    
    def _get_scale(self,xy):
        x_max = xy.iloc[:,0].max()
        x_min = xy.iloc[:,0].min()
        y_max = xy.iloc[:,1].max()
        y_min = xy.iloc[:,1].min()
        xscale = x_max - x_min
        yscale = y_max - y_min
        return (xscale , yscale)
    
    def _iv_around_vf(self):
        iv = self._iv
        try:
            vf = self._get_floating_potential()
            iv = iv[ (vf - 10 < iv['V(V)']) & (iv['V(V)']  < vf + 20) ]
            self._iv = iv
            #_plot_2d_df(self._iv,scale=True)
        except:
            print('cannot find vf')
        return 0
    
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
        div = self._div
        ddiv = self._ddiv
        iv1 = ddiv[0][ddiv[1]/ivys-1e-2 > 0].head(1).index.values
        iv2 = ddiv[0][ddiv[1]==ddiv[1].max()].index.values
        iv1 = int(iv1)+1
        iv2 = int(iv2)+1
        
        try:
            vp = self._estimate_plasma_potential()
            print(vp)
        except Exception as e: 
            print(e)
            print('cannot estimate vp')
        
        # region separator obtained
        # fit the linear to the first region
        # prepare data
        v = iv.iloc[:,0].to_numpy()
        i = iv.iloc[:,1].to_numpy()
        a = _np.vstack((v[:iv1],_np.ones(iv1)))
        
        iv3 = iv2 + int(len(v)*0.3)
        if iv3 < len(v):
            v = v[:iv3]
            i = i[:iv3]
            
        # fit the data and get the first region parameters
        delta1, ii0 = _np.linalg.lstsq(a.T,i[:iv1],rcond=None)[0]
        # the fitted linear 
        i_1st_region = delta1*v + ii0
        # inspection
        
        # fit the exponetial to the second region
        # prepare data
        ie = i - i_1st_region
        _np.warnings.filterwarnings('ignore')
        log_ie = _np.log(ie)
        offset = 0
        a = _np.vstack((v[iv1+offset:iv2],_np.ones(len(v[iv1+offset:iv2]))))
        # fit the data and get the second region parameters
        Teinverse, c1 = _np.linalg.lstsq(a.T,log_ie[iv1+offset:iv2],rcond=None)[0]
        # get the fitted curves
        #log_iefit = Teinverse*v + c1
        ie_2nd_region = _np.exp(Teinverse*v + c1)
        
        # fit the linear to the third region
        region_offset = int(len(v)*0.2)
        a = _np.vstack( (v[iv2+region_offset:], _np.ones(len(v[iv2+region_offset:])) ) )
        delta2, c2 = _np.linalg.lstsq(a.T,ie[iv2+region_offset:],rcond=None)[0]
        ie_3rd_region = delta2*v + c2
        
        vf = self._get_floating_potential() # V
        
        idx = _np.argwhere(_np.diff(_np.sign(ie_3rd_region - ie_2nd_region))).flatten()
        
        if len(idx) == 0:
            ie0 = ie[iv2]
            vp = vf
            print('cannot find intersection assume vp =%.3E ,ie0=%.3E'%(vp,ie0))
        else:
            ie0 = ie_3rd_region[idx[-1]]
            vp = v[idx[-1]]
            
        # poly fit the third region
        # instead of fit for i, fit for v instead
        # assume ie = sqrt(v_3rd_region)
        #p = _np.polyfit(ie[iv2+10:],v[iv2+10:],deg=2)
        #ie0 = p[1]/p[0]/2
        #vp = p[2]-ie0**2/p[0]
        #v_3rd_region = p[0]*(ie**2)+ p[1]*ie +p[2]
        #v_3rd_region_1st = p[1]*ie +p[2]
        #_plt.plot(v[iv2:],ie[iv2:])
        
        # parameters
        Te = 1/Teinverse # eV
        Ap = 2*_np.pi*prober*probel + 2*_np.pi*prober**2 # m-2
        ne = ie0*4/(Ap*e**1.5)*(_np.pi*me/8/Te)**0.5 # m-3
        ni = ii0*(Z*mp/Te)**0.5/(-0.61*e**1.5*Ap) # m-3
        
        # inspection plot and print processed values
        if plot:
            # the input
            _plt.plot(v,i*1e6)
            # the 1st region
            _plt.plot(v[:iv1],i_1st_region[:iv1]*1e6)
            # the 2nd region
            # find vp and plot only to vp
            ivp = _np.argwhere(_np.abs(v - vp)<0.1).flatten()[-1]
            _plt.plot(v[iv1:ivp],(ie_2nd_region[iv1:ivp]+i_1st_region[iv1:ivp])*1e6)
            # the 3rd region
            _plt.plot(v[ivp:],ie_3rd_region[ivp:]*1e6)
            #_plt.plot(v_3rd_region[iv2-10:],(ie[iv2-10:]+i_1st_region[iv2-10:])*1e6)
            #_plt.plot(v_3rd_region_1st[iv2-10:],(ie[iv2-10:]+i_1st_region[iv2-10:])*1e6)
            _plt.ylabel('I (uA)')
            _plt.xlabel('V (V)')
            #_plt.title('fitted model')
            _plt.grid()
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
        div = self._div
        ddiv = self._ddiv
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
            i1 = x[0]*_np.exp((v[v<x[1]]-x[1])/x[2])
            # v > vp
            i2 = x[5]*_np.sqrt(v[v>=x[1]]-x[1]-x[6])+x[0]-x[7]
            i = _np.hstack((i1,i2)) + ii
            return i
        
        def fit_function2(x,v):
            # v < vp
            return x[0]*_np.exp((v-x[1])/x[2])+x[3]+x[4]*v
        
        def ie_residue1(x):
            return i-fit_function1(x,v)
        
        def ie_residue2(x):
            return i[:iv2]-fit_function2(x,v[:iv2])
        
        Te, ne, ni, ii0, ie0, vp, vf = \
            self._classical_analysis(iv,prober,probel,Z,plot=False)
        print(vp)
        #x =[ie0,vp,Te,ii0,0,1e-3,0,0]
        x    = [ie0,        vp,                     Te,         ii0,        0] 
        xmin = [0,          vf,                     0,          -_np.inf,    0]
        xmax = [_np.inf,    iv2 + int(len(v)*0.3),  _np.inf,    0,          1]
        residue = _optimize.least_squares(ie_residue2, x, bounds = (xmin,xmax))
        
        # parameters
        ie0 = residue.x[0]
        vp = residue.x[1]
        Te = residue.x[2]
        ii0 = residue.x[3]
        Ap = 2*_np.pi*prober*probel + 2*_np.pi*prober**2 # m-2
        ne = ie0*4/(Ap*e**1.5)*(_np.pi*me/8/Te)**0.5 # m-3
        ni = ii0*(Z*mp/Te)**0.5/(-0.61*e**1.5*Ap) # m-3
        
        if plot:
            _plt.plot(v,i*1e6)
            _plt.plot(v[:iv2+15],fit_function2(residue.x,v[:iv2+15])*1e6)
            _plt.xlabel('V (V)')
            _plt.ylabel('I (uA)')
            _plt.grid()
            print('floating potential = %.3f V'%vf)
            print('plasma potential = %.3f V'%vp)
            print('ion saturated current = %.3f uA'%(abs(ii0)*1e6))
            print('electron saturate current = %.3f uA'%(abs(ie0)*1e6))
            print('electron temperature = %.3f eV'%Te)
            print('electron density = %.3E cm-3'%(ne*1e-6))
            print('ion density = %.3E cm-3'%(ni*1e-6))
            
        return Te,ne,ni,ii0,ie0,vp,vf

    '''wraper'''
    def smoothening(self, **kwargs):
        
        '''
        

        Parameters
        ----------
        xy : iv dataframe
            DESCRIPTION.
        alg : string
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
        xy = self._iv
        sw = kwargs.get('alg', None)
        if sw == None:
            plt_title = 'not smoothen'
            intermediate = xy
        elif sw == 'bw':
            nn = kwargs.get('n', 10)
            loop = kwargs.get('iteration', 3)
            plt_title = 'butterworth smoothen iv n=%d consecutively for %d times'%(nn,loop)
            intermediate = self._smoothxy(xy, alg = sw, n = nn)
            for l in range(loop-1):
                intermediate = self._smoothxy(intermediate, alg = sw, n= nn)
        elif sw == 'lowess':
            fracc = kwargs.get('frac',0.05)
            plt_title = 'lowess smoothen iv frac=%.2f'%fracc
            intermediate = self._smoothxy(xy, alg = sw, frac = fracc)
        #plot
        plot = kwargs.get('plot',True)
        if plot:
            _plt.title(plt_title)
            _plot_2d_df(intermediate, scale =True)
            dyx = self._avg_differential(intermediate)
            _plot_2d_df(dyx, scale = True)
            ddyx = self._avg_differential(dyx)
            _plot_2d_df(ddyx, scale = True)
            _plt.xlabel('V (V)')
            _plt.ylabel('arbitary unit')
            _plt.legend(['iv', '1st derivative', '2nd derivative'])
        mutate = kwargs.get('mutate',True)
        if mutate:
            self._iv = intermediate
            self._div = self._avg_differential(self._iv)
            self._ddiv = self._avg_differential(self._div)
        
        return intermediate
    
    def get_iv(self):
        '''
        return iv dataframe

        Returns
        -------
        pd.Dataframe
            iv characteristic dataframe when generating class.

        '''
        return self._iv
    
    def get_probe_properties(self):
        '''
        return the langmuir probe properties
        the radius and length

        Returns
        -------
        2-tuple
            (probe radius, probe length)


        '''
        return (self._prober, self._probel)
    
    def get_gas_au(self):
        '''
        return the gas mass in atomic unit (au)
        
        Returns
        -------
        int
            the gas mass in au
        '''
        return self._Z
    
    
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
        try:
            if method == 'nonlinear':
                Te,ne,ni,ii0,ie0,vp,vf =self._nonlinear_analysis(self._iv, \
                        self._prober, self._probel, self._Z, plot = plot)
                
            elif method == 'classic':
                Te,ne,ni,ii0,ie0,vp,vf =self._classical_analysis(self._iv, \
                        self._prober, self._probel, self._Z, plot = plot)
            
            self._Te = Te
            self._ne = ne
            self._ni = ni
            self._ii0 = ii0
            self._ie0 = ie0
            self._vp = vp
            self._vf = vf
            return (Te,ne,ni,ii0,ie0,vp,vf)
        except Exception as e: print(e)
                #print('The input iv is bad or not a langmuir characteristics')
            
        
    def get_plasma_parameters(self):
        return (self._Te,self._ne,self._ni,self._ii0,self._ie0,self._vp,self._vf)
        