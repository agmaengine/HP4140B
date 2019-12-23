# -*- coding: utf-8 -*-
"""
Tracer class

Authors:
    Chatchai Sirithipvanich

Last edited date:
    20/12/2019 15.46
"""
import pyvisa
import datetime
import numpy as _np
import matplotlib.pyplot as _plt
import pandas as _pd
from utils import get_system_directory as _get_system_directory

class tracer:
    def __init__(self, address, **kwargs):
        ''' estrablish remote instrument '''
        rm = pyvisa.ResourceManager()
        # GPIB Address
        self.addr = str(address)
        # Create instrument object GPIB Protocol
        self._tc = rm.open_resource('GPIB0::'+self.addr+'::INSTR')
        self._vstart = -5
        self._vstop = 5
        self._step = 0.1
        self._delay = 0.5
        spath = kwargs.get('path','./')
        self.savepath = _get_system_directory(spath)
        self.i_mode_sweep(self._vstart,self._vstop,self._step,self._delay)
        
    ''' initialize instrument '''
    def _initialize(self):
        '''
        function description
        set trigger to internal trigger 
        set VA voltage limit to 10-2 A
        set VA mode to only increasing continuously
        set Range to auto
        set auto range limit to 10-5 A
        set integration time to long
        '''
        print('initialize instruments')
        self._tc.write('T1L3A1RA1H05I3')
        print('initialized')
    
    '''sweep modes'''
    def iv_mode_sweep(self,vstart, vstop, step, rcV=1):
        '''
        parameters
        ----------
            vstart  initial sweep voltage voltage (volt)
                    allow range [-100,100]
            vstop   final sweep voltage (volt) must > vstart
                    allow range [-100,100]
            step    voltage step (volt) must < vstop - vstart
                    allow range [-100,100]
            rcV     rate of change of the sweeping voltage (V/s)
                    allow range (0,1]
        function description
        --------------------
            set the tracer sweep mode as iv mode, the tracer proceed
            to the next step when the current at the step is measured.
            use this mode to accquire current for each step of voltage.
        '''
        self._vstart = vstart
        self._vstop = vstop
        self._step = step
        self._delay = 0
        self._rcV = rcV
        
        self._initialize()
        self._tc.write('F2')
        # set VA to 0V
        self._tc.write('PA0')
        # set VA start value = vstart
        self._tc.write('PS%.1f'%vstart)
        # set VA stop value = vstop
        self._tc.write('PT%.1f'%vstop)
        # set voltage change step to step
        self._tc.write('PE%.1f'%step)
        # set step dV/dt to rcV
        self._tc.write('PV%.1f'%rcV)
        # set step delay time 0V
        self._tc.write('PD0')
        print('The tracer sweep mode is set to IV, \
              voltage starting from %.1f to %.1f with %.1f V step' \
              %(vstart, vstop, step))

    def i_mode_sweep(self,vstart, vstop, step, delay=0.5):
        '''
        parameters
        ----------
            vstart  initial sweep voltage voltage (volt)
                    allow range [-100,100]
            vstop   final sweep voltage (volt) must > vstart
                    allow range [-100,100]
            step    voltage step (volt) must < vstop - vstart
                    allow range [-100,100]
            delay   delay time before proceed to the next voltage step (s)
                    allow range (0,10]
        function description
        --------------------
            set the tracer sweep mode as i mode, the tracer wait for delay time
            then proceed to the next step wheather the current is measured or not. 
            use this mode for maximum accquiring speed but current for some steps
            will be missing!
        '''
        self._vstart = vstart
        self._vstop = vstop
        self._step = step
        self._delay = delay
        self._rcV = 0
        
        self._initialize()
        # set mode to i mode
        self._tc.write('F1')
        # set VA = 0 V
        self._tc.write('PA0')
        # set VA start value = vstart
        self._tc.write('PS%.1f'%vstart)
        # set VA stop value = vstop
        self._tc.write('PT%.1f'%vstop)
        # set voltage change step to step
        self._tc.write('PE%.1f'%step)
        # set step delay time delay
        self._tc.write('PD%.1f'%delay)
        # set step dV/dt to 0V/s
        self._tc.write('PV0')

        print('The tracer sweep mode is set to I, \
              voltage starting from %.1f to %.1f with %.1f V step' \
              %(vstart, vstop, step))
    
    def sweep(self,plot = True, save = True, **kwargs):
        '''
        Parameters
        ----------
        plot : BOOL, optional
            trigger for collected data visualizing. The default is True.
        save : BOOL, optional
            trigger for saving the collected data. The default is True.
            when the save is triggered true, path should be specify
            otherwise, save the data to the present working directory.
        Returns
        -------
        pandas.DataFrame
            return dataframe containing collected voltage and current with
            columns = ['V(V)','I(A)'].
        '''
        
        self.t = datetime.datetime.now()
        timestr = str(self.t.year)+'%02d'%self.t.month+ \
            '%02d'%self.t.day+'_'+'%02d'%self.t.hour+'%02d'%self.t.minute
        print('collecting data on '+self.t.ctime())
        self._tc.write('W1')
        IV = []
        I = []
        V = []
        trigger = True
        while trigger:
            # read the measurement
            try:
                IV.append(self._tc.read_bytes(23).decode())
            except:
                trigger = False
            
            if IV[-1][0:2] == ' N':
                im1, im2 = IV[-1].split(',')
                # split record data 
                I.append(float(im1[3:]))
                V.append(float(im2[1:]))
            
            if V[-1] == float(self._vstop):
                trigger =False
        # end while
        
        # wait for the device to make the next measurements for 50 ms
        # time.sleep(0.02)
        # record data in SI unit
        # turn I,V to numpy
        I = _np.array(I)
        V = _np.array(V)
        self._iv = _pd.DataFrame(_np.array([V,I]).T,columns=['V(V)','I(A)'])
        
        ('data collected')
        if plot:
            print('visualize data')
            _plt.plot(V,I*1e6,'-o')
            _plt.xlabel('Volt (V)')
            _plt.ylabel('Current (uA)')
            _plt.title('I-V characteristics')
            _plt.grid()
            print('done!')
        if save:
            self.savepath = kwargs.get('path',self.savepath)
            self.savepath = _get_system_directory(self.savepath)
            self._iv.to_csv( self.savepath + timestr + '.csv',index=False)
        return self._iv
    
    def get_iv(self):
        return self._iv