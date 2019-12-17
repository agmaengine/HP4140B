# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% include
import pyvisa
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% estrablish remote instrument 

rm = pyvisa.ResourceManager()
# GPIB Address
addr = str(9)
# Create instrument object GPIB Protocol
tc = rm.open_resource('GPIB0::'+addr+'::INSTR')

# send command
# tc.write(cmd)
# read
# tc.read(cmd)
# query
# tc.query(cmd)

#%% initialize instrument

# set function to I-V
# set trigger to internal trigger 
# set VA voltage limit to 10-2 A
# set VA mode to only increasing continuously
# set Range to auto
# set auto range limit to 10-5 A
# set integration time to long
def initialize():
    print('initialize instruments')
    tc.write('T1L3A1RA1H05I3')
    print('initialized')
#%% sweep setup
def continuous_sweep_setup(vstart,vstop,step,rcV=1):
    #   vstart (volt)
    #   vstop (volt) must > vstart
    #   step (volt) must < vstop - vstart
    #   rcV (V/s) rate of change of voltage
    tc.write('F2')
    # set VA to 0V
    tc.write('PA0')
    # set VA start value = -5V
    tc.write('PS%.1f'%vstart)
    # set VA stop value = 5V
    tc.write('PT%.1f'%vstop)
    # set voltage change step to 1V
    tc.write('PE%.1f'%step)
    # set step dV/dt to 1V/s
    tc.write('PV%.1f'%rcV)

def highres_sweep_setup(vstart,vstop,step):
    tc.write('F1')
    tc.write('PA0')
    # set VA start value = -5V
    tc.write('PS%.1f'%vstart)
    # set VA stop value = 5V
    tc.write('PT%.1f'%vstop)
    # set voltage change step to 1V
    tc.write('PE%.1f'%step)
    # set hold time 0.2sec
    #tc.write('PH0.1')
    # set step delay time 0.15s
    tc.write('PD0.5')
    # set step dV/dt to 0V/s
    tc.write('PV0')

print('setting sweep parameters')
initialize()
#continuous_sweep_setup(-100,100,0.1)
Vstart = 0
Vstop = 25
highres_sweep_setup(Vstart,Vstop,0.1)
print('set')
#%% sweep
I = []
V = []
print('collecting data')
tc.write('W1')
trigger = True
IV = []
while trigger:
    # read the measurement
    try:
        IV.append(tc.read_bytes(23).decode())
    except:
        trigger = False
    if IV[-1][0:2] == ' N':
        im1, im2 = IV[-1].split(',')
        # split record data 
        I.append(float(im1[3:]))
        V.append(float(im2[1:]))
    if V[-1] == float(Vstop):
        trigger =False
    # wait for the device to make the next measurements for 50 ms
    # time.sleep(0.02)

# record data in SI unit
# turn I,V to numpy
        
I = np.array(I)
V = np.array(V)
print('data collected')
#plot
print('visualize data')
plt.plot(V,I*1e6,'-o')
plt.xlabel('Volt (V)')
plt.ylabel('Current (uA)')
plt.title('I-V characteristics')
print('done!')
#%% save
VI = pd.DataFrame(np.array([V,I]).T,columns=['V(V)','I(A)'])
t = datetime.datetime.now()
timestr = str(t.year)+'%02d'%t.month+'%02d'%t.day+'_'+'%02d'%t.hour+'%02d'%t.minute
VI.to_csv('../../DataLogs/'+timestr+'.csv',index=False)
print(timestr)

#%% functions
