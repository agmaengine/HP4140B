# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% include
import pyvisa
import time
import numpy as np
import matplotlib.pyplot as plt

#%% estrablish remote instrument 

rm = pyvisa.ResourceManager()
# GPIB Address
addr = str(16)
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
print('initialize instruments')
tc.write('F2T1L3A1RA1H05I3')
print('initialized')
#%% sweep setup
def sweep_setup(vstart,vstop,step,rcV=1):
    #   vstart (volt)
    #   vstop (volt) must > vstart
    #   step (volt) must < vstop - vstart
    #   rcV (V/s) rate of change of voltage
    
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

print('setting sweep parameters')
sweep_setup(-10.1,10.1,0.1)
print('set')
#%% sweep
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
    # wait for the device to make the next measurements for 50 ms
    # time.sleep(0.02)

# record data in SI unit

#%% data processing

I = []
V = []
for i in range(len(IV)):
    # filter read data with no data
    if IV[i][0:2] == ' N':
        im1, im2 = IV[i].split(',')
        # split record data 
        I.append(float(im1[3:]))
        V.append(float(im2[1:]))

# turn I,V to numpy
        
I = np.array(I)
V = np.array(V)
print('data collected')
#%% plot
print('visualize data')
plt.plot(V,I*1e3,'-o')
plt.xlabel('Volt (V)')
plt.ylabel('Current (mA)')
plt.title('I-V characteristics')
print('done!')
#%% functions
