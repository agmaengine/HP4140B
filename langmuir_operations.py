# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 14:40:18 2019

@author: user
"""

import HP4140B
tc = HP4140B.tracer(9, path = 'C:\\Users\\user\\Documents\\DataLogs\\')
tc.i_mode_sweep(-10,85,0.2,0.2)
iv = tc.sweep()
#lm = HP4140B.langmuir(iv, 1.6e-3, 6e-3, 18)
#lm.smoothening(sw = 'bw', n = 20, iteration = 3, mutate = True)
#parameters = lm.diagnostics()
