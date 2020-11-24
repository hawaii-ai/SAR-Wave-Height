#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import glob
import h5py
import pandas as pd
from tqdm import tqdm

import preprocess
import importlib
importlib.reload(preprocess)


# In[33]:


# Create new h5 file with following:
# 1) Data separated by sat year month. No need to combine because this is for prediction.
# 2) Features scaled.
#groups = {'2015_2016':[2015, 2016], '2017':[2017], '2018':[2018]}
#groups = {'2019':[2019]}
groups = {}
for isat in range(2):
    for year in [2019]:
        for imonth in range(12):
            sat = 'A' if isat==1 else 'B'
            month = imonth+1
            name = f'S1{sat}_{year}{month:02d}S'
            groups[name] = (isat, year, month)

file_src = '/home/psadow/lts/preserve/stopa/sar_hs/data/alt/aggregated_2019.h5'
file_dest = '/home/psadow/lts/preserve/stopa/sar_hs/data/alt/sar_hs_2019.h5'

# Print fields of source file.
with h5py.File(file_src, 'r') as f:
    for k in [k for k in f.keys()]:
        print(f'{k}: {f[k].dtype}')

# Create h5.
with h5py.File(file_src, 'r') as fs, h5py.File(file_dest, 'w') as fd:
    for group_name, (sat, year, month) in groups.items():
        grp = fd.create_group(group_name)
        
        # Find examples from this sat, year, month.
        indices = np.ones_like(fs['year'][:], dtype='bool')
        indices = np.logical_and(fs['year'][:] == year, indices)
        indices = np.logical_and(fs['month'][:] == month, indices)
        indices = np.logical_and(fs['satellite'][:] == sat, indices)
        num_examples = indices.sum()
        print(f'Found {num_examples} events for sat {sat}, year {year}, month {month}.')
        
        # Write data from this year.
        grp.create_dataset('year', data=fs['year'][indices])    
        
        # Get 22 CWAVE features. 
        cwave = np.hstack([fs['S'][indices,...], fs['sigma0'][indices].reshape(-1,1), fs['normalizedVariance'][indices].reshape(-1,1)])
        cwave = preprocess.conv_cwave(cwave) # Remove extrema, then standardize with hardcoded mean,vars.
        grp.create_dataset('cwave', data=cwave)
        
        # Additional features. 
        #dx = preprocess.conv_dx(fs['dx'][indices])
        #dt = preprocess.conv_dt(fs['dt'][indices])
        #grp.create_dataset('dxdt', data=np.column_stack([dx, dt]))
        
        latSAR = fs['latSAR'][indices]
        lonSAR = fs['lonSAR'][indices]
        latSARcossin = preprocess.conv_position(latSAR) # Gets cos and sin
        lonSARcossin = preprocess.conv_position(lonSAR)
        grp.create_dataset('latlonSAR', data=np.column_stack([latSAR, lonSAR]))
        grp.create_dataset('latlonSARcossin', data=np.hstack([latSARcossin, lonSARcossin]))
        
        timeSAR = fs['timeSAR'][indices]
        todSAR = preprocess.conv_time(timeSAR)
        grp.create_dataset('timeSAR', data=timeSAR, shape=(timeSAR.shape[0], 1))
        grp.create_dataset('todSAR', data=todSAR, shape=(todSAR.shape[0], 1))
        
        incidence = preprocess.conv_incidence(fs['incidenceAngle'][indices]) # Separates into 2 var.
        grp.create_dataset('incidence', data=incidence)
        
        satellite = fs['satellite'][indices]
        grp.create_dataset('satellite', data=satellite, shape=(satellite.shape[0], 1))
        
        # Altimeter
        #hsALT = fs['hsALT'][indices]
        #grp.create_dataset('hsALT', data=hsALT, shape=(hsALT.shape[0], 1))
        
        # Get spectral data.
        re = preprocess.conv_real(fs['cspcRe'][indices,...])
        im = preprocess.conv_imaginary(fs['cspcIm'][indices,...])
        x = np.stack((re, im), axis=3)
        grp.create_dataset('spectrum', data=x)
print('Done')


# In[ ]:




