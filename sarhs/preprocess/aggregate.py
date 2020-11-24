#!/usr/bin/env python
# coding: utf-8

# In[1]:


from netCDF4 import Dataset
import numpy as np
import glob
import h5py
import pandas as pd
from tqdm import tqdm
import re
import os


# In[37]:


# Aggregate netCDF4 files into large h5 file.
files_src = glob.glob("/mnt/lts/nfs_fs02/sadow_lab/personal/quachb/sar_hs/*.nc")
files_src = [f for f in files_src if 'ALT' in f]
#file_dest =  "/mnt/tmp/psadow/sar/aggregated_ALT.h5"
file_dest =  "/mnt/lts/nfs_fs02/sadow_lab/preserve/stopa/sar_hs/data/alt/aggregated_ALT.h5"

keys = ['timeSAR', 'timeALT', 'lonSAR', 'lonALT', 'latSAR', 'latALT', 'hsALT', 'dx', 'dt', 'nk', 'hsSM', 'incidenceAngle', 'sigma0', 'normalizedVariance', 'S']
keys += ['cspcRe', 'cspcIm']

def parse_filename(filename):
    """
    Grab some meta data from filename.
    """
    filename = os.path.basename(filename)
    platform, _alt, date, _ext = re.split('_|\.', filename)
    assert _alt == 'ALT', _alt
    assert _ext == 'nc', _ext
    satellite = int(platform[2] == 'A') # Encodes type A as 1 and B as 0
    year = int(date[5:9])
    month = int(date[9:11])
    return {'satellite':satellite, 'year':year, 'month':month}

def process(x, key):
    """
    Process a netcdf variable data.variables[key]
    """
    if key == 'S':
        x.set_auto_scale(False)
        x = np.array(x[:] * float(x.scale_factor))
    return x

def aggregate(files_src, file_dest, keys=None):
    """
    Aggregate list of netcdf files into single hdf5.
    Args:
    files_src: list of netcdf filenames
    file_dest: filename of h5
    keys: If specified, only extract these fields.
    """
    
    for i, filename in tqdm(enumerate(files_src)):
        # Add file of data to large hdf5.
        #print(filename)
        data = Dataset(filename)
        meta = parse_filename(filename)        
        
        if i == 0:
            if keys is None:
                # Grab keys from first file.
                keys = data.variables.keys()
            with h5py.File(file_dest, 'w') as fdest:
                for key in keys:
                    print(key)
                    x = process(data.variables[key], key)
                    maxshape = (None,) if len(x.shape)==1 else (None, ) + x.shape[1:]
                    fdest.create_dataset(key, data=x, maxshape=maxshape)
                for key in meta:
                    temp = np.ones((data.variables[keys[0]].shape[0], ), dtype=int) * meta[key] 
                    fdest.create_dataset(key, data=temp, maxshape = (None,))
        else:
            with h5py.File(file_dest, 'a') as fdest:
                for key in keys:
                    num_prev = fdest[key].shape[0]
                    num_add = data.variables[key].shape[0]
                    fdest[key].resize(num_prev + num_add, axis = 0)
                    fdest[key][-num_add:] = process(data.variables[key], key)
                for key in meta:
                    num_prev = fdest[key].shape[0]
                    fdest[key].resize(num_prev + num_add, axis = 0)
                    fdest[key][-num_add:] = np.ones((data.variables[keys[0]].shape[0], ), dtype=int) * meta[key] 

aggregate(files_src, file_dest, keys=keys)
print("Done")

