#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:57:12 2025

@author: ammir
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:57:47 2024

@author: ammir
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segyio
import warnings
import mne
import logging
import seismic_functions as sf
# Disable all logging output
logging.disable(logging.CRITICAL)
import os
warnings.filterwarnings('ignore')
from shapely.geometry import Point, Polygon
import matplotlib.patches as patches
#%%

## gets the velocity geometry dataframe and binary
segy_folder = "/home/ammir/Desktop/Seismic Dataset/BP Benchmark 2004/ShotGather/"
files = sf.get_all_files(segy_folder)

# segy_folder = "/home/ammir/Desktop/Seismic Dataset/TeaPot/ShotGather/"
# files = segy_folder + 'npr3_field.sgy'

# sampling_df,Geometry_Dataframe,Text_headers,Seismogram=sf.get_full_acquisition_geometry(files,True)
sampling_df,Geometry_Dataframe,Text_headers=sf.get_full_acquisition_geometry(files,False)

Geometry_Dataframe = Geometry_Dataframe.sort_values(by=["FieldRecord", "CDP_TRACE"], ascending=[True, True])

#%%

index_order = Geometry_Dataframe.index.to_numpy()

sorted_data = Seismogram[index_order]

Seismogram = sorted_data

del sorted_data
#%%
## puts the sx, sy, gx and gy in correct utm coordinates and creates local dataframe
column_names = ['SourceX','SourceY','SourceZ','GroupX','GroupY','GroupZ']
# Create an empty DataFrame using only the column names
Local_Geometry_df = pd.DataFrame(columns=column_names)

Local_Geometry_df['SourceX']=Geometry_Dataframe['SourceX']/10.
Local_Geometry_df['SourceY']=Geometry_Dataframe['SourceY']/10.
Local_Geometry_df['SourceZ']=Geometry_Dataframe['SourceDepth']/10.

Local_Geometry_df['GroupX']=Geometry_Dataframe['GroupX']/10.
Local_Geometry_df['GroupY']=Geometry_Dataframe['GroupY']/10.
Local_Geometry_df['GroupZ']=Geometry_Dataframe['SourceSurfaceElevation']/10.

Local_Geometry_df['FFID']= sf.enumerateFFIDs(Geometry_Dataframe['FieldRecord'].to_numpy())


#%%
Geometry_Dataframe['SourceX']=np.asarray(Geometry_Dataframe['SourceX'].to_numpy(),dtype=float)/1000.
Geometry_Dataframe['SourceY']=np.asarray(Geometry_Dataframe['SourceY'].to_numpy(),dtype=float)/1000.

Geometry_Dataframe['GroupX']=np.asarray(Geometry_Dataframe['GroupX'].to_numpy(),dtype=float)/1000.
Geometry_Dataframe['GroupY']=np.asarray(Geometry_Dataframe['GroupY'].to_numpy(),dtype=float)/1000.
# Geometry_Dataframe['SourceDepth']/=10.
# Geometry_Dataframe['SourceSurfaceElevation']/=10.

#%%


## gets the velocity model dataframe and binary
segy_path = "/home/ammir/Desktop/Seismic Dataset/BP Benchmark 2004/Properties/vel_z6.25m_x12.5m_exact.segy"
# segy_path = "/home/ammir/Desktop/Seismic Dataset/TeaPot/filt_mig.sgy"

Model_sampling_Dataframe,Model_Dataframe,Model,text_header_velocity = sf.get_velocity_model_segy_geometry(segy_path)


#%%
## puts the sx, sy, gx and gy in correct utm coordinates
# Create an empty DataFrame using only the column names
column_names = ['SourceX','SourceY','INLINE','CROSSLINE']
Local_Model_df = pd.DataFrame(columns=column_names)

Local_Model_df['SourceX']=Model_Dataframe['SourceX']/10.
Local_Model_df['SourceY']=Model_Dataframe['SourceY']/10.

Local_Model_df['INLINE']=Model_Dataframe['CDP_TRACE']
Local_Model_df['CROSSLINE']=Model_Dataframe['offset']
# Model_Dataframe['SourceDepth']/=10.

#%%
## plots the acquisition
sf.plot_acquisition(Model_Dataframe,Geometry_Dataframe,'utm')


#%%

sf.GenerateLocalCoordinates(Local_Model_df,Local_Geometry_df,False)
sf.GenerateLocalCoordinates(Local_Model_df,Local_Model_df,True)


#%%
# remove receivers or sources outside the model boundary

condition = Local_Geometry_df['SourceX'].to_numpy() >= 0
indices = np.where(condition)[0]

Local_Geometry_df = Local_Geometry_df.drop(Local_Geometry_df[Local_Geometry_df['SourceX'] < 0].index)
Seismogram = Seismogram[indices]

###
condition = Local_Geometry_df['SourceY'].to_numpy() >= 0
indices = np.where(condition)[0]

Local_Geometry_df = Local_Geometry_df.drop(Local_Geometry_df[Local_Geometry_df['SourceY'] < 0].index)
Seismogram = Seismogram[indices]

###
condition = Local_Geometry_df['GroupX'].to_numpy() >= 0
indices = np.where(condition)[0]

Local_Geometry_df = Local_Geometry_df.drop(Local_Geometry_df[Local_Geometry_df['GroupX'] < 0].index)
Seismogram = Seismogram[indices]

###
condition = Local_Geometry_df['GroupY'].to_numpy() >= 0
indices = np.where(condition)[0]

Local_Geometry_df = Local_Geometry_df.drop(Local_Geometry_df[Local_Geometry_df['GroupY'] < 0].index)
Seismogram = Seismogram[indices]

#%%

sf.plot_acquisition(Local_Model_df,Local_Geometry_df,'utm')


#%%

sf.write_2D_geometry_file(Local_Geometry_df,"/home/ammir/Desktop/Prog/Suite/2D/Geometry/BP2004.txt")

#%%

Local_Geometry_df.to_parquet("/home/ammir/Desktop/Seismic Dataset/BP Benchmark 2004/ShotGather/SourceReceiverGeometry.parquet", index=False)
Local_Model_df.to_parquet("/home/ammir/Desktop/Seismic Dataset/BP Benchmark 2004/ShotGather/ModelGeometry.parquet", index=False)

Seismogram=Seismogram.T

np.save("/home/ammir/Desktop/Seismic Dataset/BP Benchmark 2004/ShotGather/KillTracesSeismogram.npy", Seismogram)  # Save to a file

#%%
# filename="/home/ammir/Desktop/Prog/Suite/2D/Seismograms/BP2007/NoMute/Seismogram_BP2004_Nt2001_Ntraces1439248_dt6ms.bin"
filename="/home/ammir/Desktop/Prog/Suite/2D/Models/BP2007_Density_Nx5395_Nz1911_dx12.5_dz6.25.bin"
sf.write_binary(filename,Model)

#%%

Local_Geometry_df['offset'] = np.sqrt((Local_Geometry_df['SourceX'] - Local_Geometry_df['GroupX'])**2 + 
                                    (Local_Geometry_df['SourceY'] - Local_Geometry_df['GroupY'])**2)

#%%


condition = Local_Geometry_df['offset'].to_numpy() == 0
indices = np.where(condition)[0]

Sis = -Seismogram[indices].T

perc=np.percentile(Sis,95)
plt.figure()
plt.imshow(Sis,vmin=-perc,vmax=perc,aspect='auto',cmap='gray_r')
plt.tight_layout()

#%%

wavelet = np.sum(Sis[:30,:],axis=1)

#%%
plt.figure()
plt.plot(wavelet)







