# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:54:38 2020

@author: bcamc
"""


#%% Import python packages
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.lines import Line2D
import scipy
import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate
import statsmodels.api as sm
import statsmodels.formula.api as smf
from brokenaxes import brokenaxes
import cmocean.cm as cmo
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import seawater as sw
import matplotlib.patches as patches
import datetime
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#%% Set plotting backend (ensure Spyder 'Graphics Backend' is set to automatic):
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','inline')
#%% Import Saanich data:
#------------------------------------------------------------------------------
saanich_data = pd.read_excel('C:/Users/bcamc/OneDrive/Documents/Education/UVic/Honours/Data/Saanich Data/Saanich_Master_Compilation.xlsx', sheet_name='Data', header=[1,2])
# drop 2017 non-euphotic zone depths:
saanich_data = saanich_data.drop(saanich_data.index[560:569])
# drop 2005/2006/hallam data:
saanich_data = saanich_data.drop(saanich_data.index[0:246])
#------------------------------------------------------------------------------
#%% Import depth integrated data
#------------------------------------------------------------------------------
saanich_integ  = pd.read_excel('C:/Users/bcamc/OneDrive/Documents/Education/UVic/Honours/Data/Saanich Data/Saanich_Master_Compilation.xlsx', sheet_name='Total Depth Integrated', header=[1,2])

# reset index:
saanich_integ.columns = saanich_integ.columns.droplevel(1)

# drop date header rows:
saanich_integ = saanich_integ.dropna(subset=['Date'])

# Replace duplecated headers:
saanich_integ.columns=['Date','Cruise ID','Station','LAT','LON','Light Penetration','Depth',
                         '[NO3-]','[NO2]','[PO43-]','[Si(OH)4]',
                         'CHL-A (>0.7 um)','CHL-A (0.7-2 um)','CHL-A (2-5 um)','CHL-A (0.7-5 um)','CHL-A (0.75-5 um)','CHL-A (5-20 um)','CHL-A (>20 um)','TOTAL CHL-A','CHECK Tot chl',
                         'Initial [bSiO2]', 'Final [bSiO2]','[PC] (ug/L)', '[PC] (umol/L)','[PN] (ug/L)', '[PN] (umol/L)',
                         'bSiO2 Precipitation','rhoC (mg/m2/d)','rhoC (mmol/m2/d)','rhoN (mg/m2/d)','rhoN (mmol/m2/d)',
                         'bSi:Chl','bSi:POC','bSi:PON','POC:PON','Chl:POC']
#------------------------------------------------------------------------------
# create piecewise dataframes:
saanich_integ_2010 = saanich_integ.loc[1:60]
saanich_integ_2011 = saanich_integ.loc[62:113]
saanich_integ_2012_2013 = saanich_integ.loc[115:134]
saanich_integ_2014_2015 = saanich_integ.loc[136:228]
saanich_integ_2016_2017 = saanich_integ.loc[230:313]
#------------------------------------------------------------------------------
#### Average depth-integrated values by month:
#------------------------------------------------------------------------------
# # Insert month column:
# saanich_integ.insert(1, "Month", saanich_integ['Date'].dt.month)

# # Define 1-m depth interval bins:
# bins = np.arange(0,np.max(saanich_integ['Depth']),1)

# # Bin data as averages across 1-m bins by sampling date:
# saanich_integ_binned = saanich_integ.groupby(['Month', pd.cut(saanich_integ['Depth'], bins)]).mean()

# # Rename binned column:
# saanich_integ_binned = saanich_integ_binned.rename_axis(index=['Month', 'Depth Bins'])

# # Transform dates back from integers to datetime numbers:
# saanich_integ_binned.reset_index(inplace=True) # remove index specification on columns
# # saanich_integ_binned['Date'] = pd.to_datetime(saanich_integ_binned['Date'],format=None)

# # Redefine depth data as the edge values of bins rather than averages:
# array_rep_size = np.array(np.size(saanich_integ_binned['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
# new_values = bins[:-1]+1
# saanich_integ_binned['Depth'] = np.tile(new_values,array_rep_size)

# # drop unnecessary columns:
# saanich_integ_binned = saanich_integ_binned.drop(columns=['Station','LAT','LON','Light Penetration'])
#%% Nuts: Bin, reshape, interpolate & grid sampling over time
saanich_convert = saanich_data.iloc[:,[0,6,11,12,13,14]] # Nuts
#------------------------------------------------------------------------------
# reset index:
saanich_convert.columns = saanich_convert.columns.droplevel(1)

# # filter out outliers (replacing as min of range of outliers) for plotting, where
# # outliers are defined as value-mean>2*SD:
# outliers = np.where(np.abs(saanich_convert['[NO3-]']-np.mean(saanich_convert['[NO3-]'])>2*np.std(saanich_convert['[NO3-]'])))
# NO3_min_outlier = np.min(saanich_convert['[NO3-]'].iloc[outliers[0]])
# saanich_convert['[NO3-]'].iloc[outliers[0]] = NO3_min_outlier

# outliers = np.where(np.abs(saanich_convert['[NO2]']-np.mean(saanich_convert['[NO2]'])>2*np.std(saanich_convert['[NO2]'])))
# NO2_min_outlier = np.min(saanich_convert['[NO2]'].iloc[outliers[0]])
# saanich_convert['[NO2]'].iloc[outliers[0]] = NO2_min_outlier

# outliers = np.where(np.abs(saanich_convert['[PO43-]']-np.mean(saanich_convert['[PO43-]'])>2*np.std(saanich_convert['[PO43-]'])))
# PO4_min_outlier = np.min(saanich_convert['[PO43-]'].iloc[outliers[0]])
# saanich_convert['[PO43-]'].iloc[outliers[0]] = PO4_min_outlier

# outliers = np.where(np.abs(saanich_convert['[Si(OH)4]']-np.mean(saanich_convert['[Si(OH)4]'])>2*np.std(saanich_convert['[Si(OH)4]'])))
# SiOH4_min_outlier = np.min(saanich_convert['[Si(OH)4]'].iloc[outliers[0]])
# saanich_convert['[Si(OH)4]'].iloc[outliers[0]] = SiOH4_min_outlier

# drop date header rows:
saanich_convert = saanich_convert.dropna(subset=['Date'])

# Replace non-integer values with zeros
saanich_convert['[NO3-]'] = saanich_convert['[NO3-]'].apply(lambda x: 0 if str(type(x))=="<class 'str'>" else x)
saanich_convert['[PO43-]'] = saanich_convert['[PO43-]'].apply(lambda x: 0 if str(type(x))=="<class 'str'>" else x)
saanich_convert['[Si(OH)4]'] = saanich_convert['[Si(OH)4]'].apply(lambda x: 0 if str(type(x))=="<class 'str'>" else x)

# Create seperate dataframes by data clusters over time:
saanich_convert_2010 = saanich_convert.loc[247:306]
saanich_convert_2011 = saanich_convert.loc[308:359]
saanich_convert_2012_2013 = saanich_convert.loc[361:380]
saanich_convert_2014_2015 = saanich_convert.loc[382:474]
saanich_convert_2016_2017 = saanich_convert.loc[476:559]

# Redefine columns as float data type to be readable by binning functions:
saanich_convert_2010['Date'] = saanich_convert_2010['Date'].values.astype('float64') # need to convert datetimes first
saanich_convert_2011['Date'] = saanich_convert_2011['Date'].values.astype('float64') # need to convert datetimes first
saanich_convert_2012_2013['Date'] = saanich_convert_2012_2013['Date'].values.astype('float64') # need to convert datetimes first
saanich_convert_2014_2015['Date'] = saanich_convert_2014_2015['Date'].values.astype('float64') # need to convert datetimes first
saanich_convert_2016_2017['Date'] = saanich_convert_2016_2017['Date'].values.astype('float64') # need to convert datetimes first

# Define 1-m depth interval bins:
bins = np.arange(0,np.max(saanich_convert['Depth']),1)
# bins = np.arange(0,21,1) # restricted to interpolate only over upper 20-m depths

# Bin data as averages across 1-m bins by sampling date:
nuts_binned_2010 = saanich_convert_2010.groupby(['Date', pd.cut(saanich_convert_2010['Depth'], bins)]).mean()
nuts_binned_2011 = saanich_convert_2011.groupby(['Date', pd.cut(saanich_convert_2011['Depth'], bins)]).mean()
nuts_binned_2012_2013 = saanich_convert_2012_2013.groupby(['Date', pd.cut(saanich_convert_2012_2013['Depth'], bins)]).mean()
nuts_binned_2014_2015 = saanich_convert_2014_2015.groupby(['Date', pd.cut(saanich_convert_2014_2015['Depth'], bins)]).mean()
nuts_binned_2016_2017 = saanich_convert_2016_2017.groupby(['Date', pd.cut(saanich_convert_2016_2017['Depth'], bins)]).mean()

# Drop garbage data produced:
# nuts_binned = nuts_binned.drop(nuts_binned.index[0:217])

# Rename binned column:
nuts_binned_2010 = nuts_binned_2010.rename_axis(index=['Date', 'Depth Bins'])
nuts_binned_2011 = nuts_binned_2011.rename_axis(index=['Date', 'Depth Bins'])
nuts_binned_2012_2013 = nuts_binned_2012_2013.rename_axis(index=['Date', 'Depth Bins'])
nuts_binned_2014_2015 = nuts_binned_2014_2015.rename_axis(index=['Date', 'Depth Bins'])
nuts_binned_2016_2017 = nuts_binned_2016_2017.rename_axis(index=['Date', 'Depth Bins'])

# Transform dates back from integers to datetime numbers:
nuts_binned_2010.reset_index(inplace=True) # remove index specification on columns
nuts_binned_2011.reset_index(inplace=True) # remove index specification on columns
nuts_binned_2012_2013.reset_index(inplace=True) # remove index specification on columns
nuts_binned_2014_2015.reset_index(inplace=True) # remove index specification on columns
nuts_binned_2016_2017.reset_index(inplace=True) # remove index specification on columns

nuts_binned_2010['Date'] = pd.to_datetime(nuts_binned_2010['Date'],format=None)
nuts_binned_2011['Date'] = pd.to_datetime(nuts_binned_2011['Date'],format=None)
nuts_binned_2012_2013['Date'] = pd.to_datetime(nuts_binned_2012_2013['Date'],format=None)
nuts_binned_2014_2015['Date'] = pd.to_datetime(nuts_binned_2014_2015['Date'],format=None)
nuts_binned_2016_2017['Date'] = pd.to_datetime(nuts_binned_2016_2017['Date'],format=None)

# Redefine depth data as the edge values of bins rather than averages:
array_rep_size1 = np.array(np.size(nuts_binned_2010['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
array_rep_size2 = np.array(np.size(nuts_binned_2011['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
array_rep_size3 = np.array(np.size(nuts_binned_2012_2013['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
array_rep_size4 = np.array(np.size(nuts_binned_2014_2015['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
array_rep_size5 = np.array(np.size(nuts_binned_2016_2017['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
new_values = bins[:-1]+1
nuts_binned_2010['Depth'] = np.tile(new_values,array_rep_size1)
nuts_binned_2011['Depth'] = np.tile(new_values,array_rep_size2)
nuts_binned_2012_2013['Depth'] = np.tile(new_values,array_rep_size3)
nuts_binned_2014_2015['Depth'] = np.tile(new_values,array_rep_size4)
nuts_binned_2016_2017['Depth'] = np.tile(new_values,array_rep_size5)
#------------------------------------------------------------------------------
#### Reshape data from long list format to a 'date x depth' matrix:
# NO3
# NO3_wide = nuts_binned.loc[:,['Date','Depth','[NO3-]']].pivot(index='Date', columns='Depth').T
# NO3_wide = NO3_wide.rename_axis(index=['Temp', 'Depth']).reset_index(drop=True, level='Temp')
# # PO2
# NO2_wide = nuts_binned.loc[:,['Date','Depth','[NO2]']].pivot(index='Date', columns='Depth').T
# NO2_wide = NO2_wide.rename_axis(index=['Temp', 'Depth']).reset_index(drop=True, level='Temp')
# # PO4
# PO4_wide = nuts_binned.loc[:,['Date','Depth','[PO43-]']].pivot(index='Date', columns='Depth').T
# PO4_wide = PO4_wide.rename_axis(index=['Temp', 'Depth']).reset_index(drop=True, level='Temp')
# # Si(OH)4
# Si_wide = nuts_binned.loc[:,['Date','Depth','[Si(OH)4]']].pivot(index='Date', columns='Depth').T
# Si_wide = Si_wide.rename_axis(index=['Temp', 'Depth']).reset_index(drop=True, level='Temp')

#------------------------------------------------------------------------------
#### Interpolate the data to fill in NaNs:
#### 2010
total_depth = np.max(nuts_binned_2010['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(nuts_binned_2010['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

# NO3
indF1 = nuts_binned_2010.loc[:,'[NO3-]'].notna()
date_pts = nuts_binned_2010['Date'][indF1].values
depth_pts = nuts_binned_2010['Depth'][indF1].values
values1 = nuts_binned_2010.loc[:,'[NO3-]'][indF1].values

NO3_wide_interp_2010 = scipy.interpolate.griddata((date_pts,depth_pts), values1, (xi,yi), rescale='QJ', method='linear')
NO3_wide_interp_2010[NO3_wide_interp_2010<0] = np.nan # mask negative points

# # NO2
# indF2 = nuts_binned_2010.loc[:,'[NO2]'].notna() 
# date_pts = nuts_binned_2010['Date'][indF2].values
# depth_pts = nuts_binned_2010['Depth'][indF2].values
# values2 = nuts_binned_2010.loc[:,'[NO2]'][indF2].values

# NO2_wide_interp_2010 = scipy.interpolate.griddata((date_pts,depth_pts), values2, (xi,yi), rescale='QJ', method='linear')
# NO2_wide_interp_2010[NO2_wide_interp_2010<0] = np.nan # mask negative points

# PO4
indF3 = nuts_binned_2010.loc[:,'[PO43-]'].notna() 
date_pts = nuts_binned_2010['Date'][indF3].values
depth_pts = nuts_binned_2010['Depth'][indF3].values
values3 = nuts_binned_2010.loc[:,'[PO43-]'][indF3].values

PO4_wide_interp_2010 = scipy.interpolate.griddata((date_pts,depth_pts), values3, (xi,yi), rescale='QJ', method='linear')
PO4_wide_interp_2010[PO4_wide_interp_2010<0] = np.nan # mask invalid points

# Si(OH)4
indF4 = nuts_binned_2010.loc[:,'[Si(OH)4]'].notna() 
date_pts = nuts_binned_2010['Date'][indF4].values
depth_pts = nuts_binned_2010['Depth'][indF4].values
values4 = nuts_binned_2010.loc[:,'[Si(OH)4]'][indF4].values

Si_wide_interp_2010 = scipy.interpolate.griddata((date_pts,depth_pts), values4, (xi,yi), rescale='QJ', method='linear')
Si_wide_interp_2010[Si_wide_interp_2010<0] = np.nan # mask invalid points
#------------------------------------------------------------------------------
#### 2011
total_depth = np.max(nuts_binned_2011['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(nuts_binned_2011['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

# NO3
indF1 = nuts_binned_2011.loc[:,'[NO3-]'].notna()
date_pts = nuts_binned_2011['Date'][indF1].values
depth_pts = nuts_binned_2011['Depth'][indF1].values
values1 = nuts_binned_2011.loc[:,'[NO3-]'][indF1].values

NO3_wide_interp_2011 = scipy.interpolate.griddata((date_pts,depth_pts), values1, (xi,yi), rescale='QJ', method='linear')
NO3_wide_interp_2011[NO3_wide_interp_2011<0] = 0 # mask negative points

# # NO2
# indF2 = nuts_binned_2011.loc[:,'[NO2]'].notna() 
# date_pts = nuts_binned_2011['Date'][indF2].values
# depth_pts = nuts_binned_2011['Depth'][indF2].values
# values2 = nuts_binned_2011.loc[:,'[NO2]'][indF2].values

# NO2_wide_interp_2011 = scipy.interpolate.griddata((date_pts,depth_pts), values2, (xi,yi), rescale='QJ', method='linear')
# NO2_wide_interp_2011[NO2_wide_interp_2011<0] = np.nan # mask negative points

# PO4
indF3 = nuts_binned_2011.loc[:,'[PO43-]'].notna() 
date_pts = nuts_binned_2011['Date'][indF3].values
depth_pts = nuts_binned_2011['Depth'][indF3].values
values3 = nuts_binned_2011.loc[:,'[PO43-]'][indF3].values

PO4_wide_interp_2011 = scipy.interpolate.griddata((date_pts,depth_pts), values3, (xi,yi), rescale='QJ', method='linear')
PO4_wide_interp_2011[PO4_wide_interp_2011<0] = np.nan # mask invalid points

# Si(OH)4
indF4 = nuts_binned_2011.loc[:,'[Si(OH)4]'].notna() 
date_pts = nuts_binned_2011['Date'][indF4].values
depth_pts = nuts_binned_2011['Depth'][indF4].values
values4 = nuts_binned_2011.loc[:,'[Si(OH)4]'][indF4].values

Si_wide_interp_2011 = scipy.interpolate.griddata((date_pts,depth_pts), values4, (xi,yi), rescale='QJ', method='linear')
Si_wide_interp_2011[Si_wide_interp_2011<0] = 0 # mask invalid points
#------------------------------------------------------------------------------
#### 2012-2013
total_depth = np.max(nuts_binned_2012_2013['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(nuts_binned_2012_2013['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

# NO3
indF1 = nuts_binned_2012_2013.loc[:,'[NO3-]'].notna()
date_pts = nuts_binned_2012_2013['Date'][indF1].values
depth_pts = nuts_binned_2012_2013['Depth'][indF1].values
values1 = nuts_binned_2012_2013.loc[:,'[NO3-]'][indF1].values

NO3_wide_interp_2012_2013 = scipy.interpolate.griddata((date_pts,depth_pts), values1, (xi,yi), rescale='QJ', method='linear')
NO3_wide_interp_2012_2013[NO3_wide_interp_2012_2013<0] = np.nan # mask negative points
# NO3_wide_interp_2012_2013[NO3_wide_interp_2012_2013>10] = np.nan # mask poor interpolition

# # NO2
# indF2 = nuts_binned_2012_2013.loc[:,'[NO2]'].notna() 
# date_pts = nuts_binned_2012_2013['Date'][indF2].values
# depth_pts = nuts_binned_2012_2013['Depth'][indF2].values
# values2 = nuts_binned_2012_2013.loc[:,'[NO2]'][indF2].values

# NO2_wide_interp_2012_2013 = scipy.interpolate.griddata((date_pts,depth_pts), values2, (xi,yi), rescale='QJ', method='linear')
# NO2_wide_interp_2012_2013[NO2_wide_interp_2012_2013<0] = np.nan # mask negative points

# PO4
indF3 = nuts_binned_2012_2013.loc[:,'[PO43-]'].notna() 
date_pts = nuts_binned_2012_2013['Date'][indF3].values
depth_pts = nuts_binned_2012_2013['Depth'][indF3].values
values3 = nuts_binned_2012_2013.loc[:,'[PO43-]'][indF3].values

PO4_wide_interp_2012_2013 = scipy.interpolate.griddata((date_pts,depth_pts), values3, (xi,yi), rescale='QJ', method='linear')
PO4_wide_interp_2012_2013[PO4_wide_interp_2012_2013<0] = np.nan # mask invalid points

# Si(OH)4
indF4 = nuts_binned_2012_2013.loc[:,'[Si(OH)4]'].notna() 
date_pts = nuts_binned_2012_2013['Date'][indF4].values
depth_pts = nuts_binned_2012_2013['Depth'][indF4].values
values4 = nuts_binned_2012_2013.loc[:,'[Si(OH)4]'][indF4].values

Si_wide_interp_2012_2013 = scipy.interpolate.griddata((date_pts,depth_pts), values4, (xi,yi), rescale='QJ', method='linear')
Si_wide_interp_2012_2013[Si_wide_interp_2012_2013<0] = np.nan # mask invalid points

#------------------------------------------------------------------------------
#### 2014-2015
total_depth = np.max(nuts_binned_2014_2015['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(nuts_binned_2014_2015['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

# NO3
indF1 = nuts_binned_2014_2015.loc[:,'[NO3-]'].notna()
date_pts = nuts_binned_2014_2015['Date'][indF1].values
depth_pts = nuts_binned_2014_2015['Depth'][indF1].values
values1 = nuts_binned_2014_2015.loc[:,'[NO3-]'][indF1].values

NO3_wide_interp_2014_2015 = scipy.interpolate.griddata((date_pts,depth_pts), values1, (xi,yi), rescale='QJ', method='linear')
NO3_wide_interp_2014_2015[NO3_wide_interp_2014_2015<0] = np.nan # mask negative points

# # NO2
# indF2 = nuts_binned_2014_2015.loc[:,'[NO2]'].notna() 
# date_pts = nuts_binned_2014_2015['Date'][indF2].values
# depth_pts = nuts_binned_2014_2015['Depth'][indF2].values
# values2 = nuts_binned_2014_2015.loc[:,'[NO2]'][indF2].values

# NO2_wide_interp_2014_2015 = scipy.interpolate.griddata((date_pts,depth_pts), values2, (xi,yi), rescale='QJ', method='linear')
# NO2_wide_interp_2014_2015[NO2_wide_interp_2014_2015<0] = np.nan # mask negative points

# PO4
indF3 = nuts_binned_2014_2015.loc[:,'[PO43-]'].notna() 
date_pts = nuts_binned_2014_2015['Date'][indF3].values
depth_pts = nuts_binned_2014_2015['Depth'][indF3].values
values3 = nuts_binned_2014_2015.loc[:,'[PO43-]'][indF3].values

PO4_wide_interp_2014_2015 = scipy.interpolate.griddata((date_pts,depth_pts), values3, (xi,yi), rescale='QJ', method='linear')
PO4_wide_interp_2014_2015[PO4_wide_interp_2014_2015<0] = np.nan # mask invalid points

# Si(OH)4
indF4 = nuts_binned_2014_2015.loc[:,'[Si(OH)4]'].notna() 
date_pts = nuts_binned_2014_2015['Date'][indF4].values
depth_pts = nuts_binned_2014_2015['Depth'][indF4].values
values4 = nuts_binned_2014_2015.loc[:,'[Si(OH)4]'][indF4].values

Si_wide_interp_2014_2015 = scipy.interpolate.griddata((date_pts,depth_pts), values4, (xi,yi), rescale='QJ', method='linear')
Si_wide_interp_2014_2015[Si_wide_interp_2014_2015<0] = np.nan # mask invalid points
#------------------------------------------------------------------------------
#### 2016-2017
total_depth = np.max(nuts_binned_2016_2017['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(nuts_binned_2016_2017['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

# NO3
indF1 = nuts_binned_2016_2017.loc[:,'[NO3-]'].notna()
date_pts = nuts_binned_2016_2017['Date'][indF1].values
depth_pts = nuts_binned_2016_2017['Depth'][indF1].values
values1 = nuts_binned_2016_2017.loc[:,'[NO3-]'][indF1].values

NO3_wide_interp_2016_2017 = scipy.interpolate.griddata((date_pts,depth_pts), values1, (xi,yi), rescale='QJ', method='linear')
NO3_wide_interp_2016_2017[NO3_wide_interp_2016_2017<0] = np.nan # mask negative points

# NO2
indF2 = nuts_binned_2016_2017.loc[:,'[NO2]'].notna() 
date_pts = nuts_binned_2016_2017['Date'][indF2].values
depth_pts = nuts_binned_2016_2017['Depth'][indF2].values
values2 = nuts_binned_2016_2017.loc[:,'[NO2]'][indF2].values

NO2_wide_interp_2016_2017 = scipy.interpolate.griddata((date_pts,depth_pts), values2, (xi,yi), rescale='QJ', method='linear')
NO2_wide_interp_2016_2017[NO2_wide_interp_2016_2017<0] = np.nan # mask negative points

# PO4
indF3 = nuts_binned_2016_2017.loc[:,'[PO43-]'].notna() 
date_pts = nuts_binned_2016_2017['Date'][indF3].values
depth_pts = nuts_binned_2016_2017['Depth'][indF3].values
values3 = nuts_binned_2016_2017.loc[:,'[PO43-]'][indF3].values

PO4_wide_interp_2016_2017 = scipy.interpolate.griddata((date_pts,depth_pts), values3, (xi,yi), rescale='QJ', method='linear')
PO4_wide_interp_2016_2017[PO4_wide_interp_2016_2017<0] = np.nan # mask invalid points

# Si(OH)4
indF4 = nuts_binned_2016_2017.loc[:,'[Si(OH)4]'].notna() 
date_pts = nuts_binned_2016_2017['Date'][indF4].values
depth_pts = nuts_binned_2016_2017['Depth'][indF4].values
values4 = nuts_binned_2016_2017.loc[:,'[Si(OH)4]'][indF4].values

Si_wide_interp_2016_2017 = scipy.interpolate.griddata((date_pts,depth_pts), values4, (xi,yi), rescale='QJ', method='linear')
Si_wide_interp_2016_2017[Si_wide_interp_2016_2017<0] = np.nan # mask invalid points
#------------------------------------------------------------------------------
#%% Chl: Bin, reshape, interpolate & grid sampling of chl size fractions over time
chl = saanich_data.iloc[:,[0,6,15,16,17,18,20,21,22,23]]
#------------------------------------------------------------------------------
# reset index:
chl.columns = chl.columns.droplevel(1)

# Replace duplicated headers:
chl.columns=['Date','Depth','CHL-A (>0.7um)', 'CHL-A (0.7-2um)','CHL-A (2-5um)', 'CHL-A (0.7-5um)','CHL-A (5-20um)', 'CHL-A (>20um)', 'TOTAL CHL-A', 'CHECK Tot chl']

# drop date header rows:
chl = chl.dropna(subset=['Date'])

# filter out outliers (replacing as min of range of outliers) for plotting, where
# outliers are defined as value-mean>2*SD:
outliers = np.where(np.abs(chl['CHECK Tot chl']-np.mean(chl['CHECK Tot chl'])>2*np.std(chl['CHECK Tot chl'])))
chl_tot_min_outlier = np.min(chl['CHECK Tot chl'].iloc[outliers[0]])
chl['CHECK Tot chl'].iloc[outliers[0]] = chl_tot_min_outlier

# Create seperate dataframes by data clusters over time:
chl_2010 = chl.loc[247:306]
chl_2011 = chl.loc[308:359]
chl_2012_2013 = chl.loc[361:380]
chl_2014_2015 = chl.loc[382:474]
chl_2016_2017 = chl.loc[476:559]

# Redefine columns as float data type to be readable by binning functions:
chl_2010['Date'] = chl_2010['Date'].values.astype('float64') # need to convert datetimes first
chl_2011['Date'] = chl_2011['Date'].values.astype('float64') # need to convert datetimes first
chl_2012_2013['Date'] = chl_2012_2013['Date'].values.astype('float64') # need to convert datetimes first
chl_2014_2015['Date'] = chl_2014_2015['Date'].values.astype('float64') # need to convert datetimes first
chl_2016_2017['Date'] = chl_2016_2017['Date'].values.astype('float64') # need to convert datetimes first

# Define 1-m depth interval bins:
bins = np.arange(0,np.max(chl['Depth']),1)
# bins = np.arange(0,21,1) # restricted to interpolate only over upper 20-m depths

# Bin data as averages across 1-m bins by sampling date:
chl_binned_2010 = chl_2010.groupby(['Date', pd.cut(chl_2010['Depth'], bins)]).mean()
chl_binned_2011 = chl_2011.groupby(['Date', pd.cut(chl_2011['Depth'], bins)]).mean()
chl_binned_2012_2013 = chl_2012_2013.groupby(['Date', pd.cut(chl_2012_2013['Depth'], bins)]).mean()
chl_binned_2014_2015 = chl_2014_2015.groupby(['Date', pd.cut(chl_2014_2015['Depth'], bins)]).mean()
chl_binned_2016_2017 = chl_2016_2017.groupby(['Date', pd.cut(chl_2016_2017['Depth'], bins)]).mean()

# Drop garbage data produced:
# chl_binned = chl_binned.drop(chl_binned.index[0:217])

# Rename binned column:
chl_binned_2010 = chl_binned_2010.rename_axis(index=['Date', 'Depth Bins'])
chl_binned_2011 = chl_binned_2011.rename_axis(index=['Date', 'Depth Bins'])
chl_binned_2012_2013 = chl_binned_2012_2013.rename_axis(index=['Date', 'Depth Bins'])
chl_binned_2014_2015 = chl_binned_2014_2015.rename_axis(index=['Date', 'Depth Bins'])
chl_binned_2016_2017 = chl_binned_2016_2017.rename_axis(index=['Date', 'Depth Bins'])

# Transform dates back from integers to datetime numbers:
chl_binned_2010.reset_index(inplace=True) # remove index specification on columns
chl_binned_2011.reset_index(inplace=True) # remove index specification on columns
chl_binned_2012_2013.reset_index(inplace=True) # remove index specification on columns
chl_binned_2014_2015.reset_index(inplace=True) # remove index specification on columns
chl_binned_2016_2017.reset_index(inplace=True) # remove index specification on columns

chl_binned_2010['Date'] = pd.to_datetime(chl_binned_2010['Date'],format=None)
chl_binned_2011['Date'] = pd.to_datetime(chl_binned_2011['Date'],format=None)
chl_binned_2012_2013['Date'] = pd.to_datetime(chl_binned_2012_2013['Date'],format=None)
chl_binned_2014_2015['Date'] = pd.to_datetime(chl_binned_2014_2015['Date'],format=None)
chl_binned_2016_2017['Date'] = pd.to_datetime(chl_binned_2016_2017['Date'],format=None)

# Redefine depth data as the edge values of bins rather than averages:
array_rep_size1 = np.array(np.size(chl_binned_2010['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
array_rep_size2 = np.array(np.size(chl_binned_2011['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
array_rep_size3 = np.array(np.size(chl_binned_2012_2013['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
array_rep_size4 = np.array(np.size(chl_binned_2014_2015['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
array_rep_size5 = np.array(np.size(chl_binned_2016_2017['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
new_values = bins[:-1]+1
chl_binned_2010['Depth'] = np.tile(new_values,array_rep_size1)
chl_binned_2011['Depth'] = np.tile(new_values,array_rep_size2)
chl_binned_2012_2013['Depth'] = np.tile(new_values,array_rep_size3)
chl_binned_2014_2015['Depth'] = np.tile(new_values,array_rep_size4)
chl_binned_2016_2017['Depth'] = np.tile(new_values,array_rep_size5)
#------------------------------------------------------------------------------
#### Interpolate total chl data to fill in NaNs:
#### 2010
total_depth = np.max(chl_binned_2010['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(chl_binned_2010['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

indF6 = chl_binned_2010.loc[:,'CHECK Tot chl'].notna() 
date_pts = chl_binned_2010['Date'][indF6].values
depth_pts = chl_binned_2010['Depth'][indF6].values
values6 = chl_binned_2010.loc[:,'CHECK Tot chl'][indF6].values

chl_tot_wide_interp_2010 = scipy.interpolate.griddata((date_pts,depth_pts), values6, (xi,yi), rescale='QJ', method='linear')
# chl_tot_wide_interp_2010[chl_tot_wide_interp_2010<0] = np.nan # mask invalid points

#------------------------------------------------------------------------------
# 2011
total_depth = np.max(chl_binned_2011['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(chl_binned_2011['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

indF6 = chl_binned_2011.loc[:,'CHECK Tot chl'].notna() 
date_pts = chl_binned_2011['Date'][indF6].values
depth_pts = chl_binned_2011['Depth'][indF6].values
values6 = chl_binned_2011.loc[:,'CHECK Tot chl'][indF6].values

chl_tot_wide_interp_2011 = scipy.interpolate.griddata((date_pts,depth_pts), values6, (xi,yi), rescale='QJ', method='linear')
# chl_tot_wide_interp_2011[chl_tot_wide_interp_2011<0] = np.nan # mask invalid points

#------------------------------------------------------------------------------
#### 2012-2013
total_depth = np.max(chl_binned_2012_2013['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(chl_binned_2012_2013['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

indF6 = chl_binned_2012_2013.loc[:,'CHECK Tot chl'].notna() 
date_pts = chl_binned_2012_2013['Date'][indF6].values
depth_pts = chl_binned_2012_2013['Depth'][indF6].values
values6 = chl_binned_2012_2013.loc[:,'CHECK Tot chl'][indF6].values

chl_tot_wide_interp_2012_2013 = scipy.interpolate.griddata((date_pts,depth_pts), values6, (xi,yi), rescale='QJ', method='linear')
# chl_tot_wide_interp_2012_2013[chl_tot_wide_interp_2012_2013<0] = np.nan # mask invalid points

#------------------------------------------------------------------------------
#### 2014-2015
total_depth = np.max(chl_binned_2014_2015['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(chl_binned_2014_2015['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

indF6 = chl_binned_2014_2015.loc[:,'CHECK Tot chl'].notna() 
date_pts = chl_binned_2014_2015['Date'][indF6].values
depth_pts = chl_binned_2014_2015['Depth'][indF6].values
values6 = chl_binned_2014_2015.loc[:,'CHECK Tot chl'][indF6].values

chl_tot_wide_interp_2014_2015 = scipy.interpolate.griddata((date_pts,depth_pts), values6, (xi,yi), rescale='QJ', method='linear')
# chl_tot_wide_interp_2014_2015[chl_tot_wide_interp_2014_2015<0] = np.nan # mask invalid points

#------------------------------------------------------------------------------
#### 2016-2017
total_depth = np.max(chl_binned_2016_2017['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(chl_binned_2016_2017['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

indF6 = chl_binned_2016_2017.loc[:,'CHECK Tot chl'].notna() 
date_pts = chl_binned_2016_2017['Date'][indF6].values
depth_pts = chl_binned_2016_2017['Depth'][indF6].values
values6 = chl_binned_2016_2017.loc[:,'CHECK Tot chl'][indF6].values

chl_tot_wide_interp_2016_2017 = scipy.interpolate.griddata((date_pts,depth_pts), values6, (xi,yi), rescale='QJ', method='linear')
# chl_tot_wide_interp_2016_2017[chl_tot_wide_interp_2016_2017<0] = np.nan # mask invalid points

#------------------------------------------------------------------------------
#%% Chl %: Reshape, interpolate & grid sampling of chl size fractions over time
chl = saanich_data.iloc[:,[0,6,15,16,17,18,20,21,22,23]]
#------------------------------------------------------------------------------
# reset index:
chl.columns = chl.columns.droplevel(1)

# Replace duplicated headers:
chl.columns=['Date','Depth','CHL-A (>0.7um)', 'CHL-A (0.7-2um)','CHL-A (2-5um)', 'CHL-A (0.7-5um)','CHL-A (5-20um)', 'CHL-A (>20um)', 'TOTAL CHL-A', 'CHECK Tot chl']

# drop date header rows:
chl = chl.dropna(subset=['Date'])
#------------------------------------------------------------------------------
#### Scale single depth values to percentages (2012-2013):
# first, calculate percentages for all data:
chl_percent_single = chl.iloc[:,[2,3,4,5,6,7,8,9]].div(chl.iloc[:,9].values, axis=0)*100
# Add in date/depth columns for indexing:
chl_percent_single.insert(0, 'Date', chl['Date'])
chl_percent_single.insert(1, 'Depth', chl['Depth'])
# Now limit to 2012-2013:
chl_percent_single = chl_percent_single.loc[361:380]
#------------------------------------------------------------------------------
#### Scale depth integ values to percentages for remaining data:
# create new datframe:
chl_percent_integ = saanich_integ.iloc[:,[11,12,13,14,15,16,17,18,19]].div(saanich_integ.iloc[:,19].values, axis=0)*100
# Add in date/depth columns for indexing:
chl_percent_integ.insert(0, 'Date', saanich_integ['Date'])
chl_percent_integ.insert(1, 'Depth', saanich_integ['Depth'])
#%% Particulates: Bin, reshape, interpolate & grid sampling over time
saanich_convert = saanich_data.iloc[:,[0,6,24,25,26,27,28,29]] # Particulates 
#------------------------------------------------------------------------------
# reset index:
saanich_convert.columns = saanich_convert.columns.droplevel(1)

# Replace duplecated headers:
saanich_convert.columns=['Date','Depth','Initial [bSiO2]', 'Final [bSiO2]','[PC] (ug/L)', '[PC] (umol/L)','[PN] (ug/L)', '[PN] (umol/L)']

# drop date header rows:
saanich_convert = saanich_convert.dropna(subset=['Date'])

# filter out outliers (replacing as min of range of outliers) for plotting, where
# outliers are defined as value-mean>2*SD:
outliers = np.where(np.abs(saanich_convert['Initial [bSiO2]']-np.mean(saanich_convert['Initial [bSiO2]'])>0.5*np.std(saanich_convert['Initial [bSiO2]'])))
bSi_min_outlier = np.min(saanich_convert['Initial [bSiO2]'].iloc[outliers[0]])
saanich_convert['Initial [bSiO2]'].iloc[outliers[0]] = bSi_min_outlier

outliers = np.where(np.abs(saanich_convert['[PC] (umol/L)']-np.mean(saanich_convert['[PC] (umol/L)'])>0.5*np.std(saanich_convert['[PC] (umol/L)'])))
POC_min_outlier = np.min(saanich_convert['[PC] (umol/L)'].iloc[outliers[0]])
saanich_convert['[PC] (umol/L)'].iloc[outliers[0]] = POC_min_outlier

outliers = np.where(np.abs(saanich_convert['[PN] (umol/L)']-np.mean(saanich_convert['[PN] (umol/L)'])>0.5*np.std(saanich_convert['[PN] (umol/L)'])))
PON_min_outlier = np.min(saanich_convert['[PN] (umol/L)'].iloc[outliers[0]])
saanich_convert['[PN] (umol/L)'].iloc[outliers[0]] = PON_min_outlier

# Create seperate dataframes by data clusters over time:
saanich_convert_2010 = saanich_convert.loc[247:306]
saanich_convert_2011 = saanich_convert.loc[308:359]
saanich_convert_2012_2013 = saanich_convert.loc[361:380]
saanich_convert_2014_2015 = saanich_convert.loc[382:474]
saanich_convert_2016_2017 = saanich_convert.loc[476:559]

# Redefine columns as float data type to be readable by binning functions:
saanich_convert_2010['Date'] = saanich_convert_2010['Date'].values.astype('float64') # need to convert datetimes first
saanich_convert_2011['Date'] = saanich_convert_2011['Date'].values.astype('float64') # need to convert datetimes first
saanich_convert_2012_2013['Date'] = saanich_convert_2012_2013['Date'].values.astype('float64') # need to convert datetimes first
saanich_convert_2014_2015['Date'] = saanich_convert_2014_2015['Date'].values.astype('float64') # need to convert datetimes first
saanich_convert_2016_2017['Date'] = saanich_convert_2016_2017['Date'].values.astype('float64') # need to convert datetimes first

# Define 1-m depth interval bins:
bins = np.arange(0,np.max(saanich_convert['Depth']),1)
# bins = np.arange(0,21,1) # restricted to interpolate only over upper 20-m depths

# Bin data as averages across 1-m bins by sampling date:
particulates_binned_2010 = saanich_convert_2010.groupby(['Date', pd.cut(saanich_convert_2010['Depth'], bins)]).mean()
particulates_binned_2011 = saanich_convert_2011.groupby(['Date', pd.cut(saanich_convert_2011['Depth'], bins)]).mean()
particulates_binned_2012_2013 = saanich_convert_2012_2013.groupby(['Date', pd.cut(saanich_convert_2012_2013['Depth'], bins)]).mean()
particulates_binned_2014_2015 = saanich_convert_2014_2015.groupby(['Date', pd.cut(saanich_convert_2014_2015['Depth'], bins)]).mean()
particulates_binned_2016_2017 = saanich_convert_2016_2017.groupby(['Date', pd.cut(saanich_convert_2016_2017['Depth'], bins)]).mean()

# Drop garbage data produced:
# particulates_binned = particulates_binned.drop(particulates_binned.index[0:217])

# Rename binned column:
particulates_binned_2010 = particulates_binned_2010.rename_axis(index=['Date', 'Depth Bins'])
particulates_binned_2011 = particulates_binned_2011.rename_axis(index=['Date', 'Depth Bins'])
particulates_binned_2012_2013 = particulates_binned_2012_2013.rename_axis(index=['Date', 'Depth Bins'])
particulates_binned_2014_2015 = particulates_binned_2014_2015.rename_axis(index=['Date', 'Depth Bins'])
particulates_binned_2016_2017 = particulates_binned_2016_2017.rename_axis(index=['Date', 'Depth Bins'])

# Transform dates back from integers to datetime numbers:
particulates_binned_2010.reset_index(inplace=True) # remove index specification on columns
particulates_binned_2011.reset_index(inplace=True) # remove index specification on columns
particulates_binned_2012_2013.reset_index(inplace=True) # remove index specification on columns
particulates_binned_2014_2015.reset_index(inplace=True) # remove index specification on columns
particulates_binned_2016_2017.reset_index(inplace=True) # remove index specification on columns

particulates_binned_2010['Date'] = pd.to_datetime(particulates_binned_2010['Date'],format=None)
particulates_binned_2011['Date'] = pd.to_datetime(particulates_binned_2011['Date'],format=None)
particulates_binned_2012_2013['Date'] = pd.to_datetime(particulates_binned_2012_2013['Date'],format=None)
particulates_binned_2014_2015['Date'] = pd.to_datetime(particulates_binned_2014_2015['Date'],format=None)
particulates_binned_2016_2017['Date'] = pd.to_datetime(particulates_binned_2016_2017['Date'],format=None)

# Redefine depth data as the edge values of bins rather than averages:
array_rep_size1 = np.array(np.size(particulates_binned_2010['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
array_rep_size2 = np.array(np.size(particulates_binned_2011['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
array_rep_size3 = np.array(np.size(particulates_binned_2012_2013['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
array_rep_size4 = np.array(np.size(particulates_binned_2014_2015['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
array_rep_size5 = np.array(np.size(particulates_binned_2016_2017['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
new_values = bins[:-1]+1
particulates_binned_2010['Depth'] = np.tile(new_values,array_rep_size1)
particulates_binned_2011['Depth'] = np.tile(new_values,array_rep_size2)
particulates_binned_2012_2013['Depth'] = np.tile(new_values,array_rep_size3)
particulates_binned_2014_2015['Depth'] = np.tile(new_values,array_rep_size4)
particulates_binned_2016_2017['Depth'] = np.tile(new_values,array_rep_size5)
#------------------------------------------------------------------------------
#### Reshape data from long list format to a 'date x depth' matrix:
# # bSi
# bSi_wide = particulates_binned.iloc[:,[0,2,3]].pivot(index='Date', columns='Depth').T
# bSi_wide = bSi_wide.rename_axis(index=['Temp', 'Depth']).reset_index(drop=True, level='Temp')
# # POC
# POC_wide = particulates_binned.iloc[:,[0,2,6]].pivot(index='Date', columns='Depth').T
# POC_wide = POC_wide.rename_axis(index=['Temp', 'Depth']).reset_index(drop=True, level='Temp')
# # PON
# PON_wide = particulates_binned.iloc[:,[0,2,8]].pivot(index='Date', columns='Depth').T
# PON_wide = PON_wide.rename_axis(index=['Temp', 'Depth']).reset_index(drop=True, level='Temp')

#------------------------------------------------------------------------------
#### Interpolate the data to fill in NaNs:
#### 2010
total_depth = np.max(particulates_binned_2010['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(particulates_binned_2010['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

# bSi
indF1 = particulates_binned_2010.loc[:,'Initial [bSiO2]'].notna()
date_pts = particulates_binned_2010['Date'][indF1].values
depth_pts = particulates_binned_2010['Depth'][indF1].values
values1 = particulates_binned_2010.loc[:,'Initial [bSiO2]'][indF1].values

bSi_wide_interp_2010 = scipy.interpolate.griddata((date_pts,depth_pts), values1, (xi,yi), rescale='QJ', method='linear')
# bSi_wide_interp_2010[bSi_wide_interp_2010<0] = np.nan # mask negative points

# POC
indF2 = particulates_binned_2010.loc[:,'[PC] (umol/L)'].notna() 
date_pts = particulates_binned_2010['Date'][indF2].values
depth_pts = particulates_binned_2010['Depth'][indF2].values
values2 = particulates_binned_2010.loc[:,'[PC] (umol/L)'][indF2].values

POC_wide_interp_2010 = scipy.interpolate.griddata((date_pts,depth_pts), values2, (xi,yi), rescale='QJ', method='linear')
# POC_wide_interp_2010[POC_wide_interp_2010<0] = np.nan # mask negative points

# PON
indF3 = particulates_binned_2010.loc[:,'[PN] (umol/L)'].notna() 
date_pts = particulates_binned_2010['Date'][indF3].values
depth_pts = particulates_binned_2010['Depth'][indF3].values
values3 = particulates_binned_2010.loc[:,'[PN] (umol/L)'][indF3].values

PON_wide_interp_2010 = scipy.interpolate.griddata((date_pts,depth_pts), values3, (xi,yi), rescale='QJ', method='linear')
# PON_wide_interp_2010[PON_wide_2010_interp<0] = np.nan # mask invalid points
#------------------------------------------------------------------------------
#### 2011
total_depth = np.max(particulates_binned_2011['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(particulates_binned_2011['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

# bSi
indF1 = particulates_binned_2011.loc[:,'Initial [bSiO2]'].notna()
date_pts = particulates_binned_2011['Date'][indF1].values
depth_pts = particulates_binned_2011['Depth'][indF1].values
values1 = particulates_binned_2011.loc[:,'Initial [bSiO2]'][indF1].values

bSi_wide_interp_2011 = scipy.interpolate.griddata((date_pts,depth_pts), values1, (xi,yi), rescale='QJ', method='linear')
# bSi_wide_interp_2011[bSi_wide_interp_2011<0] = np.nan # mask negative points

# POC
indF2 = particulates_binned_2011.loc[:,'[PC] (umol/L)'].notna() 
date_pts = particulates_binned_2011['Date'][indF2].values
depth_pts = particulates_binned_2011['Depth'][indF2].values
values2 = particulates_binned_2011.loc[:,'[PC] (umol/L)'][indF2].values

POC_wide_interp_2011 = scipy.interpolate.griddata((date_pts,depth_pts), values2, (xi,yi), rescale='QJ', method='linear')
# POC_wide_interp_2011[POC_wide_interp_2011<0] = np.nan # mask negative points

# PON
indF3 = particulates_binned_2011.loc[:,'[PN] (umol/L)'].notna() 
date_pts = particulates_binned_2011['Date'][indF3].values
depth_pts = particulates_binned_2011['Depth'][indF3].values
values3 = particulates_binned_2011.loc[:,'[PN] (umol/L)'][indF3].values

PON_wide_interp_2011 = scipy.interpolate.griddata((date_pts,depth_pts), values3, (xi,yi), rescale='QJ', method='linear')
# PON_wide_interp_2011[PON_wide_2011_interp<0] = np.nan # mask invalid points
#------------------------------------------------------------------------------
#### 2012-2013
total_depth = np.max(particulates_binned_2012_2013['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(particulates_binned_2012_2013['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

# bSi
indF1 = particulates_binned_2012_2013.loc[:,'Initial [bSiO2]'].notna()
date_pts = particulates_binned_2012_2013['Date'][indF1].values
depth_pts = particulates_binned_2012_2013['Depth'][indF1].values
values1 = particulates_binned_2012_2013.loc[:,'Initial [bSiO2]'][indF1].values

bSi_wide_interp_2012_2013 = scipy.interpolate.griddata((date_pts,depth_pts), values1, (xi,yi), rescale='QJ', method='linear')
# bSi_wide_interp_2012_2013[bSi_wide_interp_2012_2013<0] = np.nan # mask negative points

# POC
indF2 = particulates_binned_2012_2013.loc[:,'[PC] (umol/L)'].notna() 
date_pts = particulates_binned_2012_2013['Date'][indF2].values
depth_pts = particulates_binned_2012_2013['Depth'][indF2].values
values2 = particulates_binned_2012_2013.loc[:,'[PC] (umol/L)'][indF2].values

POC_wide_interp_2012_2013 = scipy.interpolate.griddata((date_pts,depth_pts), values2, (xi,yi), rescale='QJ', method='linear')
# POC_wide_interp_2012_2013[POC_wide_interp_2012_2013<0] = np.nan # mask negative points

# PON
indF3 = particulates_binned_2012_2013.loc[:,'[PN] (umol/L)'].notna() 
date_pts = particulates_binned_2012_2013['Date'][indF3].values
depth_pts = particulates_binned_2012_2013['Depth'][indF3].values
values3 = particulates_binned_2012_2013.loc[:,'[PN] (umol/L)'][indF3].values

PON_wide_interp_2012_2013 = scipy.interpolate.griddata((date_pts,depth_pts), values3, (xi,yi), rescale='QJ', method='linear')
# PON_wide_interp_2012_2013[PON_wide_2012_2013_interp<0] = np.nan # mask invalid points
#------------------------------------------------------------------------------
#### 2014-2015
total_depth = np.max(particulates_binned_2014_2015['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(particulates_binned_2014_2015['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

# bSi
indF1 = particulates_binned_2014_2015.loc[:,'Initial [bSiO2]'].notna()
date_pts = particulates_binned_2014_2015['Date'][indF1].values
depth_pts = particulates_binned_2014_2015['Depth'][indF1].values
values1 = particulates_binned_2014_2015.loc[:,'Initial [bSiO2]'][indF1].values

bSi_wide_interp_2014_2015 = scipy.interpolate.griddata((date_pts,depth_pts), values1, (xi,yi), rescale='QJ', method='linear')
# bSi_wide_interp_2014_2015[bSi_wide_interp_2014_2015<0] = np.nan # mask negative points

# POC
indF2 = particulates_binned_2014_2015.loc[:,'[PC] (umol/L)'].notna() 
date_pts = particulates_binned_2014_2015['Date'][indF2].values
depth_pts = particulates_binned_2014_2015['Depth'][indF2].values
values2 = particulates_binned_2014_2015.loc[:,'[PC] (umol/L)'][indF2].values

POC_wide_interp_2014_2015 = scipy.interpolate.griddata((date_pts,depth_pts), values2, (xi,yi), rescale='QJ', method='linear')
# POC_wide_interp_2014_2015[POC_wide_interp_2014_2015<0] = np.nan # mask negative points

# PON
indF3 = particulates_binned_2014_2015.loc[:,'[PN] (umol/L)'].notna() 
date_pts = particulates_binned_2014_2015['Date'][indF3].values
depth_pts = particulates_binned_2014_2015['Depth'][indF3].values
values3 = particulates_binned_2014_2015.loc[:,'[PN] (umol/L)'][indF3].values

PON_wide_interp_2014_2015 = scipy.interpolate.griddata((date_pts,depth_pts), values3, (xi,yi), rescale='QJ', method='linear')
# PON_wide_interp_2014_2015[PON_wide_2014_2015_interp<0] = np.nan # mask invalid points
#------------------------------------------------------------------------------
#### 2016-2017
total_depth = np.max(particulates_binned_2016_2017['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(particulates_binned_2016_2017['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

# bSi
indF1 = particulates_binned_2016_2017.loc[:,'Initial [bSiO2]'].notna()
date_pts = particulates_binned_2016_2017['Date'][indF1].values
depth_pts = particulates_binned_2016_2017['Depth'][indF1].values
values1 = particulates_binned_2016_2017.loc[:,'Initial [bSiO2]'][indF1].values

bSi_wide_interp_2016_2017 = scipy.interpolate.griddata((date_pts,depth_pts), values1, (xi,yi), rescale='QJ', method='linear')
# bSi_wide_interp_2016_2017[bSi_wide_interp_2016_2017<0] = np.nan # mask negative points

# POC
indF2 = particulates_binned_2016_2017.loc[:,'[PC] (umol/L)'].notna() 
date_pts = particulates_binned_2016_2017['Date'][indF2].values
depth_pts = particulates_binned_2016_2017['Depth'][indF2].values
values2 = particulates_binned_2016_2017.loc[:,'[PC] (umol/L)'][indF2].values

POC_wide_interp_2016_2017 = scipy.interpolate.griddata((date_pts,depth_pts), values2, (xi,yi), rescale='QJ', method='linear')
# POC_wide_interp_2016_2017[POC_wide_interp_2016_2017<0] = np.nan # mask negative points

# PON
indF3 = particulates_binned_2016_2017.loc[:,'[PN] (umol/L)'].notna() 
date_pts = particulates_binned_2016_2017['Date'][indF3].values
depth_pts = particulates_binned_2016_2017['Depth'][indF3].values
values3 = particulates_binned_2016_2017.loc[:,'[PN] (umol/L)'][indF3].values

PON_wide_interp_2016_2017 = scipy.interpolate.griddata((date_pts,depth_pts), values3, (xi,yi), rescale='QJ', method='linear')
# PON_wide_interp_2016_2017[PON_wide_2016_2017_interp<0] = np.nan # mask invalid points
#------------------------------------------------------------------------------

#%% Production: Bin, reshape, interpolate & grid sampling over time
saanich_convert = saanich_data.iloc[:,[0,6,30,33,34,36,37]] # Production 
#------------------------------------------------------------------------------
# reset index:
saanich_convert.columns = saanich_convert.columns.droplevel(1)

# Replace duplecated headers:
saanich_convert.columns=['Date','Depth','rhoSi (umol/L)', 'rhoC (ug/L)','rhoC (umol/L)', 'rhoN (ug/L)', 'rhoN (umol/L)']

# drop date header rows:
saanich_convert = saanich_convert.dropna(subset=['Date'])

# filter out outliers (replacing as min of range of outliers) for plotting, where
# outliers are defined as value-mean>2*SD:
outliers = np.where(np.abs(saanich_convert['rhoSi (umol/L)']-np.mean(saanich_convert['rhoSi (umol/L)'])>2*np.std(saanich_convert['rhoSi (umol/L)'])))
rhoSi_min_outlier = np.min(saanich_convert['rhoSi (umol/L)'].iloc[outliers[0]])
saanich_convert['rhoSi (umol/L)'].iloc[outliers[0]] = rhoSi_min_outlier

outliers = np.where(np.abs(saanich_convert['rhoC (umol/L)']-np.mean(saanich_convert['rhoC (umol/L)'])>0.25*np.std(saanich_convert['rhoC (umol/L)'])))
rhoC_min_outlier = np.min(saanich_convert['rhoC (umol/L)'].iloc[outliers[0]])
saanich_convert['rhoC (umol/L)'].iloc[outliers[0]] = rhoC_min_outlier

outliers = np.where(np.abs(saanich_convert['rhoN (umol/L)']-np.mean(saanich_convert['rhoN (umol/L)'])>0.25*np.std(saanich_convert['rhoN (umol/L)'])))
rhoN_min_outlier = np.min(saanich_convert['rhoN (umol/L)'].iloc[outliers[0]])
saanich_convert['rhoN (umol/L)'].iloc[outliers[0]] = rhoN_min_outlier

# Create seperate dataframes by data clusters over time:
saanich_convert_2010 = saanich_convert.loc[247:306]
saanich_convert_2011 = saanich_convert.loc[308:359]
saanich_convert_2012_2013 = saanich_convert.loc[361:380]
saanich_convert_2014_2015 = saanich_convert.loc[382:474]
saanich_convert_2016_2017 = saanich_convert.loc[476:559]

# Redefine columns as float data type to be readable by binning functions:
saanich_convert_2010['Date'] = saanich_convert_2010['Date'].values.astype('float64') # need to convert datetimes first
saanich_convert_2011['Date'] = saanich_convert_2011['Date'].values.astype('float64') # need to convert datetimes first
saanich_convert_2012_2013['Date'] = saanich_convert_2012_2013['Date'].values.astype('float64') # need to convert datetimes first
saanich_convert_2014_2015['Date'] = saanich_convert_2014_2015['Date'].values.astype('float64') # need to convert datetimes first
saanich_convert_2016_2017['Date'] = saanich_convert_2016_2017['Date'].values.astype('float64') # need to convert datetimes first

# Define 1-m depth interval bins:
bins = np.arange(0,np.max(saanich_convert['Depth']),1)
# bins = np.arange(0,21,1) # restricted to interpolate only over upper 20-m depths

# Bin data as averages across 1-m bins by sampling date:
production_binned_2010 = saanich_convert_2010.groupby(['Date', pd.cut(saanich_convert_2010['Depth'], bins)]).mean()
production_binned_2011 = saanich_convert_2011.groupby(['Date', pd.cut(saanich_convert_2011['Depth'], bins)]).mean()
production_binned_2012_2013 = saanich_convert_2012_2013.groupby(['Date', pd.cut(saanich_convert_2012_2013['Depth'], bins)]).mean()
production_binned_2014_2015 = saanich_convert_2014_2015.groupby(['Date', pd.cut(saanich_convert_2014_2015['Depth'], bins)]).mean()
production_binned_2016_2017 = saanich_convert_2016_2017.groupby(['Date', pd.cut(saanich_convert_2016_2017['Depth'], bins)]).mean()

# Drop garbage data produced:
# production_binned = production_binned.drop(production_binned.index[0:217])

# Rename binned column:
production_binned_2010 = production_binned_2010.rename_axis(index=['Date', 'Depth Bins'])
production_binned_2011 = production_binned_2011.rename_axis(index=['Date', 'Depth Bins'])
production_binned_2012_2013 = production_binned_2012_2013.rename_axis(index=['Date', 'Depth Bins'])
production_binned_2014_2015 = production_binned_2014_2015.rename_axis(index=['Date', 'Depth Bins'])
production_binned_2016_2017 = production_binned_2016_2017.rename_axis(index=['Date', 'Depth Bins'])

# Transform dates back from integers to datetime numbers:
production_binned_2010.reset_index(inplace=True) # remove index specification on columns
production_binned_2011.reset_index(inplace=True) # remove index specification on columns
production_binned_2012_2013.reset_index(inplace=True) # remove index specification on columns
production_binned_2014_2015.reset_index(inplace=True) # remove index specification on columns
production_binned_2016_2017.reset_index(inplace=True) # remove index specification on columns

production_binned_2010['Date'] = pd.to_datetime(production_binned_2010['Date'],format=None)
production_binned_2011['Date'] = pd.to_datetime(production_binned_2011['Date'],format=None)
production_binned_2012_2013['Date'] = pd.to_datetime(production_binned_2012_2013['Date'],format=None)
production_binned_2014_2015['Date'] = pd.to_datetime(production_binned_2014_2015['Date'],format=None)
production_binned_2016_2017['Date'] = pd.to_datetime(production_binned_2016_2017['Date'],format=None)

# Redefine depth data as the edge values of bins rather than averages:
array_rep_size1 = np.array(np.size(production_binned_2010['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
array_rep_size2 = np.array(np.size(production_binned_2011['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
array_rep_size3 = np.array(np.size(production_binned_2012_2013['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
array_rep_size4 = np.array(np.size(production_binned_2014_2015['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
array_rep_size5 = np.array(np.size(production_binned_2016_2017['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
new_values = bins[:-1]+1
production_binned_2010['Depth'] = np.tile(new_values,array_rep_size1)
production_binned_2011['Depth'] = np.tile(new_values,array_rep_size2)
production_binned_2012_2013['Depth'] = np.tile(new_values,array_rep_size3)
production_binned_2014_2015['Depth'] = np.tile(new_values,array_rep_size4)
production_binned_2016_2017['Depth'] = np.tile(new_values,array_rep_size5)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#### Interpolate the data to fill in NaNs:
#### 2010
total_depth = np.max(production_binned_2010['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(production_binned_2010['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

# # rhoSi
# indF1 = production_binned_2010.iloc[:,3].notna()
# date_pts = production_binned_2010['Date'][indF1].values
# depth_pts = production_binned_2010['Depth'][indF1].values
# values1 = production_binned_2010.iloc[:,3][indF1].values

# rhoSi_wide_interp_2010 = scipy.interpolate.griddata((date_pts,depth_pts), values1, (xi,yi), rescale='QJ', method='linear')
# rhoSi_wide_interp_2010[rhoSi_wide_interp_2010<0] = np.nan # mask negative points

# rhoC
indF2 = production_binned_2010.loc[:,'rhoC (umol/L)'].notna() 
date_pts = production_binned_2010['Date'][indF2].values
depth_pts = production_binned_2010['Depth'][indF2].values
values2 = production_binned_2010.loc[:,'rhoC (umol/L)'][indF2].values

rhoC_wide_interp_2010 = scipy.interpolate.griddata((date_pts,depth_pts), values2, (xi,yi), rescale='QJ', method='linear')
# rhoC_wide_interp_2010[rhoC_wide_interp_2010<0] = np.nan # mask negative points
rhoC_wide_interp_2010[rhoC_wide_interp_2010<0] = 0 # mask negative points

# rhoN
indF3 = production_binned_2010.loc[:,'rhoN (umol/L)'].notna() 
date_pts = production_binned_2010['Date'][indF3].values
depth_pts = production_binned_2010['Depth'][indF3].values
values3 = production_binned_2010.loc[:,'rhoN (umol/L)'][indF3].values

rhoN_wide_interp_2010 = scipy.interpolate.griddata((date_pts,depth_pts), values3, (xi,yi), rescale='QJ', method='linear')
rhoN_wide_interp_2010[rhoN_wide_interp_2010<0] = np.nan # mask invalid points
#------------------------------------------------------------------------------
#### 2011
total_depth = np.max(production_binned_2011['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(production_binned_2011['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

# # rhoSi
# indF1 = production_binned_2011.iloc[:,3].notna()
# date_pts = production_binned_2011['Date'][indF1].values
# depth_pts = production_binned_2011['Depth'][indF1].values
# values1 = production_binned_2011.iloc[:,3][indF1].values

# rhoSi_wide_interp_2011 = scipy.interpolate.griddata((date_pts,depth_pts), values1, (xi,yi), rescale='QJ', method='linear')
# rhoSi_wide_interp_2011[rhoSi_wide_interp_2011<0] = np.nan # mask negative points

# rhoC
indF2 = production_binned_2011.loc[:,'rhoC (umol/L)'].notna() 
date_pts = production_binned_2011['Date'][indF2].values
depth_pts = production_binned_2011['Depth'][indF2].values
values2 = production_binned_2011.loc[:,'rhoC (umol/L)'][indF2].values

rhoC_wide_interp_2011 = scipy.interpolate.griddata((date_pts,depth_pts), values2, (xi,yi), rescale='QJ', method='linear')
# rhoC_wide_interp_2011[rhoC_wide_interp_2011<0] = np.nan # mask negative points
rhoC_wide_interp_2011[rhoC_wide_interp_2011<0] = 0 # mask negative points

# rhoN
indF3 = production_binned_2011.loc[:,'rhoN (umol/L)'].notna() 
date_pts = production_binned_2011['Date'][indF3].values
depth_pts = production_binned_2011['Depth'][indF3].values
values3 = production_binned_2011.loc[:,'rhoN (umol/L)'][indF3].values

rhoN_wide_interp_2011 = scipy.interpolate.griddata((date_pts,depth_pts), values3, (xi,yi), rescale='QJ', method='linear')
rhoN_wide_interp_2011[rhoN_wide_interp_2011<0] = np.nan # mask invalid points
#------------------------------------------------------------------------------
#### 2012-2013
total_depth = np.max(production_binned_2012_2013['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(production_binned_2012_2013['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

# rhoSi
indF1 = production_binned_2012_2013.loc[:,'rhoSi (umol/L)'].notna()
date_pts = production_binned_2012_2013['Date'][indF1].values
depth_pts = production_binned_2012_2013['Depth'][indF1].values
values1 = production_binned_2012_2013.loc[:,'rhoSi (umol/L)'][indF1].values

rhoSi_wide_interp_2012_2013 = scipy.interpolate.griddata((date_pts,depth_pts), values1, (xi,yi), rescale='QJ', method='linear')
# rhoSi_wide_interp_2012_2013[rhoSi_wide_interp_2012_2013<0] = np.nan # mask negative points

# rhoC
indF2 = production_binned_2012_2013.loc[:,'rhoC (umol/L)'].notna() 
date_pts = production_binned_2012_2013['Date'][indF2].values
depth_pts = production_binned_2012_2013['Depth'][indF2].values
values2 = production_binned_2012_2013.loc[:,'rhoC (umol/L)'][indF2].values

rhoC_wide_interp_2012_2013 = scipy.interpolate.griddata((date_pts,depth_pts), values2, (xi,yi), rescale='QJ', method='linear')
# rhoC_wide_interp_2012_2013[rhoC_wide_interp_2012_2013<0] = np.nan # mask negative points
rhoC_wide_interp_2012_2013[rhoC_wide_interp_2012_2013<0] = 0 # mask negative points

# rhoN
indF3 = production_binned_2012_2013.loc[:,'rhoN (umol/L)'].notna() 
date_pts = production_binned_2012_2013['Date'][indF3].values
depth_pts = production_binned_2012_2013['Depth'][indF3].values
values3 = production_binned_2012_2013.loc[:,'rhoN (umol/L)'][indF3].values

rhoN_wide_interp_2012_2013 = scipy.interpolate.griddata((date_pts,depth_pts), values3, (xi,yi), rescale='QJ', method='linear')
rhoN_wide_interp_2012_2013[rhoN_wide_interp_2012_2013<0] = np.nan # mask invalid points
#------------------------------------------------------------------------------
#### 2014-2015
total_depth = np.max(production_binned_2014_2015['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(production_binned_2014_2015['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

# rhoSi
indF1 = production_binned_2014_2015.loc[:,'rhoSi (umol/L)'].notna()
date_pts = production_binned_2014_2015['Date'][indF1].values
depth_pts = production_binned_2014_2015['Depth'][indF1].values
values1 = production_binned_2014_2015.loc[:,'rhoSi (umol/L)'][indF1].values

rhoSi_wide_interp_2014_2015 = scipy.interpolate.griddata((date_pts,depth_pts), values1, (xi,yi), rescale='QJ', method='linear')
# rhoSi_wide_interp_2014_2015[rhoSi_wide_interp_2014_2015<0] = np.nan # mask negative points

# rhoC
indF2 = production_binned_2014_2015.loc[:,'rhoC (umol/L)'].notna() 
date_pts = production_binned_2014_2015['Date'][indF2].values
depth_pts = production_binned_2014_2015['Depth'][indF2].values
values2 = production_binned_2014_2015.loc[:,'rhoC (umol/L)'][indF2].values

rhoC_wide_interp_2014_2015 = scipy.interpolate.griddata((date_pts,depth_pts), values2, (xi,yi), rescale='QJ', method='linear')
# rhoC_wide_interp_2014_2015[rhoC_wide_interp_2014_2015<0] = np.nan # mask negative points
rhoC_wide_interp_2014_2015[rhoC_wide_interp_2014_2015<0] = 0 # mask negative points

# rhoN
indF3 = production_binned_2014_2015.loc[:,'rhoN (umol/L)'].notna() 
date_pts = production_binned_2014_2015['Date'][indF3].values
depth_pts = production_binned_2014_2015['Depth'][indF3].values
values3 = production_binned_2014_2015.loc[:,'rhoN (umol/L)'][indF3].values

rhoN_wide_interp_2014_2015 = scipy.interpolate.griddata((date_pts,depth_pts), values3, (xi,yi), rescale='QJ', method='linear')
rhoN_wide_interp_2014_2015[rhoN_wide_interp_2014_2015<0] = np.nan # mask invalid points
#------------------------------------------------------------------------------
#### 2016-2017
total_depth = np.max(production_binned_2016_2017['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(production_binned_2016_2017['Date'].values),np.arange(0,total_depth,1)) # coordinate pts

# rhoSi
indF1 = production_binned_2016_2017.loc[:,'rhoSi (umol/L)'].notna()
date_pts = production_binned_2016_2017['Date'][indF1].values
depth_pts = production_binned_2016_2017['Depth'][indF1].values
values1 = production_binned_2016_2017.loc[:,'rhoSi (umol/L)'][indF1].values

rhoSi_wide_interp_2016_2017 = scipy.interpolate.griddata((date_pts,depth_pts), values1, (xi,yi), rescale='QJ', method='linear')
# rhoSi_wide_interp_2016_2017[rhoSi_wide_interp_2016_2017<0] = np.nan # mask negative points

# rhoC
indF2 = production_binned_2016_2017.loc[:,'rhoC (umol/L)'].notna() 
date_pts = production_binned_2016_2017['Date'][indF2].values
depth_pts = production_binned_2016_2017['Depth'][indF2].values
values2 = production_binned_2016_2017.loc[:,'rhoC (umol/L)'][indF2].values

rhoC_wide_interp_2016_2017 = scipy.interpolate.griddata((date_pts,depth_pts), values2, (xi,yi), rescale='QJ', method='linear')
# rhoC_wide_interp_2016_2017[rhoC_wide_interp_2016_2017<0] = np.nan # mask negative points
rhoC_wide_interp_2016_2017[rhoC_wide_interp_2016_2017<0] = 0 # mask negative points

# rhoN
indF3 = production_binned_2016_2017.loc[:,'rhoN (umol/L)'].notna() 
date_pts = production_binned_2016_2017['Date'][indF3].values
depth_pts = production_binned_2016_2017['Depth'][indF3].values
values3 = production_binned_2016_2017.loc[:,'rhoN (umol/L)'][indF3].values

rhoN_wide_interp_2016_2017 = scipy.interpolate.griddata((date_pts,depth_pts), values3, (xi,yi), rescale='QJ', method='linear')
rhoN_wide_interp_2016_2017[rhoN_wide_interp_2016_2017<0] = np.nan # mask invalid points
#------------------------------------------------------------------------------
#%% Seasonal Chl: Bin, reshape, interpolate & grid sampling of chl size fractions over time
chl = saanich_data.iloc[:,[0,6,15,16,17,18,20,21,22,23]]
#------------------------------------------------------------------------------
# reset index:
chl.columns = chl.columns.droplevel(1)

# Replace duplicated headers:
chl.columns=['Date','Depth','CHL-A (>0.7um)', 'CHL-A (0.7-2um)','CHL-A (2-5um)', 'CHL-A (0.7-5um)','CHL-A (5-20um)', 'CHL-A (>20um)', 'TOTAL CHL-A', 'CHECK Tot chl']

# filter out outliers (replacing as min of range of outliers) for plotting, where
# outliers are defined as value-mean>2*SD:
outliers = np.where(np.abs(chl['CHECK Tot chl']-np.mean(chl['CHECK Tot chl'])>2*np.std(chl['CHECK Tot chl'])))
chl_tot_min_outlier = np.min(chl['CHECK Tot chl'].iloc[outliers[0]])
chl['CHECK Tot chl'].iloc[outliers[0]] = chl_tot_min_outlier

# drop date header rows:
chl = chl.dropna(subset=['Date'])

# Insert month column:
chl.insert(1, "Month", saanich_convert['Date'].dt.month)

# Define 1-m depth interval bins:
bins = np.arange(0,np.max(chl['Depth']),1)

# Bin data as averages across 1-m bins by sampling date:
chl_binned = chl.groupby(['Month', pd.cut(chl['Depth'], bins)]).mean()

# Rename binned column:
chl_binned = chl_binned.rename_axis(index=['Month', 'Depth Bins'])

# Transform dates back from integers to datetime numbers:
chl_binned.reset_index(inplace=True) # remove index specification on columns

# Redefine depth data as the edge values of bins rather than averages:
array_rep_size = np.array(np.size(chl_binned['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
new_values = bins[:-1]+1
chl_binned['Depth'] = np.tile(new_values,array_rep_size)
#------------------------------------------------------------------------------

#### Interpolate the data to fill in NaNs:
total_depth = np.max(chl_binned['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(chl_binned['Month'].values),np.arange(0,total_depth,1)) # coordinate pts

# total
indF6 = chl_binned.loc[:,'CHECK Tot chl'].notna() 
date_pts = chl_binned['Month'][indF6].values
depth_pts = chl_binned['Depth'][indF6].values
values6 = chl_binned.loc[:,'CHECK Tot chl'][indF6].values

chl_tot_seas_wide_interp = scipy.interpolate.griddata((date_pts,depth_pts), values6, (xi,yi), rescale='QJ', method='linear')

#------------------------------------------------------------------------------
#%% Seasonal Nuts: Bin, reshape, interpolate & grid sampling by averaging by month across time
saanich_convert = saanich_data.iloc[:,[0,6,11,12,13,14]] # Nuts
#------------------------------------------------------------------------------
# reset index:
saanich_convert.columns = saanich_convert.columns.droplevel(1)

# # filter out outliers (replacing as min of range of outliers) for plotting, where
# # outliers are defined as value-mean>2*SD:
# outliers = np.where(np.abs(saanich_convert['[NO3-]']-np.mean(saanich_convert['[NO3-]'])>2*np.std(saanich_convert['[NO3-]'])))
# NO3_min_outlier = np.min(saanich_convert['[NO3-]'].iloc[outliers[0]])
# saanich_convert['[NO3-]'].iloc[outliers[0]] = NO3_min_outlier

# outliers = np.where(np.abs(saanich_convert['[NO2]']-np.mean(saanich_convert['[NO2]'])>2*np.std(saanich_convert['[NO2]'])))
# NO2_min_outlier = np.min(saanich_convert['[NO2]'].iloc[outliers[0]])
# saanich_convert['[NO2]'].iloc[outliers[0]] = NO2_min_outlier

# outliers = np.where(np.abs(saanich_convert['[PO43-]']-np.mean(saanich_convert['[PO43-]'])>2*np.std(saanich_convert['[PO43-]'])))
# PO4_min_outlier = np.min(saanich_convert['[PO43-]'].iloc[outliers[0]])
# saanich_convert['[PO43-]'].iloc[outliers[0]] = PO4_min_outlier

# outliers = np.where(np.abs(saanich_convert['[Si(OH)4]']-np.mean(saanich_convert['[Si(OH)4]'])>2*np.std(saanich_convert['[Si(OH)4]'])))
# SiOH4_min_outlier = np.min(saanich_convert['[Si(OH)4]'].iloc[outliers[0]])
# saanich_convert['[Si(OH)4]'].iloc[outliers[0]] = SiOH4_min_outlier

# drop date header rows:
saanich_convert = saanich_convert.dropna(subset=['Date'])

# Insert month column:
saanich_convert.insert(1, "Month", saanich_convert['Date'].dt.month)

# Define 1-m depth interval bins:
bins = np.arange(0,np.max(saanich_convert['Depth']),1)

# Bin data as averages across 1-m bins by sampling date:
nuts_binned = saanich_convert.groupby(['Month', pd.cut(saanich_convert['Depth'], bins)]).mean()

# Rename binned column:
nuts_binned = nuts_binned.rename_axis(index=['Month', 'Depth Bins'])

# Transform dates back from integers to datetime numbers:
nuts_binned.reset_index(inplace=True) # remove index specification on columns
# nuts_binned['Date'] = pd.to_datetime(nuts_binned['Date'],format=None)

# Redefine depth data as the edge values of bins rather than averages:
array_rep_size = np.array(np.size(nuts_binned['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
new_values = bins[:-1]+1
nuts_binned['Depth'] = np.tile(new_values,array_rep_size)
#------------------------------------------------------------------------------
#### Interpolate the data to fill in NaNs:
total_depth = np.max(nuts_binned['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(nuts_binned['Month'].values),np.arange(0,total_depth,1)) # coordinate pts
tolerance = 1e16

# NO3
indF1 = nuts_binned.loc[:,'[NO3-]'].notna()
date_pts = nuts_binned['Month'][indF1].values
depth_pts = nuts_binned['Depth'][indF1].values
values1 = nuts_binned.loc[:,'[NO3-]'][indF1].values

NO3_seas_wide_interp = scipy.interpolate.griddata((date_pts,depth_pts), values1, (xi,yi), rescale='QJ', method='linear')
# NO3_wide_interp[NO3_wide_interp<0] = np.nan # mask negative points

# NO2
indF2 = nuts_binned.loc[:,'[NO2]'].notna() 
date_pts = nuts_binned['Month'][indF2].values
depth_pts = nuts_binned['Depth'][indF2].values
values2 = nuts_binned.loc[:,'[NO2]'][indF2].values

NO2_seas_wide_interp = scipy.interpolate.griddata((date_pts,depth_pts), values2, (xi,yi), rescale='QJ', method='linear')
# NO2_wide_interp[NO2_wide_interp<0] = np.nan # mask negative points

# PO4
indF3 = nuts_binned.loc[:,'[PO43-]'].notna() 
date_pts = nuts_binned['Month'][indF3].values
depth_pts = nuts_binned['Depth'][indF3].values
values3 = nuts_binned.loc[:,'[PO43-]'][indF3].values

PO4_seas_wide_interp = scipy.interpolate.griddata((date_pts,depth_pts), values3, (xi,yi), rescale='QJ', method='linear')
# PO4_wide_interp[PO4_wide_interp<0] = np.nan # mask invalid points

# Si(OH)4
indF4 = nuts_binned.loc[:,'[Si(OH)4]'].notna() 
date_pts = nuts_binned['Month'][indF4].values
depth_pts = nuts_binned['Depth'][indF4].values
values4 = nuts_binned.loc[:,'[Si(OH)4]'][indF4].values

Si_seas_wide_interp = scipy.interpolate.griddata((date_pts,depth_pts), values4, (xi,yi), rescale='QJ', method='linear')
# Si_wide_interp[Si_wide_interp<0] = np.nan # mask invalid points
#%% Seasonal Particulates: Bin, reshape, interpolate & grid sampling over time
saanich_convert = saanich_data.iloc[:,[0,6,24,25,26,27,28,29]] # Particulates 
#------------------------------------------------------------------------------
# reset index:
saanich_convert.columns = saanich_convert.columns.droplevel(1)

# Replace duplecated headers:
saanich_convert.columns=['Date','Depth','Initial [bSiO2]', 'Final [bSiO2]','[PC] (ug/L)', '[PC] (umol/L)','[PN] (ug/L)', '[PN] (umol/L)']

# filter out outliers (replacing as min of range of outliers) for plotting, where
# outliers are defined as value-mean>2*SD:
outliers = np.where(np.abs(saanich_convert['Initial [bSiO2]']-np.mean(saanich_convert['Initial [bSiO2]'])>0.5*np.std(saanich_convert['Initial [bSiO2]'])))
bSi_min_outlier = np.min(saanich_convert['Initial [bSiO2]'].iloc[outliers[0]])
saanich_convert['Initial [bSiO2]'].iloc[outliers[0]] = bSi_min_outlier

outliers = np.where(np.abs(saanich_convert['[PC] (umol/L)']-np.mean(saanich_convert['[PC] (umol/L)'])>0.5*np.std(saanich_convert['[PC] (umol/L)'])))
POC_min_outlier = np.min(saanich_convert['[PC] (umol/L)'].iloc[outliers[0]])
saanich_convert['[PC] (umol/L)'].iloc[outliers[0]] = POC_min_outlier

outliers = np.where(np.abs(saanich_convert['[PN] (umol/L)']-np.mean(saanich_convert['[PN] (umol/L)'])>0.5*np.std(saanich_convert['[PN] (umol/L)'])))
PON_min_outlier = np.min(saanich_convert['[PN] (umol/L)'].iloc[outliers[0]])
saanich_convert['[PN] (umol/L)'].iloc[outliers[0]] = PON_min_outlier

# drop date header rows:
saanich_convert = saanich_convert.dropna(subset=['Date'])

# Insert month column:
saanich_convert.insert(1, "Month", saanich_convert['Date'].dt.month)

# Define 1-m depth interval bins:
bins = np.arange(0,np.max(saanich_convert['Depth']),1)

# Bin data as averages across 1-m bins by sampling date:
particulates_binned = saanich_convert.groupby(['Month', pd.cut(saanich_convert['Depth'], bins)]).mean()

# Rename binned column:
particulates_binned = particulates_binned.rename_axis(index=['Month', 'Depth Bins'])

# Transform dates back from integers to datetime numbers:
particulates_binned.reset_index(inplace=True) # remove index specification on columns

# Redefine depth data as the edge values of bins rather than averages:
array_rep_size = np.array(np.size(particulates_binned['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
new_values = bins[:-1]+1
particulates_binned['Depth'] = np.tile(new_values,array_rep_size)
#------------------------------------------------------------------------------
#### Interpolate the data to fill in NaNs:
total_depth = np.max(particulates_binned['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(particulates_binned['Month'].values),np.arange(0,total_depth,1)) # coordinate pts
tolerance = 1e17

# bSi
indF1 = particulates_binned.iloc[:,3].notna()
date_pts = particulates_binned['Month'][indF1].values
depth_pts = particulates_binned['Depth'][indF1].values
values1 = particulates_binned.iloc[:,3][indF1].values

bSi_seas_wide_interp = scipy.interpolate.griddata((date_pts,depth_pts), values1, (xi,yi), rescale='QJ', method='linear')
# bSi_wide_interp[bSi_wide_interp<0] = np.nan # mask negative points

# POC
indF2 = particulates_binned.iloc[:,6].notna() 
date_pts = particulates_binned['Month'][indF2].values
depth_pts = particulates_binned['Depth'][indF2].values
values2 = particulates_binned.iloc[:,6][indF2].values

POC_seas_wide_interp = scipy.interpolate.griddata((date_pts,depth_pts), values2, (xi,yi), rescale='QJ', method='linear')
# POC_wide_interp[POC_wide_interp<0] = np.nan # mask negative points

# PON
indF3 = particulates_binned.iloc[:,8].notna() 
date_pts = particulates_binned['Month'][indF3].values
depth_pts = particulates_binned['Depth'][indF3].values
values3 = particulates_binned.iloc[:,8][indF3].values

PON_seas_wide_interp = scipy.interpolate.griddata((date_pts,depth_pts), values3, (xi,yi), rescale='QJ', method='linear')
# PON_wide_interp[PON_wide_interp<0] = np.nan # mask invalid points

#------------------------------------------------------------------------------
#%% Seasonal Production: Bin, reshape, interpolate & grid sampling over time
saanich_convert = saanich_data.iloc[:,[0,6,30,33,34,36,37]] # Production 
#------------------------------------------------------------------------------
# reset index:
saanich_convert.columns = saanich_convert.columns.droplevel(1)

# Replace duplecated headers:
saanich_convert.columns=['Date','Depth','rhoSi (umol/L)', 'rhoC (ug/L)','rhoC (umol/L)', 'rhoN (ug/L)', 'rhoN (umol/L)']

# drop date header rows:
saanich_convert = saanich_convert.dropna(subset=['Date'])

# filter out outliers (replacing as min of range of outliers) for plotting, where
# outliers are defined as value-mean>2*SD:
outliers = np.where(np.abs(saanich_convert['rhoSi (umol/L)']-np.mean(saanich_convert['rhoSi (umol/L)'])>2*np.std(saanich_convert['rhoSi (umol/L)'])))
rhoSi_min_outlier = np.min(saanich_convert['rhoSi (umol/L)'].iloc[outliers[0]])
saanich_convert['rhoSi (umol/L)'].iloc[outliers[0]] = rhoSi_min_outlier

outliers = np.where(np.abs(saanich_convert['rhoC (umol/L)']-np.mean(saanich_convert['rhoC (umol/L)'])>0.25*np.std(saanich_convert['rhoC (umol/L)'])))
rhoC_min_outlier = np.min(saanich_convert['rhoC (umol/L)'].iloc[outliers[0]])
saanich_convert['rhoC (umol/L)'].iloc[outliers[0]] = rhoC_min_outlier

outliers = np.where(np.abs(saanich_convert['rhoN (umol/L)']-np.mean(saanich_convert['rhoN (umol/L)'])>0.25*np.std(saanich_convert['rhoN (umol/L)'])))
rhoN_min_outlier = np.min(saanich_convert['rhoN (umol/L)'].iloc[outliers[0]])
saanich_convert['rhoN (umol/L)'].iloc[outliers[0]] = rhoN_min_outlier

# Insert month column:
saanich_convert.insert(1, "Month", saanich_convert['Date'].dt.month)

# Define 1-m depth interval bins:
bins = np.arange(0,np.max(saanich_convert['Depth']),1)

# Bin data as averages across 1-m bins by sampling date:
production_binned = saanich_convert.groupby(['Month', pd.cut(saanich_convert['Depth'], bins)]).mean()

# Rename binned column:
production_binned = production_binned.rename_axis(index=['Month', 'Depth Bins'])

# Transform dates back from integers to datetime numbers:
production_binned.reset_index(inplace=True) # remove index specification on columns

# Redefine depth data as the edge values of bins rather than averages:
array_rep_size = np.array(np.size(production_binned['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
new_values = bins[:-1]+1
production_binned['Depth'] = np.tile(new_values,array_rep_size)
#------------------------------------------------------------------------------
#### Interpolate the data to fill in NaNs:
total_depth = np.max(production_binned['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
xi,yi = np.meshgrid(np.unique(production_binned['Month'].values),np.arange(0,total_depth,1)) # coordinate pts
tolerance = 1e17

# rhoSi
indF1 = production_binned.iloc[:,3].notna()
date_pts = production_binned['Month'][indF1].values
depth_pts = production_binned['Depth'][indF1].values
values1 = production_binned.iloc[:,3][indF1].values

rhoSi_seas_wide_interp = scipy.interpolate.griddata((date_pts,depth_pts), values1, (xi,yi), rescale='QJ', method='linear')
rhoSi_seas_wide_interp[rhoSi_seas_wide_interp<0] = np.nan # mask negative points

# rhoC
indF2 = production_binned.iloc[:,5].notna() 
date_pts = production_binned['Month'][indF2].values
depth_pts = production_binned['Depth'][indF2].values
values2 = production_binned.iloc[:,5][indF2].values

rhoC_seas_wide_interp = scipy.interpolate.griddata((date_pts,depth_pts), values2, (xi,yi), rescale='QJ', method='linear')
# rhoC_seas_wide_interp[rhoC_seas_wide_interp<0] = np.nan # mask negative points
rhoC_seas_wide_interp[rhoC_seas_wide_interp<0] = 0 # mask negative points

# rhoN
indF3 = production_binned.iloc[:,7].notna() 
date_pts = production_binned['Month'][indF3].values
depth_pts = production_binned['Depth'][indF3].values
values3 = production_binned.iloc[:,7][indF3].values

rhoN_seas_wide_interp = scipy.interpolate.griddata((date_pts,depth_pts), values3, (xi,yi), rescale='QJ', method='linear')
rhoN_seas_wide_interp[rhoN_seas_wide_interp<0] = np.nan # mask invalid points

#------------------------------------------------------------------------------
#%% Nuts: Create new binned nutrients dataframe for timeline plot
saanich_convert = saanich_data.iloc[:,[0,6,11,12,13,14]] # Nuts
#------------------------------------------------------------------------------
# reset index:
saanich_convert.columns = saanich_convert.columns.droplevel(1)

# drop date header rows:
saanich_convert = saanich_convert.dropna(subset=['Date'])

# Replace non-integer values with zeros
saanich_convert['[NO3-]'] = saanich_convert['[NO3-]'].apply(lambda x: 0 if str(type(x))=="<class 'str'>" else x)
saanich_convert['[PO43-]'] = saanich_convert['[PO43-]'].apply(lambda x: 0 if str(type(x))=="<class 'str'>" else x)
saanich_convert['[Si(OH)4]'] = saanich_convert['[Si(OH)4]'].apply(lambda x: 0 if str(type(x))=="<class 'str'>" else x)

# Redefine columns as float data type to be readable by binning functions:
saanich_convert['Date'] = saanich_convert['Date'].values.astype('float64') # need to convert datetimes first

# Define 1-m depth interval bins:
bins = np.arange(0,np.max(saanich_convert['Depth']),1)

# Bin data as averages across 1-m bins by sampling date:
nuts_binned_new = saanich_convert.groupby(['Date', pd.cut(saanich_convert['Depth'], bins)]).mean()

# Drop garbage data produced:
# nuts_binned_new = nuts_binned_new.drop(nuts_binned_new.index[0:217])

# Rename binned column:
nuts_binned_new = nuts_binned_new.rename_axis(index=['Date', 'Depth Bins'])

# Transform dates back from integers to datetime numbers:
nuts_binned_new.reset_index(inplace=True) # remove index specification on columns
nuts_binned_new['Date'] = pd.to_datetime(nuts_binned_new['Date'],format=None)

# Redefine depth data as the edge values of bins rather than averages:
array_rep_size = np.array(np.size(nuts_binned_new['Depth'],0)/(np.size(bins,0)-1)).astype('int32')
new_values = bins[:-1]+1
nuts_binned_new['Depth'] = np.tile(new_values,array_rep_size)

x,y = np.meshgrid(np.unique(nuts_binned_new['Date']),np.arange(1,total_depth,1))

# NO3
NO3_wide = nuts_binned_new.iloc[:,[0,2,3]].pivot(index='Date', columns='Depth').T
NO3_wide = NO3_wide.rename_axis(index=['Temp', 'Depth']).reset_index(drop=True, level='Temp')

# NO3
x_condNO3 = x[pd.notna(NO3_wide)]
y_condNO3 = y[pd.notna(NO3_wide)]
#%% Nuts: contour plots
total_depth = np.max(nuts_binned_2010['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
#------------------------------------------------------------------------------
#### Set-up figure layout:
fig = plt.figure(figsize=(34,24), constrained_layout=True)
font={'family':'DejaVu Sans',
      'weight':'normal',
      'size':'22'} 
plt.rc('font', **font) # sets the specified font formatting globally

gs = fig.add_gridspec(6, 12)
# main plots
f_axx1 = fig.add_subplot(gs[0:2, :-3])
f_axx1.get_xaxis().set_ticks([])
f_axx1.get_yaxis().set_ticks([])

f_ax1 = f_axx1.inset_axes([0, 0, 1, 0.75])
f_ax1.spines['bottom'].set_visible(False)
# f_ax1.set_position([0.06403400326797382, 0.7093259490740739, 0.775, 0.2])

f_ax9 = f_axx1.inset_axes([0, 0.75, 1, 0.25])
f_ax9.spines['bottom'].set_visible(False)
f_ax9.get_xaxis().set_ticks([])
#------------------------------------------------------------------------------
f_axx2 = fig.add_subplot(gs[2:4, :-3])
f_axx2.get_xaxis().set_ticks([])
f_axx2.get_yaxis().set_ticks([])

f_ax2 = f_axx2.inset_axes([0, 0, 1, 0.75])
f_ax2.spines['bottom'].set_visible(False)
# f_ax2.set_position([0.06403400326797382, 0.3795111342592581, 0.775, 0.2])

f_ax10 = f_axx2.inset_axes([0, 0.75, 1, 0.25])
f_ax10.spines['bottom'].set_visible(False)
f_ax10.get_xaxis().set_ticks([])
#------------------------------------------------------------------------------
f_axx3 = fig.add_subplot(gs[4:6, :-3])
f_axx3.get_xaxis().set_ticks([])
f_axx3.get_yaxis().set_ticks([])

f_ax3 = f_axx3.inset_axes([0, 0, 1, 0.75])
f_ax3.spines['bottom'].set_visible(False)
# f_ax3.set_position([0.06403400326797372, 0.04969631944444486, 0.775, 0.2])

f_ax11 = f_axx3.inset_axes([0, 0.75, 1, 0.25])
f_ax11.spines['bottom'].set_visible(False)
f_ax11.get_xaxis().set_ticks([])
#------------------------------------------------------------------------------
# f_ax_del1 = fig.add_subplot(gs[0:2, 8:9])
# for axis in ['top','bottom','left','right']:
#     f_ax_del1.spines[axis].set_visible(False)
# f_ax_del1.get_xaxis().set_visible(False)
# f_ax_del1.get_yaxis().set_visible(False)

# f_ax_del2 = fig.add_subplot(gs[2:4, 8:9])
# for axis in ['top','bottom','left','right']:
#     f_ax_del2.spines[axis].set_visible(False)
# f_ax_del2.get_xaxis().set_visible(False)
# f_ax_del2.get_yaxis().set_visible(False)

# f_ax_del3 = fig.add_subplot(gs[4:6, 8:9])
# for axis in ['top','bottom','left','right']:
#     f_ax_del3.spines[axis].set_visible(False)
# f_ax_del3.get_xaxis().set_visible(False)
# f_ax_del3.get_yaxis().set_visible(False)

f_ax12 = fig.add_subplot(gs[0:2, 9:12])
f_ax13 = fig.add_subplot(gs[2:4, 9:12])
f_ax14 = fig.add_subplot(gs[4:6, 9:12])

#------------------------------------------------------------------------------
#### Plot the data:
#------------------------------------------------------------------------------
# contour_label = np.append(np.arange(0.1,0.5,0.2), np.arange(0.5,20,5))
contour_label = 5
all_vmin = 0
NO3_vmax = 30
PO4_vmax = 9
SiOH4_vmax = 55
colmap = cm.jet
#------------------------------------------------------------------------------
# NO3
# plot interpolations
h1 = f_ax1.contourf(np.unique(nuts_binned_2010['Date']),np.arange(0,total_depth,1),NO3_wide_interp_2010, 100, vmin=all_vmin, vmax=NO3_vmax, cmap=cm.jet)
h2 = f_ax1.contourf(np.unique(nuts_binned_2011['Date']),np.arange(0,total_depth,1),NO3_wide_interp_2011, 100, vmin=all_vmin, vmax=NO3_vmax, cmap=cm.jet)
# h3 = f_ax1.contourf(np.unique(nuts_binned_2012_2013['Date']),np.arange(0,total_depth,1),NO3_wide_interp_2012_2013, 100, vmin=all_vmin, vmax=NO3_vmax, cmap=cm.jet)
h4 = f_ax1.contourf(np.unique(nuts_binned_2014_2015['Date']),np.arange(0,total_depth,1),NO3_wide_interp_2014_2015, 100, vmin=all_vmin, vmax=NO3_vmax, cmap=cm.jet)
h5 = f_ax1.contourf(np.unique(nuts_binned_2016_2017['Date']),np.arange(0,total_depth,1),NO3_wide_interp_2016_2017, 100, vmin=all_vmin, vmax=NO3_vmax, cmap=cm.jet)

# plot sampling points:
# set marker size:
s = 100
# plot:
ind_grid = pd.notna(nuts_binned_2010['[NO3-]'])
f_ax1.scatter(nuts_binned_2010['Date'][ind_grid], nuts_binned_2010['Depth'][ind_grid], s, c=nuts_binned_2010['[NO3-]'][ind_grid], vmin=all_vmin, vmax=NO3_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(nuts_binned_2011['[NO3-]'])
f_ax1.scatter(nuts_binned_2011['Date'][ind_grid], nuts_binned_2011['Depth'][ind_grid], s, c=nuts_binned_2011['[NO3-]'][ind_grid], vmin=all_vmin, vmax=NO3_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(nuts_binned_2012_2013['[NO3-]'])
f_ax1.scatter(nuts_binned_2012_2013['Date'][ind_grid], nuts_binned_2012_2013['Depth'][ind_grid], s, c=nuts_binned_2012_2013['[NO3-]'][ind_grid], vmin=all_vmin, vmax=NO3_vmax, edgecolor='k', cmap=cm.jet)
ind_grid = pd.notna(nuts_binned_2014_2015['[NO3-]'])
f_ax1.scatter(nuts_binned_2014_2015['Date'][ind_grid], nuts_binned_2014_2015['Depth'][ind_grid], s, c=nuts_binned_2014_2015['[NO3-]'][ind_grid], vmin=all_vmin, vmax=NO3_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(nuts_binned_2016_2017['[NO3-]'])
f_ax1.scatter(nuts_binned_2016_2017['Date'][ind_grid], nuts_binned_2016_2017['Depth'][ind_grid], s, c=nuts_binned_2016_2017['[NO3-]'][ind_grid], vmin=all_vmin, vmax=NO3_vmax, edgecolor='w', cmap=cm.jet)

# Add contours:
contours1 = f_ax1.contour(np.unique(nuts_binned_2010['Date']),np.arange(0,total_depth,1),NO3_wide_interp_2010, contour_label, colors='black')
f_ax1.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours2 = f_ax1.contour(np.unique(nuts_binned_2011['Date']),np.arange(0,total_depth,1),NO3_wide_interp_2011, contour_label, colors='black')
f_ax1.clabel(contours2, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
# contours3 = f_ax1.contour(np.unique(nuts_binned_2012_2013['Date']),np.arange(0,total_depth,1),NO3_wide_interp_2012_2013, contour_label, colors='black')
# f_ax1.clabel(contours3, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours4 = f_ax1.contour(np.unique(nuts_binned_2014_2015['Date']),np.arange(0,total_depth,1),NO3_wide_interp_2014_2015, contour_label, colors='black')
f_ax1.clabel(contours4, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours5 = f_ax1.contour(np.unique(nuts_binned_2016_2017['Date']),np.arange(0,total_depth,1),NO3_wide_interp_2016_2017, contour_label, colors='black')
f_ax1.clabel(contours5, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

# Add formatting finishing touches:
# f_ax1.set_xlabel('Time (YYYY-MM)')
f_ax1.set_ylim(0,20)
f_ax1.set_ylabel('Depth (m)')
# cbar = fig.colorbar(h1, format="%.1f", ax=f_ax1)
# cbar.set_label(r'NO$_{3}$ ($\mathrm{\mu}$mol L$^{-1}$)')
f_ax1.invert_yaxis()
f_axx1.set_title(r' NO$_{\mathbf{3}}$+NO$_{\mathbf{2}}$', loc='left', fontweight='bold')

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

f_ax1.xaxis.set_major_locator(years)
f_ax1.xaxis.set_major_formatter(years_fmt)
f_ax1.xaxis.set_minor_locator(months)

datemin = np.datetime64(saanich_integ['Date'][saanich_integ.index[0]], 'Y')
datemax = np.datetime64(saanich_integ['Date'][saanich_integ.index[-1]], 'Y') + np.timedelta64(1, 'Y')
f_ax1.set_xlim(datemin, datemax)

f_ax1.yaxis.set_minor_locator(MultipleLocator(1))
f_ax1.yaxis.set_major_locator(MultipleLocator(5))
f_ax1.tick_params(axis='both', which='major', length=10)
f_ax1.tick_params(axis='both', which='minor', length=6)
f_ax1.tick_params(axis='y', which='both', right='True', labelleft='on')
#------------------------------------------------------------------------------
# PO4
h6 = f_ax2.contourf(np.unique(nuts_binned_2010['Date']),np.arange(0,total_depth,1),PO4_wide_interp_2010, 100, vmin=all_vmin, vmax=PO4_vmax, cmap=cm.jet)
h7 = f_ax2.contourf(np.unique(nuts_binned_2011['Date']),np.arange(0,total_depth,1),PO4_wide_interp_2011, 100, vmin=all_vmin, vmax=PO4_vmax, cmap=cm.jet)
# h8 = f_ax2.contourf(np.unique(nuts_binned_2012_2013['Date']),np.arange(0,total_depth,1),PO4_wide_interp_2012_2013, 100, vmin=all_vmin, vmax=PO4_vmax, cmap=cm.jet)
h9 = f_ax2.contourf(np.unique(nuts_binned_2014_2015['Date']),np.arange(0,total_depth,1),PO4_wide_interp_2014_2015, 100, vmin=all_vmin, vmax=PO4_vmax, cmap=cm.jet)
h10 = f_ax2.contourf(np.unique(nuts_binned_2016_2017['Date']),np.arange(0,total_depth,1),PO4_wide_interp_2016_2017, 100, vmin=all_vmin, vmax=PO4_vmax, cmap=cm.jet)

# plot sampling points:
# set marker size:
s = 100
# plot:
ind_grid = pd.notna(nuts_binned_2010['[PO43-]'])
s1 = f_ax2.scatter(nuts_binned_2010['Date'][ind_grid], nuts_binned_2010['Depth'][ind_grid], s, c=nuts_binned_2010['[PO43-]'][ind_grid], vmin=all_vmin, vmax=PO4_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(nuts_binned_2011['[PO43-]'])
s2 = f_ax2.scatter(nuts_binned_2011['Date'][ind_grid], nuts_binned_2011['Depth'][ind_grid], s, c=nuts_binned_2011['[PO43-]'][ind_grid], vmin=all_vmin, vmax=PO4_vmax,  edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(nuts_binned_2012_2013['[PO43-]'])
s3 = f_ax2.scatter(nuts_binned_2012_2013['Date'][ind_grid], nuts_binned_2012_2013['Depth'][ind_grid], s, c=nuts_binned_2012_2013['[PO43-]'][ind_grid], vmin=all_vmin, vmax=PO4_vmax,  edgecolor='k', cmap=cm.jet)
ind_grid = pd.notna(nuts_binned_2014_2015['[PO43-]'])
s4 = f_ax2.scatter(nuts_binned_2014_2015['Date'][ind_grid], nuts_binned_2014_2015['Depth'][ind_grid], s, c=nuts_binned_2014_2015['[PO43-]'][ind_grid], vmin=all_vmin, vmax=PO4_vmax,  edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(nuts_binned_2016_2017['[PO43-]'])
s5 = f_ax2.scatter(nuts_binned_2016_2017['Date'][ind_grid], nuts_binned_2016_2017['Depth'][ind_grid], s, c=nuts_binned_2016_2017['[PO43-]'][ind_grid], vmin=all_vmin, vmax=PO4_vmax,  edgecolor='w', cmap=cm.jet)

# Add contours:
contours1 = f_ax2.contour(np.unique(nuts_binned_2010['Date']),np.arange(0,total_depth,1),PO4_wide_interp_2010, contour_label, colors='black')
f_ax3.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours2 = f_ax2.contour(np.unique(nuts_binned_2011['Date']),np.arange(0,total_depth,1),PO4_wide_interp_2011, contour_label, colors='black')
f_ax3.clabel(contours2, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
# contours3 = f_ax2.contour(np.unique(nuts_binned_2012_2013['Date']),np.arange(0,total_depth,1),PO4_wide_interp_2012_2013, contour_label, colors='black')
# f_ax3.clabel(contours3, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours4 = f_ax2.contour(np.unique(nuts_binned_2014_2015['Date']),np.arange(0,total_depth,1),PO4_wide_interp_2014_2015, contour_label, colors='black')
f_ax3.clabel(contours4, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours5 = f_ax2.contour(np.unique(nuts_binned_2016_2017['Date']),np.arange(0,total_depth,1),PO4_wide_interp_2016_2017, contour_label, colors='black')
f_ax3.clabel(contours5, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

# Add formatting finishing touches:
# f_ax1.set_xlabel('Time (YYYY-MM)')
f_ax2.set_ylim(0,20)
f_ax2.set_ylabel('Depth (m)')
# cbar = fig.colorbar(h4, format="%.1f", ax=f_ax2)
# cbar.set_label(r'PO$_{4}$ ($\mathrm{\mu}$mol L$^{-1}$)')
f_ax2.invert_yaxis()
f_axx2.set_title(r' PO$_{\mathbf{4}}$', loc='left', fontweight='bold')
f_ax2.xaxis.set_major_locator(years)
f_ax2.xaxis.set_major_formatter(years_fmt)
f_ax2.xaxis.set_minor_locator(months)

datemin = np.datetime64(saanich_integ['Date'][saanich_integ.index[0]], 'Y')
datemax = np.datetime64(saanich_integ['Date'][saanich_integ.index[-1]], 'Y') + np.timedelta64(1, 'Y')
f_ax2.set_xlim(datemin, datemax)

f_ax2.yaxis.set_minor_locator(MultipleLocator(1))
f_ax2.yaxis.set_major_locator(MultipleLocator(5))
f_ax2.tick_params(axis='both', which='major', length=10)
f_ax2.tick_params(axis='both', which='minor', length=6)
f_ax2.tick_params(axis='y', which='both', right='True', labelleft='on')

# labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
# labels[-1] = '>'+str("{:.1f}".format(PO4_min_outlier))
# cbar.ax.set_yticklabels(labels)
#------------------------------------------------------------------------------
# Si(OH)4
h11 = f_ax3.contourf(np.unique(nuts_binned_2010['Date']),np.arange(0,total_depth,1), Si_wide_interp_2010, 100, vmin=all_vmin, vmax=SiOH4_vmax, cmap=cm.jet)
h12 = f_ax3.contourf(np.unique(nuts_binned_2011['Date']),np.arange(0,total_depth,1), Si_wide_interp_2011, 100, vmin=all_vmin, vmax=SiOH4_vmax, cmap=cm.jet)
# h13 = f_ax3.contourf(np.unique(nuts_binned_2012_2013['Date']),np.arange(0,total_depth,1), Si_wide_interp_2012_2013, 100, vmin=all_vmin, vmax=SiOH4_vmax, cmap=cm.jet)
h14 = f_ax3.contourf(np.unique(nuts_binned_2014_2015['Date']),np.arange(0,total_depth,1), Si_wide_interp_2014_2015, 100, vmin=all_vmin, vmax=SiOH4_vmax, cmap=cm.jet)
h15 = f_ax3.contourf(np.unique(nuts_binned_2016_2017['Date']),np.arange(0,total_depth,1), Si_wide_interp_2016_2017, 100, vmin=all_vmin, vmax=SiOH4_vmax, cmap=cm.jet)

# plot sampling points:
# set marker size:
s = 100
# plot:
ind_grid = pd.notna(nuts_binned_2010['[Si(OH)4]'])
s1 = f_ax3.scatter(nuts_binned_2010['Date'][ind_grid], nuts_binned_2010['Depth'][ind_grid], s, c=nuts_binned_2010['[Si(OH)4]'][ind_grid], vmin=all_vmin, vmax=SiOH4_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(nuts_binned_2011['[Si(OH)4]'])
s2 = f_ax3.scatter(nuts_binned_2011['Date'][ind_grid], nuts_binned_2011['Depth'][ind_grid], s, c=nuts_binned_2011['[Si(OH)4]'][ind_grid], vmin=all_vmin, vmax=SiOH4_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(nuts_binned_2012_2013['[Si(OH)4]'])
s3 = f_ax3.scatter(nuts_binned_2012_2013['Date'][ind_grid], nuts_binned_2012_2013['Depth'][ind_grid], s, c=nuts_binned_2012_2013['[Si(OH)4]'][ind_grid], vmin=all_vmin, vmax=SiOH4_vmax, edgecolor='k', cmap=cm.jet)
ind_grid = pd.notna(nuts_binned_2014_2015['[Si(OH)4]'])
s4 = f_ax3.scatter(nuts_binned_2014_2015['Date'][ind_grid], nuts_binned_2014_2015['Depth'][ind_grid], s, c=nuts_binned_2014_2015['[Si(OH)4]'][ind_grid], vmin=all_vmin, vmax=SiOH4_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(nuts_binned_2016_2017['[Si(OH)4]'])
s5 = f_ax3.scatter(nuts_binned_2016_2017['Date'][ind_grid], nuts_binned_2016_2017['Depth'][ind_grid], s, c=nuts_binned_2016_2017['[Si(OH)4]'][ind_grid], vmin=all_vmin, vmax=SiOH4_vmax, edgecolor='w', cmap=cm.jet)

# Add contours:
contours1 = f_ax3.contour(np.unique(nuts_binned_2010['Date']),np.arange(0,total_depth,1),Si_wide_interp_2010, contour_label, colors='black')
f_ax3.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours2 = f_ax3.contour(np.unique(nuts_binned_2011['Date']),np.arange(0,total_depth,1),Si_wide_interp_2011, contour_label, colors='black')
f_ax3.clabel(contours2, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
# contours3 = f_ax3.contour(np.unique(nuts_binned_2012_2013['Date']),np.arange(0,total_depth,1),Si_wide_interp_2012_2013, contour_label, colors='black')
# f_ax3.clabel(contours3, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours4 = f_ax3.contour(np.unique(nuts_binned_2014_2015['Date']),np.arange(0,total_depth,1),Si_wide_interp_2014_2015, contour_label, colors='black')
f_ax3.clabel(contours4, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours5 = f_ax3.contour(np.unique(nuts_binned_2016_2017['Date']),np.arange(0,total_depth,1),Si_wide_interp_2016_2017, contour_label, colors='black')
f_ax3.clabel(contours5, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

# Add formatting finishing touches:
# f_ax1.set_xlabel('Time (YYYY-MM)')
f_ax3.set_ylim(0,20)
f_ax3.set_ylabel('Depth (m)')
f_ax3.set_xlabel('Time (Years)')
# cbar = fig.colorbar(h4, format="%.1f", ax=f_ax3)
# cbar.set_label(r'Si(OH)$_{4}$ ($\mathrm{\mu}$mol L$^{-1}$)')
f_ax3.invert_yaxis()
f_axx3.set_title(r' Si(OH)$_{\mathbf{4}}$', loc='left', fontweight='bold')
f_ax3.xaxis.set_major_locator(years)
f_ax3.xaxis.set_major_formatter(years_fmt)
f_ax3.xaxis.set_minor_locator(months)

datemin = np.datetime64(saanich_integ['Date'][saanich_integ.index[0]], 'Y')
datemax = np.datetime64(saanich_integ['Date'][saanich_integ.index[-1]], 'Y') + np.timedelta64(1, 'Y')
f_ax3.set_xlim(datemin, datemax)

f_ax3.yaxis.set_minor_locator(MultipleLocator(1))
f_ax3.yaxis.set_major_locator(MultipleLocator(5))
f_ax3.tick_params(axis='both', which='major', length=10)
f_ax3.tick_params(axis='both', which='minor', length=6)
f_ax3.tick_params(axis='y', which='both', right='True', labelleft='on')

# labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
# labels[-1] = '>'+str("{:.1f}".format(SiOH4_min_outlier))
# cbar.ax.set_yticklabels(labels)
#------------------------------------------------------------------------------
#### Add integrated values with growing seasons
# NO3
ind1 = pd.notna(saanich_integ_2010['[NO3-]'])
ind2 = pd.notna(saanich_integ_2011['[NO3-]'])
ind3 = pd.notna(saanich_integ_2014_2015['[NO3-]'])
ind4 = pd.notna(saanich_integ_2016_2017['[NO3-]'])
f_ax9.plot(saanich_integ_2010['Date'][ind1], saanich_integ_2010['[NO3-]'][ind1], 'k--.', markersize=14,clip_on=False)
f_ax9.plot(saanich_integ_2011['Date'][ind2], saanich_integ_2011['[NO3-]'][ind2], 'k--.', markersize=14,clip_on=False)
f_ax9.plot(saanich_integ_2014_2015['Date'][ind3], saanich_integ_2014_2015['[NO3-]'][ind3], 'k--.', markersize=14,clip_on=False)
f_ax9.plot(saanich_integ_2016_2017['Date'][ind4], saanich_integ_2016_2017['[NO3-]'][ind4], 'k--.', markersize=14,clip_on=False)
f_ax9.set_ylabel('(mmol m$^{-2}$)')
# f_ax9.set_xlim(np.min(saanich_integ['Date']),np.max(saanich_integ['Date']))
f_ax9.set_xlim(datemin, datemax)
f_ax9.autoscale(enable=True, axis='y', tight=True)
f_ax9.tick_params(axis='y', which='major', length=10)
f_ax9.tick_params(axis='y', which='minor', length=6)
start, end = f_ax9.get_ylim()
f_ax9.yaxis.set_ticks(np.arange(start, end, 250))

color_code = 'dimgray'
f_ax9.axvspan(datetime.date(2010,3,1), datetime.date(2010,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9.axvspan(datetime.date(2011,3,1), datetime.date(2011,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9.axvspan(datetime.date(2012,3,1), datetime.date(2012,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9.axvspan(datetime.date(2013,3,1), datetime.date(2013,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9.axvspan(datetime.date(2014,3,1), datetime.date(2014,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9.axvspan(datetime.date(2015,3,1), datetime.date(2015,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9.axvspan(datetime.date(2016,3,1), datetime.date(2016,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9.axvspan(datetime.date(2017,3,1), datetime.date(2017,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)

f_ax9.tick_params(axis='y', which='both', right='True', labelleft='on')

# PO4
ind1 = pd.notna(saanich_integ_2010['[PO43-]'])
ind2 = pd.notna(saanich_integ_2011['[PO43-]'])
ind3 = pd.notna(saanich_integ_2014_2015['[PO43-]'])
ind4 = pd.notna(saanich_integ_2016_2017['[PO43-]'])
f_ax10.plot(saanich_integ_2010['Date'][ind1], saanich_integ_2010['[PO43-]'][ind1], 'k--.', markersize=14, clip_on=False)
f_ax10.plot(saanich_integ_2011['Date'][ind2], saanich_integ_2011['[PO43-]'][ind2], 'k--.', markersize=14, clip_on=False)
f_ax10.plot(saanich_integ_2014_2015['Date'][ind3], saanich_integ_2014_2015['[PO43-]'][ind3], 'k--.', markersize=14, clip_on=False)
f_ax10.plot(saanich_integ_2016_2017['Date'][ind4], saanich_integ_2016_2017['[PO43-]'][ind4], 'k--.', markersize=14, clip_on=False)
f_ax10.set_ylabel('(mmol m$^{-2}$)')
# f_ax10.set_xlim(np.min(saanich_integ['Date']),np.max(saanich_integ['Date']))
f_ax10.set_xlim(datemin, datemax)
f_ax10.autoscale(enable=True, axis='y', tight=True)
f_ax10.tick_params(axis='y', which='major', length=10)
f_ax10.tick_params(axis='y', which='minor', length=6)
start, end = f_ax10.get_ylim()
f_ax10.yaxis.set_ticks(np.arange(0, 301, 100))

f_ax10.axvspan(datetime.date(2010,3,1), datetime.date(2010,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2011,3,1), datetime.date(2011,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2012,3,1), datetime.date(2012,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2013,3,1), datetime.date(2013,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2014,3,1), datetime.date(2014,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2015,3,1), datetime.date(2015,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2016,3,1), datetime.date(2016,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2017,3,1), datetime.date(2017,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)

f_ax10.tick_params(axis='y', which='both', right='True', labelleft='on')

# Si(OH)4
ind1 = pd.notna(saanich_integ_2010['[Si(OH)4]'])
ind2 = pd.notna(saanich_integ_2011['[Si(OH)4]'])
ind3 = pd.notna(saanich_integ_2014_2015['[Si(OH)4]'])
ind4 = pd.notna(saanich_integ_2016_2017['[Si(OH)4]'])
f_ax11.plot(saanich_integ_2010['Date'][ind1], saanich_integ_2010['[Si(OH)4]'][ind1], 'k--.', markersize=14, clip_on=False)
f_ax11.plot(saanich_integ_2011['Date'][ind2], saanich_integ_2011['[Si(OH)4]'][ind2], 'k--.', markersize=14, clip_on=False)
f_ax11.plot(saanich_integ_2014_2015['Date'][ind3], saanich_integ_2014_2015['[Si(OH)4]'][ind3], 'k--.', markersize=14, clip_on=False)
f_ax11.plot(saanich_integ_2016_2017['Date'][ind4], saanich_integ_2016_2017['[Si(OH)4]'][ind4], 'k--.', markersize=14, clip_on=False)
f_ax11.set_ylabel('(mmol m$^{-2}$)')
# f_ax11.set_xlim(np.min(saanich_integ['Date']),np.max(saanich_integ['Date']))
f_ax11.set_xlim(datemin, datemax)
f_ax11.autoscale(enable=True, axis='y', tight=True)
f_ax11.tick_params(axis='y', which='major', length=10)
f_ax11.tick_params(axis='y', which='minor', length=6)

f_ax11.axvspan(datetime.date(2010,3,1), datetime.date(2010,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2011,3,1), datetime.date(2011,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2012,3,1), datetime.date(2012,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2013,3,1), datetime.date(2013,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2014,3,1), datetime.date(2014,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2015,3,1), datetime.date(2015,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2016,3,1), datetime.date(2016,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2017,3,1), datetime.date(2017,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)

f_ax11.tick_params(axis='y', which='both', right='True', labelleft='on')
#------------------------------------------------------------------------------
#### Plot seasonally averaged data:
total_depth = np.max(nuts_binned['Depth'])+1
# NO3
h16 = f_ax12.contourf(np.unique(nuts_binned['Month']),np.arange(0,total_depth,1),NO3_seas_wide_interp, 100, cmap=cm.jet)

ind_grid = pd.notna(nuts_binned['[NO3-]'])
s1 = f_ax12.scatter(nuts_binned['Month'][ind_grid], nuts_binned['Depth'][ind_grid], 100, c=nuts_binned['[NO3-]'][ind_grid], edgecolor='w', cmap=cm.jet, clip_on=False)

contours1 = f_ax12.contour(np.unique(nuts_binned['Month']),np.arange(0,total_depth,1),NO3_seas_wide_interp, contour_label, colors='black')
f_ax12.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

f_ax12.set_ylabel('Depth (m)')
# f_ax12.set_xlabel('Month')
f_ax12.set_xlim([1,12])
cbar = fig.colorbar(h16, ax=f_ax12, pad=-0.02, aspect=30)
cbar.set_label(r'($\mathrm{\mu}$mol L$^{-1}$)')
f_ax12.invert_yaxis()
f_ax12.set_title(r' NO$_{\mathbf{3}}$+NO$_{\mathbf{2}}$', loc='left', fontweight='bold')

f_ax12.set_xticks(np.arange(1,13,1))
f_ax12.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
f_ax12.yaxis.set_minor_locator(MultipleLocator(1))
f_ax12.tick_params(axis='both', which='major', length=10)
f_ax12.tick_params(axis='y', which='minor', length=6)
f_ax12.tick_params(axis='y', which='both', right='True', labelleft='on')

# PO4
h17 = f_ax13.contourf(np.unique(nuts_binned['Month']),np.arange(0,total_depth,1),PO4_seas_wide_interp, 100, cmap=cm.jet)

ind_grid = pd.notna(nuts_binned['[PO43-]'])
s1 = f_ax13.scatter(nuts_binned['Month'][ind_grid], nuts_binned['Depth'][ind_grid], 100, c=nuts_binned['[PO43-]'][ind_grid], edgecolor='w', cmap=cm.jet, clip_on=False)

contours1 = f_ax13.contour(np.unique(nuts_binned['Month']),np.arange(0,total_depth,1),PO4_seas_wide_interp, contour_label, colors='black')
f_ax13.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

f_ax13.set_ylabel('Depth (m)')
# f_ax13.set_xlabel('Month')
f_ax13.set_xlim([1,12])
cbar = fig.colorbar(h17, ax=f_ax13, pad=-0.02, aspect=30)
cbar.set_label(r'($\mathrm{\mu}$mol L$^{-1}$)')
f_ax13.invert_yaxis()
f_ax13.set_title(r' PO$_{\mathbf{4}}$', loc='left', fontweight='bold')

f_ax13.set_xticks(np.arange(1,13,1))
f_ax13.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
f_ax13.yaxis.set_minor_locator(MultipleLocator(1))
f_ax13.tick_params(axis='both', which='major', length=10)
f_ax13.tick_params(axis='y', which='minor', length=6)
f_ax13.tick_params(axis='y', which='both', right='True', labelleft='on')

# Si(OH)4
h18 = f_ax14.contourf(np.unique(nuts_binned['Month']),np.arange(0,total_depth,1),Si_seas_wide_interp, 100, cmap=cm.jet)

ind_grid = pd.notna(nuts_binned['[Si(OH)4]'])
s1 = f_ax14.scatter(nuts_binned['Month'][ind_grid], nuts_binned['Depth'][ind_grid], 100, c=nuts_binned['[Si(OH)4]'][ind_grid], edgecolor='w', cmap=cm.jet, clip_on=False)

contours1 = f_ax14.contour(np.unique(nuts_binned['Month']),np.arange(0,total_depth,1),Si_seas_wide_interp, contour_label, colors='black')
f_ax14.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

f_ax14.set_ylabel('Depth (m)')
f_ax14.set_xlabel('Month')
f_ax14.set_xlim([1,12])
cbar = fig.colorbar(h18, ax=f_ax14, pad=-0.02, aspect=30)
cbar.set_label(r'($\mathrm{\mu}$mol L$^{-1}$)')
f_ax14.invert_yaxis()
f_ax14.set_title(r' Si(OH)$_{\mathbf{4}}$', loc='left', fontweight='bold')

f_ax14.set_xticks(np.arange(1,13,1))
f_ax14.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
f_ax14.yaxis.set_minor_locator(MultipleLocator(1))
f_ax14.tick_params(axis='both', which='major', length=10)
f_ax14.tick_params(axis='y', which='minor', length=6)
f_ax14.tick_params(axis='y', which='both', right='True', labelleft='on')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### Collate figure handles for a vector graphic format:
h_nuts = [h1,h2,h4,h5,h6,h7,h9,h10,h11,h12,h14,h15,h16,h17,h18] # collate plot handles
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% Particulates: contour plots
total_depth = np.max(chl_binned_2010['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
#------------------------------------------------------------------------------
#### Set-up figure layout:
fig = plt.figure(figsize=(34,24), constrained_layout=True)
font={'family':'DejaVu Sans',
      'weight':'normal',
      'size':'22'} 
plt.rc('font', **font) # sets the specified font formatting globally

gs = fig.add_gridspec(8, 12)
# main plots
f_axx1 = fig.add_subplot(gs[0:2, :-3])
f_axx1.get_xaxis().set_ticks([])
f_axx1.get_yaxis().set_ticks([])

f_ax1 = f_axx1.inset_axes([0, 0, 1, 0.75])
f_ax1.spines['bottom'].set_visible(False)
# f_ax1.set_position([0.0655148039215687, 0.7872004279637013, 0.73, 0.14])

f_ax9 = f_axx1.inset_axes([0, 0.75, 1, 0.25])
f_ax9.spines['bottom'].set_visible(False)
f_ax9.get_xaxis().set_ticks([])
#------------------------------------------------------------------------------
f_axx2 = fig.add_subplot(gs[2:4, :-3])
f_axx2.get_xaxis().set_ticks([])
f_axx2.get_yaxis().set_ticks([])

f_ax2 = f_axx2.inset_axes([0, 0, 1, 0.75])
f_ax2.spines['bottom'].set_visible(False)
# f_ax2.set_position([0.0681700326797383, 0.5406122046764178, 0.73, 0.14])

f_ax10 = f_axx2.inset_axes([0, 0.75, 1, 0.25])
f_ax10.spines['bottom'].set_visible(False)
f_ax10.get_xaxis().set_ticks([])
#------------------------------------------------------------------------------
f_axx3 = fig.add_subplot(gs[4:6, :-3])
f_axx3.get_xaxis().set_ticks([])
f_axx3.get_yaxis().set_ticks([])

f_ax3 = f_axx3.inset_axes([0, 0, 1, 0.75])
f_ax3.spines['bottom'].set_visible(False)
# f_ax3.set_position([0.0681700326797384, 0.29515426206043044, 0.73, 0.14])

f_ax11 = f_axx3.inset_axes([0, 0.75, 1, 0.25])
f_ax11.spines['bottom'].set_visible(False)
f_ax11.get_xaxis().set_ticks([])
#------------------------------------------------------------------------------
f_axx4 = fig.add_subplot(gs[6:8, :-3])
f_axx4.get_xaxis().set_ticks([])
f_axx4.get_yaxis().set_ticks([])

f_ax4 = f_axx4.inset_axes([0, 0, 1, 0.75])
f_ax4.spines['bottom'].set_visible(False)
# f_ax4.set_position([0.0681700326797384, 0.04969631944444397, 0.73, 0.14])

f_ax12 = f_axx4.inset_axes([0, 0.75, 1, 0.25])
f_ax12.spines['bottom'].set_visible(False)
f_ax12.get_xaxis().set_ticks([])
#------------------------------------------------------------------------------
# f_ax_del1 = fig.add_subplot(gs[0:2, 8:9])
# f_ax_del1.spines['top'].set_visible(False)
# f_ax_del1.spines['bottom'].set_visible(False)
# f_ax_del1.spines['left'].set_visible(False)
# f_ax_del1.spines['right'].set_visible(False)
# f_ax_del1.get_xaxis().set_visible(False)
# f_ax_del1.get_yaxis().set_visible(False)

# f_ax_del2 = fig.add_subplot(gs[2:4, 8:9])
# f_ax_del2.spines['top'].set_visible(False)
# f_ax_del2.spines['bottom'].set_visible(False)
# f_ax_del2.spines['left'].set_visible(False)
# f_ax_del2.spines['right'].set_visible(False)
# f_ax_del2.get_xaxis().set_visible(False)
# f_ax_del2.get_yaxis().set_visible(False)

# f_ax_del3 = fig.add_subplot(gs[4:6, 8:9])
# f_ax_del3.spines['top'].set_visible(False)
# f_ax_del3.spines['bottom'].set_visible(False)
# f_ax_del3.spines['left'].set_visible(False)
# f_ax_del3.spines['right'].set_visible(False)
# f_ax_del3.get_xaxis().set_visible(False)
# f_ax_del3.get_yaxis().set_visible(False)

# f_ax_del4 = fig.add_subplot(gs[6:8, 8:9])
# f_ax_del4.spines['top'].set_visible(False)
# f_ax_del4.spines['bottom'].set_visible(False)
# f_ax_del4.spines['left'].set_visible(False)
# f_ax_del4.spines['right'].set_visible(False)
# f_ax_del4.get_xaxis().set_visible(False)
# f_ax_del4.get_yaxis().set_visible(False)

f_ax13 = fig.add_subplot(gs[0:2, 9:12])
f_ax14 = fig.add_subplot(gs[2:4, 9:12])
f_ax15 = fig.add_subplot(gs[4:6, 9:12])
f_ax16 = fig.add_subplot(gs[6:8, 9:12])
#------------------------------------------------------------------------------
#### Plot the data:
#------------------------------------------------------------------------------
# contour_label = np.append(np.arange(0.1,0.5,0.2), np.arange(0.5,20,5))
contour_label = 3
contour_label2 = 2
all_vmin = 0
chl_vmax = chl_tot_min_outlier
bSi_vmax = bSi_min_outlier
POC_vmax = POC_min_outlier
PON_vmax = PON_min_outlier
colmap = cm.jet
#------------------------------------------------------------------------------
# total
h1 = f_ax1.contourf(np.unique(chl_binned_2010['Date']),np.arange(0,total_depth,1),chl_tot_wide_interp_2010, 100, vmin=all_vmin, vmax=chl_vmax, cmap=colmap)
h2 = f_ax1.contourf(np.unique(chl_binned_2011['Date']),np.arange(0,total_depth,1),chl_tot_wide_interp_2011, 100, vmin=all_vmin, vmax=chl_vmax, cmap=colmap)
# h3 = f_ax1.contourf(np.unique(chl_binned_2012_2013['Date']),np.arange(0,total_depth,1),chl_tot_wide_interp_2012_2013, 100, vmin=all_vmin, vmax=chl_vmax, cmap=colmap)
h4 = f_ax1.contourf(np.unique(chl_binned_2014_2015['Date']),np.arange(0,total_depth,1),chl_tot_wide_interp_2014_2015, 100, vmin=all_vmin, vmax=chl_vmax, cmap=colmap)
h5 = f_ax1.contourf(np.unique(chl_binned_2016_2017['Date']),np.arange(0,total_depth,1),chl_tot_wide_interp_2016_2017, 100, vmin=all_vmin, vmax=chl_vmax, cmap=colmap)

# plot sampling points:
# set marker size:
s = 100
# plot:
ind_grid = pd.notna(chl_binned_2010.loc[:,'CHECK Tot chl'])
s1 = f_ax1.scatter(chl_binned_2010['Date'][ind_grid], chl_binned_2010['Depth'][ind_grid], s, c=chl_binned_2010.loc[:,'CHECK Tot chl'][ind_grid], vmin=all_vmin, vmax=chl_vmax, edgecolor='w', cmap=colmap)
ind_grid = pd.notna(chl_binned_2011.loc[:,'CHECK Tot chl'])
s2 = f_ax1.scatter(chl_binned_2011['Date'][ind_grid], chl_binned_2011['Depth'][ind_grid], s, c=chl_binned_2011.loc[:,'CHECK Tot chl'][ind_grid], vmin=all_vmin, vmax=chl_vmax, edgecolor='w', cmap=colmap)
ind_grid = pd.notna(chl_binned_2012_2013.loc[:,'CHECK Tot chl'])
s3 = f_ax1.scatter(chl_binned_2012_2013['Date'][ind_grid], chl_binned_2012_2013['Depth'][ind_grid], s, c=chl_binned_2012_2013.loc[:,'CHECK Tot chl'][ind_grid], vmin=all_vmin, vmax=chl_vmax, edgecolor='k', cmap=colmap)
ind_grid = pd.notna(chl_binned_2014_2015.loc[:,'CHECK Tot chl'])
s4 = f_ax1.scatter(chl_binned_2014_2015['Date'][ind_grid], chl_binned_2014_2015['Depth'][ind_grid], s, c=chl_binned_2014_2015.loc[:,'CHECK Tot chl'][ind_grid], vmin=all_vmin, vmax=chl_vmax, edgecolor='w', cmap=colmap)
ind_grid = pd.notna(chl_binned_2016_2017.loc[:,'CHECK Tot chl'])
s5 = f_ax1.scatter(chl_binned_2016_2017['Date'][ind_grid], chl_binned_2016_2017['Depth'][ind_grid], s, c=chl_binned_2016_2017.loc[:,'CHECK Tot chl'][ind_grid], vmin=all_vmin, vmax=chl_vmax, edgecolor='w', cmap=colmap)

# Add contours:
contours1 = f_ax1.contour(np.unique(chl_binned_2010['Date']),np.arange(0,total_depth,1),chl_tot_wide_interp_2010, contour_label, colors='black')
f_ax1.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours2 = f_ax1.contour(np.unique(chl_binned_2011['Date']),np.arange(0,total_depth,1),chl_tot_wide_interp_2011, contour_label, colors='black')
f_ax1.clabel(contours2, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
# contours3 = f_ax1.contour(np.unique(chl_binned_2012_2013['Date']),np.arange(0,total_depth,1),chl_tot_wide_interp_2012_2013, contour_label, colors='black')
# f_ax1.clabel(contours3, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours4 = f_ax1.contour(np.unique(chl_binned_2014_2015['Date']),np.arange(0,total_depth,1),chl_tot_wide_interp_2014_2015, contour_label, colors='black')
f_ax1.clabel(contours4, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours5 = f_ax1.contour(np.unique(chl_binned_2016_2017['Date']),np.arange(0,total_depth,1),chl_tot_wide_interp_2016_2017, contour_label, colors='black')
f_ax1.clabel(contours5, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

# Add formatting finishing touches:
# f_ax1.set_xlabel('Time (YYYY-MM)')
f_ax1.set_ylim(0,20)
f_ax1.set_ylabel('Depth (m)')
# cbar = fig.colorbar(h4, ax=f_ax1)
# cbar.set_label('Chlorophyll $\mathit{a}$ \n(total, $\mathrm{\mu}$g L$^{-1}$)')
f_ax1.invert_yaxis()
f_axx1.set_title(' Chl$\mathbf{_{Total}}$', loc='left', fontweight='bold')
# f_ax1.set_xlabel('Time (Year)')

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

f_ax1.xaxis.set_major_locator(years)
f_ax1.xaxis.set_major_formatter(years_fmt)
f_ax1.xaxis.set_minor_locator(months)

datemin = np.datetime64(saanich_integ['Date'][saanich_integ.index[0]], 'Y')
datemax = np.datetime64(saanich_integ['Date'][saanich_integ.index[-1]], 'Y') + np.timedelta64(1, 'Y')
f_ax1.set_xlim(datemin, datemax)

f_ax1.yaxis.set_minor_locator(MultipleLocator(1))
f_ax1.tick_params(axis='both', which='major', length=10)
f_ax1.tick_params(axis='both', which='minor', length=6)
f_ax1.tick_params(axis='y', which='both', right='True', labelleft='on')

# labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
# labels[-1] = '>'+str("{:.1f}".format(chl_tot_min_outlier))
# cbar.ax.set_yticklabels(labels)
#------------------------------------------------------------------------------
total_depth = np.max(particulates_binned_2010['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)

# POC
h6 = f_ax2.contourf(np.unique(particulates_binned_2010['Date']),np.arange(0,total_depth,1),POC_wide_interp_2010, 100, vmin=all_vmin, vmax=POC_vmax, cmap=cm.jet)
h7 = f_ax2.contourf(np.unique(particulates_binned_2011['Date']),np.arange(0,total_depth,1),POC_wide_interp_2011, 100, vmin=all_vmin, vmax=POC_vmax, cmap=cm.jet)
# h8 = f_ax2.contourf(np.unique(particulates_binned_2012_2013['Date']),np.arange(0,total_depth,1),POC_wide_interp_2012_2013, 100, vmin=all_vmin, vmax=POC_vmax, cmap=cm.jet)
h9 = f_ax2.contourf(np.unique(particulates_binned_2014_2015['Date']),np.arange(0,total_depth,1),POC_wide_interp_2014_2015, 100, vmin=all_vmin, vmax=POC_vmax, cmap=cm.jet)
h10 = f_ax2.contourf(np.unique(particulates_binned_2016_2017['Date']),np.arange(0,total_depth,1),POC_wide_interp_2016_2017, 100, vmin=all_vmin, vmax=POC_vmax, cmap=cm.jet)

# plot sampling points:
# set marker size:
s = 100
# plot:
ind_grid = pd.notna(particulates_binned_2010.loc[:,'[PC] (umol/L)'])
s1 = f_ax2.scatter(particulates_binned_2010['Date'][ind_grid], particulates_binned_2010['Depth'][ind_grid], s, c=particulates_binned_2010.loc[:,'[PC] (umol/L)'][ind_grid], vmin=all_vmin, vmax=POC_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(particulates_binned_2011.loc[:,'[PC] (umol/L)'])
s2 = f_ax2.scatter(particulates_binned_2011['Date'][ind_grid], particulates_binned_2011['Depth'][ind_grid], s, c=particulates_binned_2011.loc[:,'[PC] (umol/L)'][ind_grid], vmin=all_vmin, vmax=POC_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(particulates_binned_2012_2013.loc[:,'[PC] (umol/L)'])
s3 = f_ax2.scatter(particulates_binned_2012_2013['Date'][ind_grid], particulates_binned_2012_2013['Depth'][ind_grid], s, c=particulates_binned_2012_2013.loc[:,'[PC] (umol/L)'][ind_grid], vmin=all_vmin, vmax=POC_vmax, edgecolor='k', cmap=cm.jet)
ind_grid = pd.notna(particulates_binned_2014_2015.loc[:,'[PC] (umol/L)'])
s4 = f_ax2.scatter(particulates_binned_2014_2015['Date'][ind_grid], particulates_binned_2014_2015['Depth'][ind_grid], s, c=particulates_binned_2014_2015.loc[:,'[PC] (umol/L)'][ind_grid], vmin=all_vmin, vmax=POC_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(particulates_binned_2016_2017.loc[:,'[PC] (umol/L)'])
s5 = f_ax2.scatter(particulates_binned_2016_2017['Date'][ind_grid], particulates_binned_2016_2017['Depth'][ind_grid], s, c=particulates_binned_2016_2017.loc[:,'[PC] (umol/L)'][ind_grid], vmin=all_vmin, vmax=POC_vmax, edgecolor='w', cmap=cm.jet)

# Add contours:
contours1 = f_ax2.contour(np.unique(particulates_binned_2010['Date']),np.arange(0,total_depth,1),POC_wide_interp_2010, contour_label, colors='black')
f_ax2.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours2 = f_ax2.contour(np.unique(particulates_binned_2011['Date']),np.arange(0,total_depth,1),POC_wide_interp_2011, contour_label, colors='black')
f_ax2.clabel(contours2, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
# contours3 = f_ax2.contour(np.unique(particulates_binned_2012_2013['Date']),np.arange(0,total_depth,1),POC_wide_interp_2012_2013, contour_label, colors='black')
# f_ax2.clabel(contours3, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours4 = f_ax2.contour(np.unique(particulates_binned_2014_2015['Date']),np.arange(0,total_depth,1),POC_wide_interp_2014_2015, contour_label, colors='black')
f_ax2.clabel(contours4, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours5 = f_ax2.contour(np.unique(particulates_binned_2016_2017['Date']),np.arange(0,total_depth,1),POC_wide_interp_2016_2017, contour_label, colors='black')
f_ax2.clabel(contours5, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

# Add formatting finishing touches:
# f_ax1.set_xlabel('Time (YYYY-MM)')
f_ax2.set_ylim(0,20)
f_ax2.set_ylabel('Depth (m)')
# cbar = fig.colorbar(h2, ax=f_ax2)
# cbar.set_label('Particulate Organic Carbon \n(POC, $\mathrm{\mu}$mol L$^{-1}$)')
f_ax2.invert_yaxis()
f_axx2.set_title(' POC', loc='left', fontweight='bold')
f_ax2.xaxis.set_major_locator(years)
f_ax2.xaxis.set_major_formatter(years_fmt)
f_ax2.xaxis.set_minor_locator(months)

datemin = np.datetime64(saanich_integ['Date'][saanich_integ.index[0]], 'Y')
datemax = np.datetime64(saanich_integ['Date'][saanich_integ.index[-1]], 'Y') + np.timedelta64(1, 'Y')
f_ax2.set_xlim(datemin, datemax)

f_ax2.yaxis.set_minor_locator(MultipleLocator(1))
f_ax2.tick_params(axis='both', which='major', length=10)
f_ax2.tick_params(axis='both', which='minor', length=6)
f_ax2.tick_params(axis='y', which='both', right='True', labelleft='on')

# labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
# labels[-1] = '>'+str("{:.1f}".format(POC_min_outlier))
# cbar.ax.set_yticklabels(labels)
#------------------------------------------------------------------------------
# PON
h11 = f_ax3.contourf(np.unique(particulates_binned_2010['Date']),np.arange(0,total_depth,1),PON_wide_interp_2010, 100, vmin=all_vmin, vmax=PON_vmax, cmap=cm.jet)
h12 = f_ax3.contourf(np.unique(particulates_binned_2011['Date']),np.arange(0,total_depth,1),PON_wide_interp_2011, 100, vmin=all_vmin, vmax=PON_vmax, cmap=cm.jet)
# h13 = f_ax3.contourf(np.unique(particulates_binned_2012_2013['Date']),np.arange(0,total_depth,1),PON_wide_interp_2012_2013, 100, vmin=all_vmin, vmax=PON_vmax, cmap=cm.jet)
h14 = f_ax3.contourf(np.unique(particulates_binned_2014_2015['Date']),np.arange(0,total_depth,1),PON_wide_interp_2014_2015, 100, vmin=all_vmin, vmax=PON_vmax, cmap=cm.jet)
h15 = f_ax3.contourf(np.unique(particulates_binned_2016_2017['Date']),np.arange(0,total_depth,1),PON_wide_interp_2016_2017, 100, vmin=all_vmin, vmax=PON_vmax, cmap=cm.jet)

# plot sampling points:
# set marker size:
s = 100
# plot:
ind_grid = pd.notna(particulates_binned_2010.loc[:,'[PN] (umol/L)'])
s1 = f_ax3.scatter(particulates_binned_2010['Date'][ind_grid], particulates_binned_2010['Depth'][ind_grid], s, c=particulates_binned_2010.loc[:,'[PN] (umol/L)'][ind_grid], vmin=all_vmin, vmax=PON_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(particulates_binned_2011.loc[:,'[PN] (umol/L)'])
s2 = f_ax3.scatter(particulates_binned_2011['Date'][ind_grid], particulates_binned_2011['Depth'][ind_grid], s, c=particulates_binned_2011.loc[:,'[PN] (umol/L)'][ind_grid], vmin=all_vmin, vmax=PON_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(particulates_binned_2012_2013.loc[:,'[PN] (umol/L)'])
s3 = f_ax3.scatter(particulates_binned_2012_2013['Date'][ind_grid], particulates_binned_2012_2013['Depth'][ind_grid], s, c=particulates_binned_2012_2013.loc[:,'[PN] (umol/L)'][ind_grid], vmin=all_vmin, vmax=PON_vmax, edgecolor='k', cmap=cm.jet)
ind_grid = pd.notna(particulates_binned_2014_2015.loc[:,'[PN] (umol/L)'])
s4 = f_ax3.scatter(particulates_binned_2014_2015['Date'][ind_grid], particulates_binned_2014_2015['Depth'][ind_grid], s, c=particulates_binned_2014_2015.loc[:,'[PN] (umol/L)'][ind_grid], vmin=all_vmin, vmax=PON_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(particulates_binned_2016_2017.loc[:,'[PN] (umol/L)'])
s5 = f_ax3.scatter(particulates_binned_2016_2017['Date'][ind_grid], particulates_binned_2016_2017['Depth'][ind_grid], s, c=particulates_binned_2016_2017.loc[:,'[PN] (umol/L)'][ind_grid], vmin=all_vmin, vmax=PON_vmax, edgecolor='w', cmap=cm.jet)

# Add contours:
contours1 = f_ax3.contour(np.unique(particulates_binned_2010['Date']),np.arange(0,total_depth,1),PON_wide_interp_2010, contour_label, colors='black')
f_ax3.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours2 = f_ax3.contour(np.unique(particulates_binned_2011['Date']),np.arange(0,total_depth,1),PON_wide_interp_2011, contour_label, colors='black')
f_ax3.clabel(contours2, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
# contours3 = f_ax3.contour(np.unique(particulates_binned_2012_2013['Date']),np.arange(0,total_depth,1),PON_wide_interp_2012_2013, contour_label, colors='black')
# f_ax3.clabel(contours3, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours4 = f_ax3.contour(np.unique(particulates_binned_2014_2015['Date']),np.arange(0,total_depth,1),PON_wide_interp_2014_2015, contour_label, colors='black')
f_ax3.clabel(contours4, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours5 = f_ax3.contour(np.unique(particulates_binned_2016_2017['Date']),np.arange(0,total_depth,1),PON_wide_interp_2016_2017, contour_label, colors='black')
f_ax3.clabel(contours5, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

# Add formatting finishing touches:
f_ax3.set_ylim(0,20)
f_ax3.set_ylabel('Depth (m)')
# cbar = fig.colorbar(h5, ax=f_ax3)
# cbar.set_label('Particulate Organic Nitrogen \n(PON, $\mathrm{\mu}$mol L$^{-1}$)')
f_ax3.invert_yaxis()
f_axx3.set_title(' PON', loc='left', fontweight='bold')
f_ax3.xaxis.set_major_locator(years)
f_ax3.xaxis.set_major_formatter(years_fmt)
f_ax3.xaxis.set_minor_locator(months)

datemin = np.datetime64(saanich_integ['Date'][saanich_integ.index[0]], 'Y')
datemax = np.datetime64(saanich_integ['Date'][saanich_integ.index[-1]], 'Y') + np.timedelta64(1, 'Y')
f_ax3.set_xlim(datemin, datemax)

f_ax3.yaxis.set_minor_locator(MultipleLocator(1))
f_ax3.tick_params(axis='both', which='major', length=10)
f_ax3.tick_params(axis='both', which='minor', length=6)
f_ax3.tick_params(axis='y', which='both', right='True', labelleft='on')

# labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
# labels[-1] = '>'+str("{:.2f}".format(PON_min_outlier))
# cbar.ax.set_yticklabels(labels)
#------------------------------------------------------------------------------
# bSi
h16 = f_ax4.contourf(np.unique(particulates_binned_2010['Date']),np.arange(0,total_depth,1),bSi_wide_interp_2010, 100, vmin=all_vmin, vmax=bSi_vmax, cmap=cm.jet)
h17 = f_ax4.contourf(np.unique(particulates_binned_2011['Date']),np.arange(0,total_depth,1),bSi_wide_interp_2011, 100, vmin=all_vmin, vmax=bSi_vmax, cmap=cm.jet)
# h18 = f_ax4.contourf(np.unique(particulates_binned_2012_2013['Date']),np.arange(0,total_depth,1),bSi_wide_interp_2012_2013, 100, vmin=all_vmin, vmax=bSi_vmax, cmap=cm.jet)
h19 = f_ax4.contourf(np.unique(particulates_binned_2014_2015['Date']),np.arange(0,total_depth,1),bSi_wide_interp_2014_2015, 100, vmin=all_vmin, vmax=bSi_vmax, cmap=cm.jet)
h20 = f_ax4.contourf(np.unique(particulates_binned_2016_2017['Date']),np.arange(0,total_depth,1),bSi_wide_interp_2016_2017, 100, vmin=all_vmin, vmax=bSi_vmax, cmap=cm.jet)

# plot sampling points:
# set marker size:
s = 100
# plot:
ind_grid = pd.notna(particulates_binned_2010['Initial [bSiO2]'])
s1 = f_ax4.scatter(particulates_binned_2010['Date'][ind_grid], particulates_binned_2010['Depth'][ind_grid], s, c=particulates_binned_2010['Initial [bSiO2]'][ind_grid], vmin=all_vmin, vmax=bSi_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(particulates_binned_2011['Initial [bSiO2]'])
s2 = f_ax4.scatter(particulates_binned_2011['Date'][ind_grid], particulates_binned_2011['Depth'][ind_grid], s, c=particulates_binned_2011['Initial [bSiO2]'][ind_grid], vmin=all_vmin, vmax=bSi_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(particulates_binned_2012_2013['Initial [bSiO2]'])
s3 = f_ax4.scatter(particulates_binned_2012_2013['Date'][ind_grid], particulates_binned_2012_2013['Depth'][ind_grid], s, c=particulates_binned_2012_2013['Initial [bSiO2]'][ind_grid], vmin=all_vmin, vmax=bSi_vmax, edgecolor='k', cmap=cm.jet)
ind_grid = pd.notna(particulates_binned_2014_2015['Initial [bSiO2]'])
s4 = f_ax4.scatter(particulates_binned_2014_2015['Date'][ind_grid], particulates_binned_2014_2015['Depth'][ind_grid], s, c=particulates_binned_2014_2015['Initial [bSiO2]'][ind_grid], vmin=all_vmin, vmax=bSi_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(particulates_binned_2016_2017['Initial [bSiO2]'])
s5 = f_ax4.scatter(particulates_binned_2016_2017['Date'][ind_grid], particulates_binned_2016_2017['Depth'][ind_grid], s, c=particulates_binned_2016_2017['Initial [bSiO2]'][ind_grid], vmin=all_vmin, vmax=bSi_vmax, edgecolor='w', cmap=cm.jet)

# Add contours:
contours1 = f_ax4.contour(np.unique(particulates_binned_2010['Date']),np.arange(0,total_depth,1),bSi_wide_interp_2010, contour_label, colors='black')
f_ax1.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours2 = f_ax4.contour(np.unique(particulates_binned_2011['Date']),np.arange(0,total_depth,1),bSi_wide_interp_2011, contour_label, colors='black')
f_ax1.clabel(contours2, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
# contours3 = f_ax4.contour(np.unique(particulates_binned_2012_2013['Date']),np.arange(0,total_depth,1),bSi_wide_interp_2012_2013, contour_label, colors='black')
# f_ax1.clabel(contours3, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours4 = f_ax4.contour(np.unique(particulates_binned_2014_2015['Date']),np.arange(0,total_depth,1),bSi_wide_interp_2014_2015, contour_label, colors='black')
f_ax1.clabel(contours4, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours5 = f_ax4.contour(np.unique(particulates_binned_2016_2017['Date']),np.arange(0,total_depth,1),bSi_wide_interp_2016_2017, contour_label, colors='black')
f_ax1.clabel(contours5, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

# Add formatting finishing touches:
f_ax4.set_xlabel('Time (Years)')
f_ax4.set_ylim(0,20)
f_ax4.set_ylabel('Depth (m)')
# cbar = fig.colorbar(h1, ax=f_ax4)
# cbar.set_label('Biogenic Silica \n(bSiO$_{\mathrm{2}}$, $\mathrm{\mu}$mol L$^{-1}$)')
f_ax4.invert_yaxis()
f_axx4.set_title(r' bSiO$_{\mathbf{2}}$', loc='left', fontweight='bold')

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

f_ax4.xaxis.set_major_locator(years)
f_ax4.xaxis.set_major_formatter(years_fmt)
f_ax4.xaxis.set_minor_locator(months)

datemin = np.datetime64(saanich_integ['Date'][saanich_integ.index[0]], 'Y')
datemax = np.datetime64(saanich_integ['Date'][saanich_integ.index[-1]], 'Y') + np.timedelta64(1, 'Y')
f_ax4.set_xlim(datemin, datemax)

f_ax4.yaxis.set_minor_locator(MultipleLocator(1))
f_ax4.tick_params(axis='both', which='major', length=10)
f_ax4.tick_params(axis='both', which='minor', length=6)
f_ax4.tick_params(axis='y', which='both', right='True', labelleft='on')

# labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
# labels[-1] = '>'+str("{:.2f}".format(bSi_min_outlier))
# cbar.ax.set_yticklabels(labels)
#------------------------------------------------------------------------------
#### Add integrated values with growing seasons
# Total chl
width = 10
# plot integrated stacked %
f_ax9.bar(chl_percent_integ['Date'], chl_percent_integ['CHL-A (0.7-2 um)'], width, label='0.7-2 $\mu$m', color='blue')
f_ax9.bar(chl_percent_integ['Date'], chl_percent_integ['CHL-A (2-5 um)'], width, label='2-5 $\mu$m', color='orange')
f_ax9.bar(chl_percent_integ['Date'], chl_percent_integ['CHL-A (0.7-5 um)'], width, label='0.7-5 $\mu$m', color='green')
f_ax9.bar(chl_percent_integ['Date'], chl_percent_integ['CHL-A (5-20 um)'], width, label='5-20 $\mu$m', color='red')
f_ax9.bar(chl_percent_integ['Date'], chl_percent_integ['CHL-A (>20 um)'], width, label='>20 $\mu$m', color='black')
# plot single depth stacked %
f_ax9.bar(chl_percent_single['Date'], chl_percent_single['CHL-A (0.7-2um)'], width, color='blue')
f_ax9.bar(chl_percent_single['Date'], chl_percent_single['CHL-A (2-5um)'], width, color='orange')
f_ax9.bar(chl_percent_single['Date'], chl_percent_single['CHL-A (0.7-5um)'], width, color='green')
f_ax9.bar(chl_percent_single['Date'], chl_percent_single['CHL-A (5-20um)'], width, color='red')
f_ax9.bar(chl_percent_single['Date'], chl_percent_single['CHL-A (>20um)'], width, color='black')

f_ax9.set_ylim([0,100])
f_ax9.tick_params(axis='y', which='major', length=10)
f_ax9.tick_params(axis='y', which='minor', length=6)
# f_ax9.set_ylabel('Chl $\mathit{a}$ \nSize \nFractions \n(%)')
f_ax9.set_ylabel('(%)')
f_ax9.legend(loc='upper center', fontsize=16, ncol=5, bbox_to_anchor=(0.5, 1.5), fancybox=True)
f_ax9.set_xlim(datemin, datemax)
f_ax9.yaxis.set_ticks(np.arange(0, 125, 25))

color_code = 'dimgray'
f_ax9.axvspan(datetime.date(2010,3,1), datetime.date(2010,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9.axvspan(datetime.date(2011,3,1), datetime.date(2011,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9.axvspan(datetime.date(2012,3,1), datetime.date(2012,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9.axvspan(datetime.date(2013,3,1), datetime.date(2013,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9.axvspan(datetime.date(2014,3,1), datetime.date(2014,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9.axvspan(datetime.date(2015,3,1), datetime.date(2015,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9.axvspan(datetime.date(2016,3,1), datetime.date(2016,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9.axvspan(datetime.date(2017,3,1), datetime.date(2017,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)

f_ax9.tick_params(axis='y', which='both', right='True', labelleft='on')

# f_ax9.set_ylim(0,2000)
# POC
ind1 = pd.notna(saanich_integ_2010.loc[:,'[PC] (umol/L)'])
ind2 = pd.notna(saanich_integ_2011.loc[:,'[PC] (umol/L)'])
ind3 = pd.notna(saanich_integ_2014_2015.loc[:,'[PC] (umol/L)'])
ind4 = pd.notna(saanich_integ_2016_2017.loc[:,'[PC] (umol/L)'])
f_ax10.plot(saanich_integ_2010['Date'][ind1], saanich_integ_2010.loc[:,'[PC] (umol/L)'][ind1], 'k--.', markersize=14, clip_on=False)
f_ax10.plot(saanich_integ_2011['Date'][ind2], saanich_integ_2011.loc[:,'[PC] (umol/L)'][ind2], 'k--.', markersize=14, clip_on=False)
f_ax10.plot(saanich_integ_2014_2015['Date'][ind3], saanich_integ_2014_2015.loc[:,'[PC] (umol/L)'][ind3], 'k--.', markersize=14, clip_on=False)
f_ax10.plot(saanich_integ_2016_2017['Date'][ind4], saanich_integ_2016_2017.loc[:,'[PC] (umol/L)'][ind4], 'k--.', markersize=14, clip_on=False)
f_ax10.set_ylabel('\n(mmol m$^{-2}$)')
# f_ax10.set_xlim(np.min(saanich_integ['Date']),np.max(saanich_integ['Date']))
f_ax10.set_xlim(datemin, datemax)
f_ax10.autoscale(enable=True, axis='y', tight=True)

# f_ax10.ticklabel_format(axis='y', scilimits=(0,2), useMathText=True) # set tick labels to scientific notation
# t = f_ax10.yaxis.get_offset_text()
# t.set_x(-0.035) # offset scientific notation above ticks
# f_ax10.set_ylim(0,1700)
# f_ax10.yaxis.set_ticks(np.arange(0, 1700, 500))
# labels = f_ax10.get_yticks().tolist()
# labels[0] = ''
# f_ax10.set_yticklabels(labels)

f_ax10.tick_params(axis='y', which='major', length=10)
f_ax10.tick_params(axis='y', which='minor', length=6)

f_ax10.axvspan(datetime.date(2010,3,1), datetime.date(2010,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2011,3,1), datetime.date(2011,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2012,3,1), datetime.date(2012,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2013,3,1), datetime.date(2013,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2014,3,1), datetime.date(2014,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2015,3,1), datetime.date(2015,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2016,3,1), datetime.date(2016,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2017,3,1), datetime.date(2017,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)

f_ax10.tick_params(axis='y', which='both', right='True', labelleft='on')

# f_ax11.set_ylim(0,2000)
# PON
ind1 = pd.notna(saanich_integ_2010.loc[:,'[PN] (umol/L)'])
ind2 = pd.notna(saanich_integ_2011.loc[:,'[PN] (umol/L)'])
ind3 = pd.notna(saanich_integ_2014_2015.loc[:,'[PN] (umol/L)'])
ind4 = pd.notna(saanich_integ_2016_2017.loc[:,'[PN] (umol/L)'])
f_ax11.plot(saanich_integ_2010['Date'][ind1], saanich_integ_2010.loc[:,'[PN] (umol/L)'][ind1], 'k--.', markersize=14, clip_on=False)
f_ax11.plot(saanich_integ_2011['Date'][ind2], saanich_integ_2011.loc[:,'[PN] (umol/L)'][ind2], 'k--.', markersize=14, clip_on=False)
f_ax11.plot(saanich_integ_2014_2015['Date'][ind3], saanich_integ_2014_2015.loc[:,'[PN] (umol/L)'][ind3], 'k--.', markersize=14, clip_on=False)
f_ax11.plot(saanich_integ_2016_2017['Date'][ind4], saanich_integ_2016_2017.loc[:,'[PN] (umol/L)'][ind4], 'k--.', markersize=14, clip_on=False)
f_ax11.set_ylabel('\n(mmol m$^{-2}$)')
# f_ax11.set_xlim(np.min(saanich_integ['Date']),np.max(saanich_integ['Date']))
f_ax11.set_xlim(datemin, datemax)
f_ax11.autoscale(enable=True, axis='y', tight=True)
f_ax11.ticklabel_format(axis='y', scilimits=(0,3), useMathText=True) # set tick labels to scientific notation

f_ax11.tick_params(axis='y', which='major', length=10)
f_ax11.tick_params(axis='y', which='minor', length=6)
start, end = f_ax11.get_ylim()
f_ax11.yaxis.set_ticks(np.arange(0, 301, 100))

f_ax11.axvspan(datetime.date(2010,3,1), datetime.date(2010,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2011,3,1), datetime.date(2011,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2012,3,1), datetime.date(2012,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2013,3,1), datetime.date(2013,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2014,3,1), datetime.date(2014,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2015,3,1), datetime.date(2015,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2016,3,1), datetime.date(2016,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2017,3,1), datetime.date(2017,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)

f_ax11.tick_params(axis='y', which='both', right='True', labelleft='on')

# f_ax11.set_ylim(0,2000)

# bSi
ind1 = pd.notna(saanich_integ_2010['Initial [bSiO2]'])
ind2 = pd.notna(saanich_integ_2011['Initial [bSiO2]'])
ind3 = pd.notna(saanich_integ_2014_2015['Initial [bSiO2]'])
ind4 = pd.notna(saanich_integ_2016_2017['Initial [bSiO2]'])
f_ax12.plot(saanich_integ_2010['Date'][ind1], saanich_integ_2010['Initial [bSiO2]'][ind1], 'k--.', markersize=14, clip_on=False)
f_ax12.plot(saanich_integ_2011['Date'][ind2], saanich_integ_2011['Initial [bSiO2]'][ind2], 'k--.', markersize=14, clip_on=False)
f_ax12.plot(saanich_integ_2014_2015['Date'][ind3], saanich_integ_2014_2015['Initial [bSiO2]'][ind3], 'k--.', markersize=14, clip_on=False)
f_ax12.plot(saanich_integ_2016_2017['Date'][ind4], saanich_integ_2016_2017['Initial [bSiO2]'][ind4], 'k--.', markersize=14, clip_on=False)
f_ax12.set_ylabel('\n(mmol m$^{-2}$)')
# f_ax12.set_xlim(np.min(saanich_integ['Date']),np.max(saanich_integ['Date']))
f_ax12.set_xlim(datemin, datemax)
f_ax12.autoscale(enable=True, axis='y', tight=True)
f_ax12.tick_params(axis='y', which='major', length=10)
f_ax12.tick_params(axis='y', which='minor', length=6)
color_code = 'dimgray'
f_ax12.axvspan(datetime.date(2010,3,1), datetime.date(2010,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax12.axvspan(datetime.date(2011,3,1), datetime.date(2011,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax12.axvspan(datetime.date(2012,3,1), datetime.date(2012,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax12.axvspan(datetime.date(2013,3,1), datetime.date(2013,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax12.axvspan(datetime.date(2014,3,1), datetime.date(2014,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax12.axvspan(datetime.date(2015,3,1), datetime.date(2015,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax12.axvspan(datetime.date(2016,3,1), datetime.date(2016,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax12.axvspan(datetime.date(2017,3,1), datetime.date(2017,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)

f_ax12.tick_params(axis='y', which='both', right='True', labelleft='on')
#------------------------------------------------------------------------------
#### Plot seasonally averaged data:
total_depth = np.max(chl_binned['Depth'])+1
# total chl
h21 = f_ax13.contourf(np.unique(chl_binned['Month']),np.arange(0,total_depth,1),chl_tot_seas_wide_interp, 100, cmap=cm.jet)

ind_grid = pd.notna(chl_binned['CHECK Tot chl'])
s1 = f_ax13.scatter(chl_binned['Month'][ind_grid], chl_binned['Depth'][ind_grid], 100, c=chl_binned['CHECK Tot chl'][ind_grid], edgecolor='w', cmap=cm.jet, clip_on=False)

contours1 = f_ax13.contour(np.unique(chl_binned['Month']),np.arange(0,total_depth,1),chl_tot_seas_wide_interp, contour_label, colors='black')
f_ax13.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

f_ax13.set_ylabel('Depth (m)')
# f_ax13.set_xlabel('Month')
f_ax13.set_xlim([1,12])
cbar = fig.colorbar(h21, ax=f_ax13, pad=-0.02, aspect=30)
cbar.set_label('($\mathrm{\mu}$g L$^{-1}$)')
f_ax13.invert_yaxis()
f_ax13.set_title(r' Chl$\mathbf{_{Total}}$', loc='left', fontweight='bold')

f_ax13.set_xticks(np.arange(1,13,1))
f_ax13.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
f_ax13.tick_params(axis='both', which='major', length=10)
f_ax13.tick_params(axis='both', which='minor', length=6)

labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
labels[-1] = '>'+str("{:.1f}".format(chl_tot_min_outlier))
cbar.ax.set_yticklabels(labels)

f_ax13.yaxis.set_minor_locator(MultipleLocator(1))
f_ax13.tick_params(axis='both', which='major', length=10)
f_ax13.tick_params(axis='y', which='minor', length=6)
f_ax13.tick_params(axis='y', which='both', right='True', labelleft='on')


# POC
h22 = f_ax14.contourf(np.unique(particulates_binned['Month']),np.arange(0,total_depth,1),POC_seas_wide_interp, 100, cmap=cm.jet)

ind_grid = pd.notna(particulates_binned['[PC] (umol/L)'])
s1 = f_ax14.scatter(particulates_binned['Month'][ind_grid], particulates_binned['Depth'][ind_grid], 100, c=particulates_binned['[PC] (umol/L)'][ind_grid], edgecolor='w', cmap=cm.jet, clip_on=False)

contours1 = f_ax14.contour(np.unique(particulates_binned['Month']),np.arange(0,total_depth,1),POC_seas_wide_interp, contour_label, colors='black')
f_ax14.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

f_ax14.set_ylabel('Depth (m)')
# f_ax14.set_xlabel('Month')
f_ax14.set_xlim([1,12])
cbar = fig.colorbar(h22, ax=f_ax14, pad=-0.02, aspect=30)
cbar.set_label('($\mathrm{\mu}$mol L$^{-1}$)')
f_ax14.invert_yaxis()
f_ax14.set_title(r' POC', loc='left', fontweight='bold')

f_ax14.set_xticks(np.arange(1,13,1))
f_ax14.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
f_ax14.tick_params(axis='both', which='major', length=10)
f_ax14.tick_params(axis='both', which='minor', length=6)

labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
labels[-1] = '>'+str("{:.1f}".format(POC_min_outlier))
cbar.ax.set_yticklabels(labels)

f_ax14.yaxis.set_minor_locator(MultipleLocator(1))
f_ax14.tick_params(axis='both', which='major', length=10)
f_ax14.tick_params(axis='y', which='minor', length=6)
f_ax14.tick_params(axis='y', which='both', right='True', labelleft='on')

# PON
h23 = f_ax15.contourf(np.unique(particulates_binned['Month']),np.arange(0,total_depth,1),PON_seas_wide_interp, 100, cmap=cm.jet)

ind_grid = pd.notna(particulates_binned['[PN] (umol/L)'])
s1 = f_ax15.scatter(particulates_binned['Month'][ind_grid], particulates_binned['Depth'][ind_grid], 100, c=particulates_binned['[PN] (umol/L)'][ind_grid], edgecolor='w', cmap=cm.jet, clip_on=False)

contours1 = f_ax15.contour(np.unique(particulates_binned['Month']),np.arange(0,total_depth,1),PON_seas_wide_interp, contour_label, colors='black')
f_ax15.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

f_ax15.set_ylabel('Depth (m)')
f_ax15.set_xlim([1,12])
cbar = fig.colorbar(h23, ax=f_ax15, pad=-0.02, aspect=30)
cbar.set_label('($\mathrm{\mu}$mol L$^{-1}$)')
f_ax15.invert_yaxis()
f_ax15.set_title(r' PON', loc='left', fontweight='bold')

f_ax15.set_xticks(np.arange(1,13,1))
f_ax15.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
f_ax15.tick_params(axis='both', which='major', length=10)
f_ax15.tick_params(axis='both', which='minor', length=6)

labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
labels[-1] = '>'+str("{:.2f}".format(PON_min_outlier))
cbar.ax.set_yticklabels(labels)

f_ax15.yaxis.set_minor_locator(MultipleLocator(1))
f_ax15.tick_params(axis='both', which='major', length=10)
f_ax15.tick_params(axis='y', which='minor', length=6)
f_ax15.tick_params(axis='y', which='both', right='True', labelleft='on')

# bSi
total_depth = np.max(particulates_binned['Depth'])+1
h24 = f_ax16.contourf(np.unique(particulates_binned['Month']),np.arange(0,total_depth,1),bSi_seas_wide_interp, 100, cmap=cm.jet)

ind_grid = pd.notna(particulates_binned['Initial [bSiO2]'])
s1 = f_ax16.scatter(particulates_binned['Month'][ind_grid], particulates_binned['Depth'][ind_grid], 100, c=particulates_binned['Initial [bSiO2]'][ind_grid], edgecolor='w', cmap=cm.jet, clip_on=False)

contours1 = f_ax16.contour(np.unique(particulates_binned['Month']),np.arange(0,total_depth,1),bSi_seas_wide_interp, contour_label, colors='black')
f_ax16.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

f_ax16.set_ylabel('Depth (m)')
f_ax16.set_xlabel('Month')
f_ax16.set_xlim([1,12])
cbar = fig.colorbar(h24, ax=f_ax16, format="%1.1f", pad=-0.02, aspect=30)
cbar.set_label('($\mathrm{\mu}$mol L$^{-1}$)')
f_ax16.invert_yaxis()
f_ax16.set_title(r' bSiO$_{\mathbf{2}}$', loc='left', fontweight='bold')

f_ax16.set_xticks(np.arange(1,13,1))
f_ax16.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
f_ax16.tick_params(axis='both', which='major', length=10)
f_ax16.tick_params(axis='both', which='minor', length=6)

labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
labels[-1] = '>'+str("{:.1f}".format(bSi_min_outlier))
cbar.ax.set_yticklabels(labels)

f_ax16.yaxis.set_minor_locator(MultipleLocator(1))
f_ax16.tick_params(axis='both', which='major', length=10)
f_ax16.tick_params(axis='y', which='minor', length=6)
f_ax16.tick_params(axis='y', which='both', right='True', labelleft='on')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### Collate figure handles for a vector graphic format:
h_part = [h1,h2,h4,h5,h6,h7,h9,h10,h11,h12,h14,h15,h16,h17,h19,h20,h21,h22,h23,h24] # collate plot handles
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% Production: contour plots
total_depth = np.max(production_binned_2010['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
#------------------------------------------------------------------------------
#### Set-up figure layout:
fig = plt.figure(figsize=(34,24), constrained_layout=True)
font={'family':'DejaVu Sans',
      'weight':'normal',
      'size':'22'} 
plt.rc('font', **font) # sets the specified font formatting globally

gs = fig.add_gridspec(6, 12)
# main plots
f_axx1 = fig.add_subplot(gs[0:2, :-3])
for axis in ['left','right']:
    f_axx1.spines[axis].set_visible(False)
f_axx1.get_xaxis().set_ticks([])
f_axx1.get_yaxis().set_ticks([])

f_ax1 = f_axx1.inset_axes([0, 0, 1, 0.75])
f_ax1.spines['bottom'].set_visible(False)
# f_ax1.set_position([0.06996741830065356, 0.7093259490740736, 0.765, 0.2])

f_ax9 = f_axx1.inset_axes([0, 0.75, 1, 0.25])
f_ax9.spines['bottom'].set_visible(False)
f_ax9.get_xaxis().set_ticks([])

# add axes for axis break
for axis in ['left','right']:
    f_ax9.spines[axis].set_visible(False)
f_ax9.get_xaxis().set_visible(False)
f_ax9.get_yaxis().set_visible(False)

f_ax9_a = f_ax9.inset_axes([0, 0, 1, 0.7])
f_ax9_a.spines['top'].set_visible(False)
f_ax9_a.spines['bottom'].set_visible(False)
f_ax9_a.get_xaxis().set_ticks([])

f_ax9_b = f_ax9.inset_axes([0,0.8,1,0.2], sharex = f_ax9_a)
f_ax9_b.spines['bottom'].set_visible(False)
f_ax9_b.get_xaxis().set_ticks([])
#------------------------------------------------------------------------------
f_axx2 = fig.add_subplot(gs[2:4, :-3])
f_axx2.get_xaxis().set_ticks([])
f_axx2.get_yaxis().set_ticks([])

f_ax2 = f_axx2.inset_axes([0, 0, 1, 0.75])
f_ax2.spines['bottom'].set_visible(False)
# f_ax2.set_position([0.06137877450980385, 0.3803791898148143, 0.775, 0.2])

f_ax10 = f_axx2.inset_axes([0, 0.75, 1, 0.25])
f_ax10.spines['bottom'].set_visible(False)
f_ax10.get_xaxis().set_ticks([])
#------------------------------------------------------------------------------
f_axx3 = fig.add_subplot(gs[4:6, :-3])
f_axx3.get_xaxis().set_ticks([])
f_axx3.get_yaxis().set_ticks([])

f_ax3 = f_axx3.inset_axes([0, 0, 1, 0.75])
f_ax3.spines['bottom'].set_visible(False)
# f_ax3.set_position([0.06137877450980389, 0.04593474537037023, 0.775, 0.2])

f_ax11 = f_axx3.inset_axes([0, 0.75, 1, 0.25])
f_ax11.spines['bottom'].set_visible(False)
f_ax11.get_xaxis().set_ticks([])
#------------------------------------------------------------------------------
# f_ax_del1 = fig.add_subplot(gs[0:2, 8:9])
# f_ax_del1.spines['top'].set_visible(False)
# f_ax_del1.spines['bottom'].set_visible(False)
# f_ax_del1.spines['left'].set_visible(False)
# f_ax_del1.spines['right'].set_visible(False)
# f_ax_del1.get_xaxis().set_visible(False)
# f_ax_del1.get_yaxis().set_visible(False)

# f_ax_del2 = fig.add_subplot(gs[2:4, 8:9])
# f_ax_del2.spines['top'].set_visible(False)
# f_ax_del2.spines['bottom'].set_visible(False)
# f_ax_del2.spines['left'].set_visible(False)
# f_ax_del2.spines['right'].set_visible(False)
# f_ax_del2.get_xaxis().set_visible(False)
# f_ax_del2.get_yaxis().set_visible(False)

# f_ax_del3 = fig.add_subplot(gs[4:6, 8:9])
# f_ax_del3.spines['top'].set_visible(False)
# f_ax_del3.spines['bottom'].set_visible(False)
# f_ax_del3.spines['left'].set_visible(False)
# f_ax_del3.spines['right'].set_visible(False)
# f_ax_del3.get_xaxis().set_visible(False)
# f_ax_del3.get_yaxis().set_visible(False)

f_ax12 = fig.add_subplot(gs[0:2, 9:12])
f_ax13 = fig.add_subplot(gs[2:4, 9:12])
f_ax14 = fig.add_subplot(gs[4:6, 9:12])

for axis in ['top','bottom','left','right']:
    f_ax14.spines[axis].set_visible(False)
f_ax14.get_xaxis().set_visible(False)
f_ax14.get_yaxis().set_visible(False)

cax = fig.add_axes([0.69, 0.0350117129629631, 0.009, 0.2715])
#------------------------------------------------------------------------------
#### Plot the data:
#------------------------------------------------------------------------------
# contour_label = np.append(np.arange(0.1,0.5,0.2), np.arange(0.5,20,5))
contour_label = 2
contour_label2 = 4
all_vmin = 0
rhoSi_vmin = -5
rhoSi_vmax = rhoSi_min_outlier
rhoC_vmax = rhoC_min_outlier
rhoN_vmax = rhoN_min_outlier
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# rhoC
h1 = f_ax1.contourf(np.unique(production_binned_2010['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2010, 100, vmin=all_vmin, vmax=rhoC_vmax, cmap=cm.jet)
h2 = f_ax1.contourf(np.unique(production_binned_2011['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2011, 100, vmin=all_vmin, vmax=rhoC_vmax, cmap=cm.jet)
# h3 = f_ax1.contourf(np.unique(production_binned_2012_2013['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2012_2013, 100, vmin=all_vmin, vmax=rhoC_vmax, cmap=cm.jet)
h4 = f_ax1.contourf(np.unique(production_binned_2014_2015['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2014_2015, 100, vmin=all_vmin, vmax=rhoC_vmax, cmap=cm.jet)
h5 = f_ax1.contourf(np.unique(production_binned_2016_2017['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2016_2017, 100, vmin=all_vmin, vmax=rhoC_vmax, cmap=cm.jet)

# plot sampling points:
# set marker size:
s = 100
# plot:
ind_grid = pd.notna(production_binned_2010['rhoC (umol/L)'])
s1 = f_ax1.scatter(production_binned_2010['Date'][ind_grid], production_binned_2010['Depth'][ind_grid], s, c=production_binned_2010['rhoC (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoC_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2011['rhoC (umol/L)'])
s2 = f_ax1.scatter(production_binned_2011['Date'][ind_grid], production_binned_2011['Depth'][ind_grid], s, c=production_binned_2011['rhoC (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoC_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2012_2013['rhoC (umol/L)'])
s3 = f_ax1.scatter(production_binned_2012_2013['Date'][ind_grid], production_binned_2012_2013['Depth'][ind_grid], s, c=production_binned_2012_2013['rhoC (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoC_vmax, edgecolor='k', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2014_2015['rhoC (umol/L)'])
s4 = f_ax1.scatter(production_binned_2014_2015['Date'][ind_grid], production_binned_2014_2015['Depth'][ind_grid], s, c=production_binned_2014_2015['rhoC (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoC_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2016_2017['rhoC (umol/L)'])
s5 = f_ax1.scatter(production_binned_2016_2017['Date'][ind_grid], production_binned_2016_2017['Depth'][ind_grid], s, c=production_binned_2016_2017['rhoC (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoC_vmax, edgecolor='w', cmap=cm.jet)


# Add contours:
contours1 = f_ax1.contour(np.unique(production_binned_2010['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2010, contour_label, colors='black')
f_ax1.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours2 = f_ax1.contour(np.unique(production_binned_2011['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2011, contour_label, colors='black')
f_ax1.clabel(contours2, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
# contours3 = f_ax1.contour(np.unique(production_binned_2012_2013['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2012_2013, contour_label, colors='black')
# f_ax1.clabel(contours3, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours4 = f_ax1.contour(np.unique(production_binned_2014_2015['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2014_2015, contour_label2, colors='black')
f_ax1.clabel(contours4, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours5 = f_ax1.contour(np.unique(production_binned_2016_2017['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2016_2017, contour_label2, colors='black')
f_ax1.clabel(contours5, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

# Add formatting finishing touches:
# f_ax1.set_xlabel('Time (YYYY-MM)')
f_ax1.set_ylim(0,20)
f_ax1.set_ylabel('Depth (m)')
# cbar = fig.colorbar(h5, ax=f_ax1, format="%1.1f")
# cbar.set_label(r' $\mathrm{\rho}$C ($\mathrm{\mu}$mol L$^{-1}$ d$^{-1}$)')
f_ax1.invert_yaxis()
f_axx1.set_title(r' $\mathbf{\rho}$C', loc='left', fontweight='bold')
f_ax1.xaxis.set_major_locator(years)
f_ax1.xaxis.set_major_formatter(years_fmt)
f_ax1.xaxis.set_minor_locator(months)

datemin = np.datetime64(saanich_integ['Date'][saanich_integ.index[0]], 'Y')
datemax = np.datetime64(saanich_integ['Date'][saanich_integ.index[-1]], 'Y') + np.timedelta64(1, 'Y')
f_ax1.set_xlim(datemin, datemax)

f_ax1.yaxis.set_minor_locator(MultipleLocator(1))
f_ax1.yaxis.set_major_locator(MultipleLocator(5))
f_ax1.tick_params(axis='both', which='major', length=10)
f_ax1.tick_params(axis='both', which='minor', length=6)
f_ax1.tick_params(axis='y', which='both', right='True', labelleft='on')

# labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
# labels[-1] = '>'+str("{:.1f}".format(rhoC_min_outlier))
# cbar.ax.set_yticklabels(labels)
#------------------------------------------------------------------------------
# rhoN
h6 = f_ax2.contourf(np.unique(production_binned_2010['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2010, 100, vmin=all_vmin, vmax=rhoN_vmax, cmap=cm.jet)
h7 = f_ax2.contourf(np.unique(production_binned_2011['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2011, 100, vmin=all_vmin, vmax=rhoN_vmax, cmap=cm.jet)
# h8 = f_ax2.contourf(np.unique(production_binned_2012_2013['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2012_2013, 100, vmin=all_vmin, vmax=rhoN_vmax, cmap=cm.jet)
h9 = f_ax2.contourf(np.unique(production_binned_2014_2015['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2014_2015, 100, vmin=all_vmin, vmax=rhoN_vmax, cmap=cm.jet)
h10 = f_ax2.contourf(np.unique(production_binned_2016_2017['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2016_2017, 100, vmin=all_vmin, vmax=rhoN_vmax, cmap=cm.jet)

# plot sampling points:
# set marker size:
s = 100
# plot:
ind_grid = pd.notna(production_binned_2010['rhoN (umol/L)'])
s1 = f_ax2.scatter(production_binned_2010['Date'][ind_grid], production_binned_2010['Depth'][ind_grid], s, c=production_binned_2010['rhoN (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoN_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2011['rhoN (umol/L)'])
s2 = f_ax2.scatter(production_binned_2011['Date'][ind_grid], production_binned_2011['Depth'][ind_grid], s, c=production_binned_2011['rhoN (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoN_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2012_2013['rhoN (umol/L)'])
s3 = f_ax2.scatter(production_binned_2012_2013['Date'][ind_grid], production_binned_2012_2013['Depth'][ind_grid], s, c=production_binned_2012_2013['rhoN (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoN_vmax, edgecolor='k', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2014_2015['rhoN (umol/L)'])
s4 = f_ax2.scatter(production_binned_2014_2015['Date'][ind_grid], production_binned_2014_2015['Depth'][ind_grid], s, c=production_binned_2014_2015['rhoN (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoN_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2016_2017['rhoN (umol/L)'])
s5 = f_ax2.scatter(production_binned_2016_2017['Date'][ind_grid], production_binned_2016_2017['Depth'][ind_grid], s, c=production_binned_2016_2017['rhoN (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoN_vmax, edgecolor='w', cmap=cm.jet)

# Add contours:
contours1 = f_ax2.contour(np.unique(production_binned_2010['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2010, contour_label, colors='black')
f_ax2.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours2 = f_ax2.contour(np.unique(production_binned_2011['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2011, contour_label, colors='black')
f_ax2.clabel(contours2, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
# contours3 = f_ax2.contour(np.unique(production_binned_2012_2013['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2012_2013, contour_label, colors='black')
# f_ax2.clabel(contours3, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours4 = f_ax2.contour(np.unique(production_binned_2014_2015['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2014_2015, contour_label2, colors='black')
f_ax2.clabel(contours4, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours5 = f_ax2.contour(np.unique(production_binned_2016_2017['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2016_2017, contour_label2, colors='black')
f_ax2.clabel(contours5, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

# Add formatting finishing touches:
# f_ax1.set_xlabel('Time (YYYY-MM)')
f_ax2.set_ylim(0,20)
f_ax2.set_ylabel('Depth (m)')
# f_ax2.set_xlabel('Time (Years)')
# cbar = fig.colorbar(h5, ax=f_ax2)
# cbar.set_label(r'$\mathrm{\rho}$N ($\mathrm{\mu}$mol L$^{-1}$ d$^{-1}$)')
f_ax2.invert_yaxis()
f_axx2.set_title(r' $\mathbf{\rho}$N', loc='left', fontweight='bold')
f_ax2.xaxis.set_major_locator(years)
f_ax2.xaxis.set_major_formatter(years_fmt)
f_ax2.xaxis.set_minor_locator(months)

datemin = np.datetime64(saanich_integ['Date'][saanich_integ.index[0]], 'Y')
datemax = np.datetime64(saanich_integ['Date'][saanich_integ.index[-1]], 'Y') + np.timedelta64(1, 'Y')
f_ax2.set_xlim(datemin, datemax)

f_ax2.yaxis.set_minor_locator(MultipleLocator(1))
f_ax2.yaxis.set_major_locator(MultipleLocator(5))
f_ax2.tick_params(axis='both', which='major', length=10)
f_ax2.tick_params(axis='both', which='minor', length=6)
f_ax2.tick_params(axis='y', which='both', right='True', labelleft='on')

# labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
# labels[-1] = '>'+str("{:.2f}".format(rhoN_min_outlier))
# cbar.ax.set_yticklabels(labels)
#------------------------------------------------------------------------------
# rhoSi
# h11 = f_ax3.contourf(np.unique(production_binned_2010['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2010, 100, vmin=rhoSi_vmin, vmax=rhoSi_vmax, cmap=cm.jet)
# h12 = f_ax3.contourf(np.unique(production_binned_2011['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2011, 100, vmin=rhoSi_vmin, vmax=rhoSi_vmax,  cmap=cm.jet)
# h13 = f_ax3.contourf(np.unique(production_binned_2012_2013['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2012_2013, 100, vmin=rhoSi_vmin, vmax=rhoSi_vmax,  cmap=cm.jet)
h14 = f_ax3.contourf(np.unique(production_binned_2014_2015['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2014_2015, 100, vmin=rhoSi_vmin, vmax=rhoSi_vmax,  cmap=cm.jet)
h15 = f_ax3.contourf(np.unique(production_binned_2016_2017['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2016_2017, 100, vmin=rhoSi_vmin, vmax=rhoSi_vmax,  cmap=cm.jet)


# plot sampling points:
# set marker size:
s = 100
# plot:
# ind_grid = pd.notna(production_binned_2010['rhoSi (umol/L)'])
# s1 = f_ax3.scatter(production_binned_2010['Date'][ind_grid], production_binned_2010['Depth'][ind_grid], s, c=production_binned_2010['rhoSi (umol/L)'][ind_grid], vmin=rhoSi_vmin, vmax=rhoSi_vmax,  edgecolor='k', cmap=cm.jet)
# ind_grid = pd.notna(production_binned_2011['rhoSi (umol/L)'])
# s2 = f_ax3.scatter(production_binned_2011['Date'][ind_grid], production_binned_2011['Depth'][ind_grid], s, c=production_binned_2011['rhoSi (umol/L)'][ind_grid], vmin=rhoSi_vmin, vmax=rhoSi_vmax,  edgecolor='k', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2012_2013['rhoSi (umol/L)'])
s3 = f_ax3.scatter(production_binned_2012_2013['Date'][ind_grid], production_binned_2012_2013['Depth'][ind_grid], s, c=production_binned_2012_2013['rhoSi (umol/L)'][ind_grid], vmin=rhoSi_vmin, vmax=rhoSi_vmax,  edgecolor='k', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2014_2015['rhoSi (umol/L)'])
s4 = f_ax3.scatter(production_binned_2014_2015['Date'][ind_grid], production_binned_2014_2015['Depth'][ind_grid], s, c=production_binned_2014_2015['rhoSi (umol/L)'][ind_grid], vmin=rhoSi_vmin, vmax=rhoSi_vmax,  edgecolor='k', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2016_2017['rhoSi (umol/L)'])
s5 = f_ax3.scatter(production_binned_2016_2017['Date'][ind_grid], production_binned_2016_2017['Depth'][ind_grid], s, c=production_binned_2016_2017['rhoSi (umol/L)'][ind_grid], vmin=rhoSi_vmin, vmax=rhoSi_vmax,  edgecolor='k', cmap=cm.jet)


# Add contours:
# contours1 = f_ax3.contour(np.unique(production_binned_2010['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2010, contour_label, colors='black')
# f_ax3.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
# contours2 = f_ax3.contour(np.unique(production_binned_2011['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2011, contour_label, colors='black')
# f_ax3.clabel(contours2, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
# contours3 = f_ax3.contour(np.unique(production_binned_2012_2013['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2012_2013, contour_label, colors='black')
# f_ax3.clabel(contours3, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours4 = f_ax3.contour(np.unique(production_binned_2014_2015['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2014_2015, contour_label2, colors='black')
f_ax3.clabel(contours4, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours5 = f_ax3.contour(np.unique(production_binned_2016_2017['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2016_2017, contour_label2, colors='black')
f_ax3.clabel(contours5, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

# Add formatting finishing touches:
# f_ax3.set_xlabel('Time (YYYY-MM)')
f_ax3.set_ylim([0, 20])
f_ax3.set_ylabel('Depth (m)')
f_ax3.set_xlabel('Time (Years)')
cbar = fig.colorbar(h15, cax=cax, pad=-0.02, aspect=30)
cbar.set_label(r'($\mathrm{\mu}$mol L$^{-1}$ d$^{-1}$)')
f_ax3.invert_yaxis()
f_axx3.set_title(r' $\mathbf{\rho}$Si', loc='left', fontweight='bold')

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

f_ax3.xaxis.set_major_locator(years)
f_ax3.xaxis.set_major_formatter(years_fmt)
f_ax3.xaxis.set_minor_locator(months)

datemin = np.datetime64(saanich_integ['Date'][saanich_integ.index[0]], 'Y')
datemax = np.datetime64(saanich_integ['Date'][saanich_integ.index[-1]], 'Y') + np.timedelta64(1, 'Y')
f_ax3.set_xlim(datemin, datemax)

f_ax3.yaxis.set_minor_locator(MultipleLocator(1))
f_ax3.yaxis.set_major_locator(MultipleLocator(5))
f_ax3.tick_params(axis='both', which='major', length=10)
f_ax3.tick_params(axis='both', which='minor', length=6)
f_ax3.tick_params(axis='y', which='both', right='True', labelleft='on')

labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
labels[-1] = '>'+str("{:.1f}".format(rhoSi_min_outlier))
cbar.ax.set_yticklabels(labels)

# add indication of different methods used:
f_ax3.axvspan(datetime.date(2013,2,1), datetime.date(2015,12,15), facecolor='blue', edgecolor='k', alpha=0.1, zorder=0, label=r'$^{32}$Si derived')
f_ax3.axvspan(datetime.date(2016,9,1), datetime.date(2017,10,15), facecolor='red', edgecolor='k', alpha=0.1, zorder=0, label=r'Net bSiO$_{2}$ derived')
f_ax3.legend(loc='upper left', fancybox=True, shadow=True, frameon=True)
#------------------------------------------------------------------------------
#### Add integrated values with growing seasons:
#------------------------------------------------------------------------------
# rhoC
ind1 = pd.notna(saanich_integ_2010.loc[:,'rhoC (mmol/m2/d)'])
ind2 = pd.notna(saanich_integ_2011.loc[:,'rhoC (mmol/m2/d)'])
# ind3 = pd.notna(saanich_integ_2012_2013.loc[:,'rhoC (mmol/m2/d)'])
ind4 = pd.notna(saanich_integ_2014_2015.loc[:,'rhoC (mmol/m2/d)'])
ind5 = pd.notna(saanich_integ_2016_2017.loc[:,'rhoC (mmol/m2/d)'])

f_ax9_a.plot(saanich_integ_2010['Date'][ind1], saanich_integ_2010.loc[:,'rhoC (mmol/m2/d)'][ind1], 'k--.', markersize=14)
f_ax9_a.plot(saanich_integ_2011['Date'][ind2], saanich_integ_2011.loc[:,'rhoC (mmol/m2/d)'][ind2], 'k--.', markersize=14)
# f_ax9_a.plot(saanich_integ_2012_2013['Date'][ind3], saanich_integ_2012_2013.loc[:,'rhoC (mmol/m2/d)'][ind3], 'k--.', markersize=14)
f_ax9_a.plot(saanich_integ_2014_2015['Date'][ind4], saanich_integ_2014_2015.loc[:,'rhoC (mmol/m2/d)'][ind4], 'k--.', markersize=14)
f_ax9_a.plot(saanich_integ_2016_2017['Date'][ind5], saanich_integ_2016_2017.loc[:,'rhoC (mmol/m2/d)'][ind5], 'k--.', markersize=14)

f_ax9_b.plot(saanich_integ_2010['Date'][ind1], saanich_integ_2010.loc[:,'rhoC (mmol/m2/d)'][ind1], 'k--.', markersize=14)
f_ax9_b.plot(saanich_integ_2011['Date'][ind2], saanich_integ_2011.loc[:,'rhoC (mmol/m2/d)'][ind2], 'k--.', markersize=14)
# f_ax9_b.plot(saanich_integ_2012_2013['Date'][ind3], saanich_integ_2012_2013.loc[:,'rhoC (mmol/m2/d)'][ind3], 'k--.', markersize=14)
f_ax9_b.plot(saanich_integ_2014_2015['Date'][ind4], saanich_integ_2014_2015.loc[:,'rhoC (mmol/m2/d)'][ind4], 'k--.', markersize=14)
f_ax9_b.plot(saanich_integ_2016_2017['Date'][ind5], saanich_integ_2016_2017.loc[:,'rhoC (mmol/m2/d)'][ind5], 'k--.', markersize=14)

d = .25  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=2, clip_on=False)
f_ax9_b.plot([0, 1], [0, 0], transform=f_ax9_b.transAxes, **kwargs)
f_ax9_a.plot([0, 1], [1, 1], transform=f_ax9_a.transAxes, **kwargs)


# f_ax9.set_ylabel('(mmol m$^{-2}$ d$^{-1}$)')
f_ax9_a.set_ylabel('       (mmol m$^{-2}$ d$^{-1}$) \n')
# f_ax9.set_xlim(np.min(saanich_integ['Date']),np.max(saanich_integ['Date']))
f_ax9_a.set_xlim(datemin, datemax)
f_ax9_b.set_xlim(datemin, datemax)
# f_ax9.autoscale(enable=True, axis='y', tight=True)
# f_ax9.ticklabel_format(axis='y', scilimits=(0,3), useMathText=True) # set tick labels to scientific notation
# t = f_ax9.yaxis.get_offset_text()
# t.set_x(-0.03) # offset scientific notation above ticks
# f_ax9.set_ylim(0, 16000)
# f_ax9.yaxis.set_ticks(np.arange(0, 16000, 5000))
f_ax9_a.tick_params(axis='y', which='major', length=10)
f_ax9_a.tick_params(axis='y', which='minor', length=6)

f_ax9_b.tick_params(axis='y', which='major', length=10)
f_ax9_b.tick_params(axis='y', which='minor', length=6)
# labels = f_ax9.get_yticks().tolist()
# labels[0] = ''
# f_ax9.set_yticklabels(labels)

# highlight growing seasons:
f_ax9_a.axvspan(datetime.date(2010,3,1), datetime.date(2010,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_a.axvspan(datetime.date(2011,3,1), datetime.date(2011,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_a.axvspan(datetime.date(2012,3,1), datetime.date(2012,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_a.axvspan(datetime.date(2013,3,1), datetime.date(2013,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_a.axvspan(datetime.date(2014,3,1), datetime.date(2014,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_a.axvspan(datetime.date(2015,3,1), datetime.date(2015,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_a.axvspan(datetime.date(2016,3,1), datetime.date(2016,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_a.axvspan(datetime.date(2017,3,1), datetime.date(2017,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)

f_ax9_b.axvspan(datetime.date(2010,3,1), datetime.date(2010,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_b.axvspan(datetime.date(2011,3,1), datetime.date(2011,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_b.axvspan(datetime.date(2012,3,1), datetime.date(2012,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_b.axvspan(datetime.date(2013,3,1), datetime.date(2013,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_b.axvspan(datetime.date(2014,3,1), datetime.date(2014,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_b.axvspan(datetime.date(2015,3,1), datetime.date(2015,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_b.axvspan(datetime.date(2016,3,1), datetime.date(2016,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_b.axvspan(datetime.date(2017,3,1), datetime.date(2017,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)

f_ax9_a.set_ylim([0,750])
f_ax9_b.set_ylim([1200,1300])
f_ax9_a.yaxis.set_major_locator(MultipleLocator(200))
f_ax9_a.tick_params(axis='y', which='both', right='True', labelleft='on')
f_ax9_b.tick_params(axis='y', which='both', right='True', labelleft='on')

# f_ax10.set_ylim(0,2000)
# rhoN
ind1 = pd.notna(saanich_integ_2010.loc[:,'rhoN (mmol/m2/d)'])
ind2 = pd.notna(saanich_integ_2011.loc[:,'rhoN (mmol/m2/d)'])
# ind3 = pd.notna(saanich_integ_2012_2013.loc[:,29])
ind4 = pd.notna(saanich_integ_2014_2015.loc[:,'rhoN (mmol/m2/d)'])
ind5 = pd.notna(saanich_integ_2016_2017.loc[:,'rhoN (mmol/m2/d)'])
f_ax10.plot(saanich_integ_2010['Date'][ind1], saanich_integ_2010.loc[:,'rhoN (mmol/m2/d)'][ind1], 'k--.', markersize=14)
f_ax10.plot(saanich_integ_2011['Date'][ind2], saanich_integ_2011.loc[:,'rhoN (mmol/m2/d)'][ind2], 'k--.', markersize=14)
# f_ax10.plot(saanich_integ_2012_2013['Date'][ind3], saanich_integ_2012_2013.loc[:,'rhoN (mmol/m2/d)'][ind3], 'k--.', markersize=14)
f_ax10.plot(saanich_integ_2014_2015['Date'][ind4], saanich_integ_2014_2015.loc[:,'rhoN (mmol/m2/d)'][ind4], 'k--.', markersize=14)
f_ax10.plot(saanich_integ_2016_2017['Date'][ind5], saanich_integ_2016_2017.loc[:,'rhoN (mmol/m2/d)'][ind5], 'k--.', markersize=14)
f_ax10.set_ylabel('(mmol m$^{-2}$ d$^{-1}$)')
# f_ax10.set_xlim(np.min(saanich_integ['Date']),np.max(saanich_integ['Date']))
f_ax10.set_xlim(datemin, datemax)
f_ax10.autoscale(enable=True, axis='y', tight=True)
f_ax10.tick_params(axis='y', which='major', length=10)
f_ax10.tick_params(axis='y', which='minor', length=6)
start, end = f_ax10.get_ylim()
f_ax10.yaxis.set_ticks(np.arange(0, 61, 10))

# highlight growing seasons:
f_ax10.axvspan(datetime.date(2010,3,1), datetime.date(2010,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2011,3,1), datetime.date(2011,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2012,3,1), datetime.date(2012,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2013,3,1), datetime.date(2013,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2014,3,1), datetime.date(2014,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2015,3,1), datetime.date(2015,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2016,3,1), datetime.date(2016,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2017,3,1), datetime.date(2017,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)

f_ax10.tick_params(axis='y', which='both', right='True', labelleft='on')

# f_ax11.set_ylim(0,2000)

# rhoSi
f_ax11.hlines(0, datetime.date(2010,1,1), datetime.date(2017,12,31), 'r', 'dashed', linewidth=2.5)

ind1 = pd.notna(saanich_integ_2014_2015.loc[:,'bSiO2 Precipitation'])
ind2 = pd.notna(saanich_integ_2016_2017.loc[:,'bSiO2 Precipitation'])
f_ax11.plot(saanich_integ_2014_2015['Date'][ind1], saanich_integ_2014_2015['bSiO2 Precipitation'][ind1], 'k--.', markersize=14)
f_ax11.plot(saanich_integ_2016_2017['Date'][ind2], saanich_integ_2016_2017['bSiO2 Precipitation'][ind2], 'k--.', markersize=14)
f_ax11.set_ylabel('(mmol m$^{-2}$ d$^{-1}$)')
# f_ax11.set_xlim(np.min(saanich_integ['Date']),np.max(saanich_integ['Date']))
f_ax11.set_xlim(datemin, datemax)
f_ax11.autoscale(enable=True, axis='y', tight=True)
f_ax11.tick_params(axis='y', which='major', length=10)
f_ax11.tick_params(axis='y', which='minor', length=6)
start, end = f_ax11.get_ylim()
f_ax11.yaxis.set_ticks(np.arange(0, end, 50))
start, end = f_ax11.get_ylim()
f_ax11.yaxis.set_ticks(np.append(np.arange(start,0,start),np.arange(0,81,20)))
f_ax11.set_ylim(-20,80)

# highlight growing seasons:
color_code = 'dimgray'
# f_ax11.axvspan(datetime.date(2010,3,1), datetime.date(2010,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
# f_ax11.axvspan(datetime.date(2011,3,1), datetime.date(2011,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
# f_ax11.axvspan(datetime.date(2012,3,1), datetime.date(2012,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2013,3,1), datetime.date(2013,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2014,3,1), datetime.date(2014,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2015,3,1), datetime.date(2015,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2016,3,1), datetime.date(2016,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2017,3,1), datetime.date(2017,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)

# add indication of different methods used:
f_ax11.axvspan(datetime.date(2013,2,1), datetime.date(2015,12,15), facecolor='blue', edgecolor='k', alpha=0.1, zorder=0, label=r'$^{32}$Si derived')
f_ax11.axvspan(datetime.date(2016,9,1), datetime.date(2017,10,15), facecolor='red', edgecolor='k', alpha=0.1, zorder=0, label=r'Net bSiO$_{2}$ derived')
# f_ax11.legend(loc='upper left', fancybox=True, shadow=True, frameon=True)
f_ax11.tick_params(axis='y', which='both', right='True', labelleft='on')
#------------------------------------------------------------------------------
#### Plot seasonally averaged data:
total_depth = np.max(production_binned['Depth'])+1

# rhoC
h16 = f_ax12.contourf(np.unique(production_binned['Month']),np.arange(0,total_depth,1),rhoC_seas_wide_interp, 100, cmap=cm.jet)

ind_grid = pd.notna(production_binned['rhoC (umol/L)'])
s1 = f_ax12.scatter(production_binned['Month'][ind_grid], production_binned['Depth'][ind_grid], 100, c=production_binned['rhoC (umol/L)'][ind_grid], edgecolor='w', cmap=cm.jet, clip_on=False)

contours1 = f_ax12.contour(np.unique(production_binned['Month']),np.arange(0,total_depth,1),rhoC_seas_wide_interp, contour_label, colors='black')
f_ax12.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

f_ax12.set_ylabel('Depth (m)')
# f_ax12.set_xlabel('Month')
f_ax12.set_xlim([1,12])
cbar = fig.colorbar(h16, ax=f_ax12, format="%1.1f", pad=-0.02, aspect=30)
cbar.set_label(r'($\mathrm{\mu}$mol L$^{-1}$ d$^{-1}$)')
f_ax12.invert_yaxis()
f_ax12.set_title(r' $\mathbf{\rho}$C', loc='left', fontweight='bold')

f_ax12.set_xticks(np.arange(1,13,1))
f_ax12.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)

labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
labels[-1] = '>'+str("{:.1f}".format(rhoC_min_outlier))
cbar.ax.set_yticklabels(labels)

f_ax12.yaxis.set_minor_locator(MultipleLocator(1))
f_ax12.tick_params(axis='both', which='major', length=10)
f_ax12.tick_params(axis='y', which='minor', length=6)
f_ax12.tick_params(axis='y', which='both', right='True', labelleft='on')

# rhoN
h17 = f_ax13.contourf(np.unique(production_binned['Month']),np.arange(0,total_depth,1),rhoN_seas_wide_interp, 100, cmap=cm.jet)

ind_grid = pd.notna(production_binned['rhoN (umol/L)'])
s1 = f_ax13.scatter(production_binned['Month'][ind_grid], production_binned['Depth'][ind_grid], 100, c=production_binned['rhoN (umol/L)'][ind_grid], edgecolor='w', cmap=cm.jet, clip_on=False)

contours1 = f_ax13.contour(np.unique(production_binned['Month']),np.arange(0,total_depth,1),rhoN_seas_wide_interp, contour_label, colors='black')
f_ax13.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

f_ax13.set_ylabel('Depth (m)')
f_ax13.set_xlabel('Month')
f_ax13.set_xlim([1,12])
cbar = fig.colorbar(h17, ax=f_ax13, pad=-0.02, aspect=30)
cbar.set_label(r'($\mathrm{\mu}$mol L$^{-1}$ d$^{-1}$)')
f_ax13.invert_yaxis()
f_ax13.set_title(r' $\mathbf{\rho}$N', loc='left', fontweight='bold')

f_ax13.set_xticks(np.arange(1,13,1))
f_ax13.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)

labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
labels[-1] = '>'+str("{:.2f}".format(rhoN_min_outlier))
cbar.ax.set_yticklabels(labels)

f_ax13.yaxis.set_minor_locator(MultipleLocator(1))
f_ax13.tick_params(axis='both', which='major', length=10)
f_ax13.tick_params(axis='y', which='minor', length=6)
f_ax13.tick_params(axis='y', which='both', right='True', labelleft='on')

# # rhoSi
# contour_label = 5
# h1 = f_ax14.contourf(np.unique(production_binned['Month']),np.arange(0,total_depth,1),rhoSi_seas_wide_interp, 100, cmap=cm.jet)

# ind_grid = pd.notna(production_binned['rhoSi (umol/L)'])
# s1 = f_ax14.scatter(production_binned['Month'][ind_grid], production_binned['Depth'][ind_grid], 50, c=production_binned['rhoSi (umol/L)'][ind_grid], edgecolor='k', cmap=cm.jet, clip_on=False)

# contours1 = f_ax14.contour(np.unique(production_binned['Month']),np.arange(0,total_depth,1),rhoSi_seas_wide_interp, contour_label, colors='black')
# f_ax14.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

# f_ax14.set_ylabel('Depth (m)')
# # f_ax14.set_xlabel('Month')
# f_ax14.set_xlim([1,12])
# cbar = fig.colorbar(h1, ax=f_ax14, format="%1.1f")
# cbar.set_label(r' $\mathrm{\rho}$Si ($\mathrm{\mu}$mol L$^{-1}$ d$^{-1}$)')
# f_ax14.invert_yaxis()
# f_ax14.set_title(r' $\mathbf{\rho}$Si', loc='left', fontweight='bold')

# f_ax14.set_xticks(np.arange(1,13,1))
# f_ax14.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=90)

# labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
# labels[-1] = '>'+str("{:.1f}".format(rhoSi_min_outlier))
# cbar.ax.set_yticklabels(labels)

# f_ax14.yaxis.set_minor_locator(MultipleLocator(1))
# f_ax14.tick_params(axis='both', which='major', length=10)
# f_ax14.tick_params(axis='y', which='minor', length=6)
#------------------------------------------------------------------------------

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### Collate figure handles for a vector graphic format:
h_prod1 = [h1,h2,h4,h5,h6,h7,h9,h10,h14,h15,h16,h17] # collate plot handles
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% Production: contour plots
total_depth = np.max(production_binned_2010['Depth'])+1 # max depth (+1 to compensate to include last depth in plot)
#------------------------------------------------------------------------------
#### Set-up figure layout:
fig = plt.figure(figsize=(34,24), constrained_layout=True)
font={'family':'DejaVu Sans',
      'weight':'normal',
      'size':'22'} 
plt.rc('font', **font) # sets the specified font formatting globally

gs = fig.add_gridspec(6, 12)
# main plots
f_axx1 = fig.add_subplot(gs[0:2, :-3])
for axis in ['left','right']:
    f_axx1.spines[axis].set_visible(False)
f_axx1.get_xaxis().set_ticks([])
f_axx1.get_yaxis().set_ticks([])

f_ax1 = f_axx1.inset_axes([0, 0, 1, 0.75])
f_ax1.spines['bottom'].set_visible(False)
# f_ax1.set_position([0.06996741830065356, 0.7093259490740736, 0.765, 0.2])

f_ax9 = f_axx1.inset_axes([0, 0.75, 1, 0.25])
f_ax9.spines['bottom'].set_visible(False)
f_ax9.get_xaxis().set_ticks([])

# add axes for axis break
for axis in ['left','right']:
    f_ax9.spines[axis].set_visible(False)
f_ax9.get_xaxis().set_visible(False)
f_ax9.get_yaxis().set_visible(False)

f_ax9_a = f_ax9.inset_axes([0, 0, 1, 0.7])
f_ax9_a.spines['top'].set_visible(False)
f_ax9_a.spines['bottom'].set_visible(False)
f_ax9_a.get_xaxis().set_ticks([])

f_ax9_b = f_ax9.inset_axes([0,0.8,1,0.2])
f_ax9_b.spines['bottom'].set_visible(False)
f_ax9_b.get_xaxis().set_ticks([])
#------------------------------------------------------------------------------
f_axx2 = fig.add_subplot(gs[2:4, :-3])
f_axx2.get_xaxis().set_ticks([])
f_axx2.get_yaxis().set_ticks([])

f_ax2 = f_axx2.inset_axes([0, 0, 1, 0.75])
f_ax2.spines['bottom'].set_visible(False)
# f_ax2.set_position([0.06137877450980385, 0.3803791898148143, 0.775, 0.2])

f_ax10 = f_axx2.inset_axes([0, 0.75, 1, 0.25])
f_ax10.spines['bottom'].set_visible(False)
f_ax10.get_xaxis().set_ticks([])
#------------------------------------------------------------------------------
f_axx3 = fig.add_subplot(gs[4:6, :-3])
f_axx3.get_xaxis().set_ticks([])
f_axx3.get_yaxis().set_ticks([])

f_ax3 = f_axx3.inset_axes([0, 0, 1, 0.75])
f_ax3.spines['bottom'].set_visible(False)
# f_ax3.set_position([0.06137877450980389, 0.04593474537037023, 0.775, 0.2])

f_ax11 = f_axx3.inset_axes([0, 0.75, 1, 0.25])
f_ax11.spines['bottom'].set_visible(False)
f_ax11.get_xaxis().set_ticks([])
#------------------------------------------------------------------------------
# f_ax_del1 = fig.add_subplot(gs[0:2, 8:9])
# f_ax_del1.spines['top'].set_visible(False)
# f_ax_del1.spines['bottom'].set_visible(False)
# f_ax_del1.spines['left'].set_visible(False)
# f_ax_del1.spines['right'].set_visible(False)
# f_ax_del1.get_xaxis().set_visible(False)
# f_ax_del1.get_yaxis().set_visible(False)

# f_ax_del2 = fig.add_subplot(gs[2:4, 8:9])
# f_ax_del2.spines['top'].set_visible(False)
# f_ax_del2.spines['bottom'].set_visible(False)
# f_ax_del2.spines['left'].set_visible(False)
# f_ax_del2.spines['right'].set_visible(False)
# f_ax_del2.get_xaxis().set_visible(False)
# f_ax_del2.get_yaxis().set_visible(False)

# f_ax_del3 = fig.add_subplot(gs[4:6, 8:9])
# f_ax_del3.spines['top'].set_visible(False)
# f_ax_del3.spines['bottom'].set_visible(False)
# f_ax_del3.spines['left'].set_visible(False)
# f_ax_del3.spines['right'].set_visible(False)
# f_ax_del3.get_xaxis().set_visible(False)
# f_ax_del3.get_yaxis().set_visible(False)

f_ax12 = fig.add_subplot(gs[0:2, 9:12])
f_ax13 = fig.add_subplot(gs[2:4, 9:12])
f_ax14 = fig.add_subplot(gs[4:6, 9:12])

for axis in ['top','bottom','left','right']:
    f_ax14.spines[axis].set_visible(False)
f_ax14.get_xaxis().set_visible(False)
f_ax14.get_yaxis().set_visible(False)

cax = fig.add_axes([0.69, 0.0350117129629631, 0.009, 0.2715])
#------------------------------------------------------------------------------
#### Plot the data:
#------------------------------------------------------------------------------
# contour_label = np.append(np.arange(0.1,0.5,0.2), np.arange(0.5,20,5))
contour_label = 2
contour_label2 = 4
all_vmin = 0
rhoSi_vmin = -5
rhoSi_vmax = rhoSi_min_outlier
rhoC_vmax = rhoC_min_outlier
rhoN_vmax = rhoN_min_outlier
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# rhoC
h1 = f_ax1.contourf(np.unique(production_binned_2010['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2010, 100, vmin=all_vmin, vmax=rhoC_vmax, cmap=cm.jet)
h2 = f_ax1.contourf(np.unique(production_binned_2011['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2011, 100, vmin=all_vmin, vmax=rhoC_vmax, cmap=cm.jet)
# h3 = f_ax1.contourf(np.unique(production_binned_2012_2013['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2012_2013, 100, vmin=all_vmin, vmax=rhoC_vmax, cmap=cm.jet)
h4 = f_ax1.contourf(np.unique(production_binned_2014_2015['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2014_2015, 100, vmin=all_vmin, vmax=rhoC_vmax, cmap=cm.jet)
h5 = f_ax1.contourf(np.unique(production_binned_2016_2017['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2016_2017, 100, vmin=all_vmin, vmax=rhoC_vmax, cmap=cm.jet)

# plot sampling points:
# set marker size:
s = 100
# plot:
ind_grid = pd.notna(production_binned_2010['rhoC (umol/L)'])
s1 = f_ax1.scatter(production_binned_2010['Date'][ind_grid], production_binned_2010['Depth'][ind_grid], s, c=production_binned_2010['rhoC (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoC_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2011['rhoC (umol/L)'])
s2 = f_ax1.scatter(production_binned_2011['Date'][ind_grid], production_binned_2011['Depth'][ind_grid], s, c=production_binned_2011['rhoC (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoC_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2012_2013['rhoC (umol/L)'])
s3 = f_ax1.scatter(production_binned_2012_2013['Date'][ind_grid], production_binned_2012_2013['Depth'][ind_grid], s, c=production_binned_2012_2013['rhoC (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoC_vmax, edgecolor='k', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2014_2015['rhoC (umol/L)'])
s4 = f_ax1.scatter(production_binned_2014_2015['Date'][ind_grid], production_binned_2014_2015['Depth'][ind_grid], s, c=production_binned_2014_2015['rhoC (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoC_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2016_2017['rhoC (umol/L)'])
s5 = f_ax1.scatter(production_binned_2016_2017['Date'][ind_grid], production_binned_2016_2017['Depth'][ind_grid], s, c=production_binned_2016_2017['rhoC (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoC_vmax, edgecolor='w', cmap=cm.jet)


# Add contours:
contours1 = f_ax1.contour(np.unique(production_binned_2010['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2010, contour_label, colors='black')
f_ax1.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours2 = f_ax1.contour(np.unique(production_binned_2011['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2011, contour_label, colors='black')
f_ax1.clabel(contours2, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
# contours3 = f_ax1.contour(np.unique(production_binned_2012_2013['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2012_2013, contour_label, colors='black')
# f_ax1.clabel(contours3, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours4 = f_ax1.contour(np.unique(production_binned_2014_2015['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2014_2015, contour_label2, colors='black')
f_ax1.clabel(contours4, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours5 = f_ax1.contour(np.unique(production_binned_2016_2017['Date']),np.arange(0,total_depth,1),rhoC_wide_interp_2016_2017, contour_label2, colors='black')
f_ax1.clabel(contours5, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

# Add formatting finishing touches:
# f_ax1.set_xlabel('Time (YYYY-MM)')
f_ax1.set_ylim(0,20)
f_ax1.set_ylabel('Depth (m)')
# cbar = fig.colorbar(h5, ax=f_ax1, format="%1.1f")
# cbar.set_label(r' $\mathrm{\rho}$C ($\mathrm{\mu}$mol L$^{-1}$ d$^{-1}$)')
f_ax1.invert_yaxis()
f_axx1.set_title(r' $\mathit{\rho}$C', loc='left', fontweight='bold')
f_ax1.xaxis.set_major_locator(years)
f_ax1.xaxis.set_major_formatter(years_fmt)
f_ax1.xaxis.set_minor_locator(months)

datemin = np.datetime64(saanich_integ['Date'][saanich_integ.index[0]], 'Y')
datemax = np.datetime64(saanich_integ['Date'][saanich_integ.index[-1]], 'Y') + np.timedelta64(1, 'Y')
f_ax1.set_xlim(datemin, datemax)

f_ax1.yaxis.set_minor_locator(MultipleLocator(1))
f_ax1.yaxis.set_major_locator(MultipleLocator(5))
f_ax1.tick_params(axis='both', which='major', length=10)
f_ax1.tick_params(axis='both', which='minor', length=6)
f_ax1.tick_params(axis='y', which='both', right='True', labelleft='on')

# labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
# labels[-1] = '>'+str("{:.1f}".format(rhoC_min_outlier))
# cbar.ax.set_yticklabels(labels)
#------------------------------------------------------------------------------
# rhoN
h6 = f_ax2.contourf(np.unique(production_binned_2010['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2010, 100, vmin=all_vmin, vmax=rhoN_vmax, cmap=cm.jet)
h7 = f_ax2.contourf(np.unique(production_binned_2011['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2011, 100, vmin=all_vmin, vmax=rhoN_vmax, cmap=cm.jet)
# h8 = f_ax2.contourf(np.unique(production_binned_2012_2013['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2012_2013, 100, vmin=all_vmin, vmax=rhoN_vmax, cmap=cm.jet)
h9 = f_ax2.contourf(np.unique(production_binned_2014_2015['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2014_2015, 100, vmin=all_vmin, vmax=rhoN_vmax, cmap=cm.jet)
h10 = f_ax2.contourf(np.unique(production_binned_2016_2017['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2016_2017, 100, vmin=all_vmin, vmax=rhoN_vmax, cmap=cm.jet)

# plot sampling points:
# set marker size:
s = 100
# plot:
ind_grid = pd.notna(production_binned_2010['rhoN (umol/L)'])
s1 = f_ax2.scatter(production_binned_2010['Date'][ind_grid], production_binned_2010['Depth'][ind_grid], s, c=production_binned_2010['rhoN (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoN_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2011['rhoN (umol/L)'])
s2 = f_ax2.scatter(production_binned_2011['Date'][ind_grid], production_binned_2011['Depth'][ind_grid], s, c=production_binned_2011['rhoN (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoN_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2012_2013['rhoN (umol/L)'])
s3 = f_ax2.scatter(production_binned_2012_2013['Date'][ind_grid], production_binned_2012_2013['Depth'][ind_grid], s, c=production_binned_2012_2013['rhoN (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoN_vmax, edgecolor='k', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2014_2015['rhoN (umol/L)'])
s4 = f_ax2.scatter(production_binned_2014_2015['Date'][ind_grid], production_binned_2014_2015['Depth'][ind_grid], s, c=production_binned_2014_2015['rhoN (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoN_vmax, edgecolor='w', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2016_2017['rhoN (umol/L)'])
s5 = f_ax2.scatter(production_binned_2016_2017['Date'][ind_grid], production_binned_2016_2017['Depth'][ind_grid], s, c=production_binned_2016_2017['rhoN (umol/L)'][ind_grid], vmin=all_vmin, vmax=rhoN_vmax, edgecolor='w', cmap=cm.jet)

# Add contours:
contours1 = f_ax2.contour(np.unique(production_binned_2010['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2010, contour_label, colors='black')
f_ax2.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours2 = f_ax2.contour(np.unique(production_binned_2011['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2011, contour_label, colors='black')
f_ax2.clabel(contours2, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
# contours3 = f_ax2.contour(np.unique(production_binned_2012_2013['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2012_2013, contour_label, colors='black')
# f_ax2.clabel(contours3, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours4 = f_ax2.contour(np.unique(production_binned_2014_2015['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2014_2015, contour_label2, colors='black')
f_ax2.clabel(contours4, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours5 = f_ax2.contour(np.unique(production_binned_2016_2017['Date']),np.arange(0,total_depth,1),rhoN_wide_interp_2016_2017, contour_label2, colors='black')
f_ax2.clabel(contours5, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

# Add formatting finishing touches:
# f_ax1.set_xlabel('Time (YYYY-MM)')
f_ax2.set_ylim(0,20)
f_ax2.set_ylabel('Depth (m)')
# f_ax2.set_xlabel('Time (Years)')
# cbar = fig.colorbar(h5, ax=f_ax2)
# cbar.set_label(r'$\mathrm{\rho}$N ($\mathrm{\mu}$mol L$^{-1}$ d$^{-1}$)')
f_ax2.invert_yaxis()
f_axx2.set_title(r' $\mathit{\rho}$N', loc='left', fontweight='bold')
f_ax2.xaxis.set_major_locator(years)
f_ax2.xaxis.set_major_formatter(years_fmt)
f_ax2.xaxis.set_minor_locator(months)

datemin = np.datetime64(saanich_integ['Date'][saanich_integ.index[0]], 'Y')
datemax = np.datetime64(saanich_integ['Date'][saanich_integ.index[-1]], 'Y') + np.timedelta64(1, 'Y')
f_ax2.set_xlim(datemin, datemax)

f_ax2.yaxis.set_minor_locator(MultipleLocator(1))
f_ax2.yaxis.set_major_locator(MultipleLocator(5))
f_ax2.tick_params(axis='both', which='major', length=10)
f_ax2.tick_params(axis='both', which='minor', length=6)
f_ax2.tick_params(axis='y', which='both', right='True', labelleft='on')

# labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
# labels[-1] = '>'+str("{:.2f}".format(rhoN_min_outlier))
# cbar.ax.set_yticklabels(labels)
#------------------------------------------------------------------------------
# rhoSi
# h11 = f_ax3.contourf(np.unique(production_binned_2010['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2010, 100, vmin=rhoSi_vmin, vmax=rhoSi_vmax, cmap=cm.jet)
# h12 = f_ax3.contourf(np.unique(production_binned_2011['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2011, 100, vmin=rhoSi_vmin, vmax=rhoSi_vmax,  cmap=cm.jet)
# h13 = f_ax3.contourf(np.unique(production_binned_2012_2013['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2012_2013, 100, vmin=rhoSi_vmin, vmax=rhoSi_vmax,  cmap=cm.jet)
h14 = f_ax3.contourf(np.unique(production_binned_2014_2015['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2014_2015, 100, vmin=rhoSi_vmin, vmax=rhoSi_vmax,  cmap=cm.jet)
h15 = f_ax3.contourf(np.unique(production_binned_2016_2017['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2016_2017, 100, vmin=rhoSi_vmin, vmax=rhoSi_vmax,  cmap=cm.jet)


# plot sampling points:
# set marker size:
s = 100
# plot:
# ind_grid = pd.notna(production_binned_2010['rhoSi (umol/L)'])
# s1 = f_ax3.scatter(production_binned_2010['Date'][ind_grid], production_binned_2010['Depth'][ind_grid], s, c=production_binned_2010['rhoSi (umol/L)'][ind_grid], vmin=rhoSi_vmin, vmax=rhoSi_vmax,  edgecolor='k', cmap=cm.jet)
# ind_grid = pd.notna(production_binned_2011['rhoSi (umol/L)'])
# s2 = f_ax3.scatter(production_binned_2011['Date'][ind_grid], production_binned_2011['Depth'][ind_grid], s, c=production_binned_2011['rhoSi (umol/L)'][ind_grid], vmin=rhoSi_vmin, vmax=rhoSi_vmax,  edgecolor='k', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2012_2013['rhoSi (umol/L)'])
s3 = f_ax3.scatter(production_binned_2012_2013['Date'][ind_grid], production_binned_2012_2013['Depth'][ind_grid], s, c=production_binned_2012_2013['rhoSi (umol/L)'][ind_grid], vmin=rhoSi_vmin, vmax=rhoSi_vmax,  edgecolor='k', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2014_2015['rhoSi (umol/L)'])
s4 = f_ax3.scatter(production_binned_2014_2015['Date'][ind_grid], production_binned_2014_2015['Depth'][ind_grid], s, c=production_binned_2014_2015['rhoSi (umol/L)'][ind_grid], vmin=rhoSi_vmin, vmax=rhoSi_vmax,  edgecolor='k', cmap=cm.jet)
ind_grid = pd.notna(production_binned_2016_2017['rhoSi (umol/L)'])
s5 = f_ax3.scatter(production_binned_2016_2017['Date'][ind_grid], production_binned_2016_2017['Depth'][ind_grid], s, c=production_binned_2016_2017['rhoSi (umol/L)'][ind_grid], vmin=rhoSi_vmin, vmax=rhoSi_vmax,  edgecolor='k', cmap=cm.jet)


# Add contours:
# contours1 = f_ax3.contour(np.unique(production_binned_2010['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2010, contour_label, colors='black')
# f_ax3.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
# contours2 = f_ax3.contour(np.unique(production_binned_2011['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2011, contour_label, colors='black')
# f_ax3.clabel(contours2, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
# contours3 = f_ax3.contour(np.unique(production_binned_2012_2013['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2012_2013, contour_label, colors='black')
# f_ax3.clabel(contours3, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours4 = f_ax3.contour(np.unique(production_binned_2014_2015['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2014_2015, contour_label2, colors='black')
f_ax3.clabel(contours4, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
contours5 = f_ax3.contour(np.unique(production_binned_2016_2017['Date']),np.arange(0,total_depth,1),rhoSi_wide_interp_2016_2017, contour_label2, colors='black')
f_ax3.clabel(contours5, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

# Add formatting finishing touches:
# f_ax3.set_xlabel('Time (YYYY-MM)')
f_ax3.set_ylim([0, 20])
f_ax3.set_ylabel('Depth (m)')
f_ax3.set_xlabel('Time (Years)')
cbar = fig.colorbar(h15, cax=cax, pad=-0.02, aspect=30)
cbar.set_label(r'($\mathrm{\mu}$mol L$^{-1}$ d$^{-1}$)')
f_ax3.invert_yaxis()
f_axx3.set_title(r' $\mathit{\rho}$Si', loc='left', fontweight='bold')

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

f_ax3.xaxis.set_major_locator(years)
f_ax3.xaxis.set_major_formatter(years_fmt)
f_ax3.xaxis.set_minor_locator(months)

datemin = np.datetime64(saanich_integ['Date'][saanich_integ.index[0]], 'Y')
datemax = np.datetime64(saanich_integ['Date'][saanich_integ.index[-1]], 'Y') + np.timedelta64(1, 'Y')
f_ax3.set_xlim(datemin, datemax)

f_ax3.yaxis.set_minor_locator(MultipleLocator(1))
f_ax3.yaxis.set_major_locator(MultipleLocator(5))
f_ax3.tick_params(axis='both', which='major', length=10)
f_ax3.tick_params(axis='both', which='minor', length=6)
f_ax3.tick_params(axis='y', which='both', right='True', labelleft='on')

labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
labels[-1] = '>'+str("{:.1f}".format(rhoSi_min_outlier))
cbar.ax.set_yticklabels(labels)

# add indication of different methods used:
f_ax3.axvspan(datetime.date(2013,2,1), datetime.date(2015,12,15), facecolor='blue', edgecolor='k', alpha=0.1, zorder=0, label=r'$^{32}$Si derived')
f_ax3.axvspan(datetime.date(2016,9,1), datetime.date(2017,10,15), facecolor='red', edgecolor='k', alpha=0.1, zorder=0, label=r'Net bSiO$_{2}$ derived')
f_ax3.legend(loc='upper left', fancybox=True, shadow=True, frameon=True)
#------------------------------------------------------------------------------
#### Add integrated values with growing seasons:
#------------------------------------------------------------------------------
# rhoC
ind1 = pd.notna(saanich_integ_2010.loc[:,'rhoC (mmol/m2/d)'])
ind2 = pd.notna(saanich_integ_2011.loc[:,'rhoC (mmol/m2/d)'])
# ind3 = pd.notna(saanich_integ_2012_2013.loc[:,'rhoC (mmol/m2/d)'])
ind4 = pd.notna(saanich_integ_2014_2015.loc[:,'rhoC (mmol/m2/d)'])
ind5 = pd.notna(saanich_integ_2016_2017.loc[:,'rhoC (mmol/m2/d)'])

f_ax9_a.plot(saanich_integ_2010['Date'][ind1], saanich_integ_2010.loc[:,'rhoC (mmol/m2/d)'][ind1], 'k--.', markersize=14)
f_ax9_a.plot(saanich_integ_2011['Date'][ind2], saanich_integ_2011.loc[:,'rhoC (mmol/m2/d)'][ind2], 'k--.', markersize=14)
# f_ax9_a.plot(saanich_integ_2012_2013['Date'][ind3], saanich_integ_2012_2013.loc[:,'rhoC (mmol/m2/d)'][ind3], 'k--.', markersize=14)
f_ax9_a.plot(saanich_integ_2014_2015['Date'][ind4], saanich_integ_2014_2015.loc[:,'rhoC (mmol/m2/d)'][ind4], 'k--.', markersize=14)
f_ax9_a.plot(saanich_integ_2016_2017['Date'][ind5], saanich_integ_2016_2017.loc[:,'rhoC (mmol/m2/d)'][ind5], 'k--.', markersize=14)

f_ax9_b.plot(saanich_integ_2010['Date'][ind1], saanich_integ_2010.loc[:,'rhoC (mmol/m2/d)'][ind1], 'k--.', markersize=14)
f_ax9_b.plot(saanich_integ_2011['Date'][ind2], saanich_integ_2011.loc[:,'rhoC (mmol/m2/d)'][ind2], 'k--.', markersize=14)
# f_ax9_b.plot(saanich_integ_2012_2013['Date'][ind3], saanich_integ_2012_2013.loc[:,'rhoC (mmol/m2/d)'][ind3], 'k--.', markersize=14)
f_ax9_b.plot(saanich_integ_2014_2015['Date'][ind4], saanich_integ_2014_2015.loc[:,'rhoC (mmol/m2/d)'][ind4], 'k--.', markersize=14)
f_ax9_b.plot(saanich_integ_2016_2017['Date'][ind5], saanich_integ_2016_2017.loc[:,'rhoC (mmol/m2/d)'][ind5], 'k--.', markersize=14)

d = .25  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=2, clip_on=False)
f_ax9_b.plot([0, 1], [0, 0], transform=f_ax9_b.transAxes, **kwargs)
f_ax9_a.plot([0, 1], [1, 1], transform=f_ax9_a.transAxes, **kwargs)

# f_ax9.set_ylabel('(mmol m$^{-2}$ d$^{-1}$)')
f_ax9_a.set_ylabel('       (mmol m$^{-2}$ d$^{-1}$) \n')
# f_ax9.set_xlim(np.min(saanich_integ['Date']),np.max(saanich_integ['Date']))
f_ax9_a.set_xlim(datemin, datemax)
f_ax9_b.set_xlim(datemin, datemax)
# f_ax9.autoscale(enable=True, axis='y', tight=True)
# f_ax9.ticklabel_format(axis='y', scilimits=(0,3), useMathText=True) # set tick labels to scientific notation
# t = f_ax9.yaxis.get_offset_text()
# t.set_x(-0.03) # offset scientific notation above ticks
# f_ax9.set_ylim(0, 16000)
# f_ax9.yaxis.set_ticks(np.arange(0, 16000, 5000))
f_ax9_a.tick_params(axis='y', which='major', length=10)
f_ax9_a.tick_params(axis='y', which='minor', length=6)

f_ax9_b.tick_params(axis='y', which='major', length=10)
f_ax9_b.tick_params(axis='y', which='minor', length=6)
# labels = f_ax9.get_yticks().tolist()
# labels[0] = ''
# f_ax9.set_yticklabels(labels)

# highlight growing seasons:
f_ax9_a.axvspan(datetime.date(2010,3,1), datetime.date(2010,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_a.axvspan(datetime.date(2011,3,1), datetime.date(2011,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_a.axvspan(datetime.date(2012,3,1), datetime.date(2012,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_a.axvspan(datetime.date(2013,3,1), datetime.date(2013,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_a.axvspan(datetime.date(2014,3,1), datetime.date(2014,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_a.axvspan(datetime.date(2015,3,1), datetime.date(2015,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_a.axvspan(datetime.date(2016,3,1), datetime.date(2016,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_a.axvspan(datetime.date(2017,3,1), datetime.date(2017,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)

f_ax9_b.axvspan(datetime.date(2010,3,1), datetime.date(2010,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_b.axvspan(datetime.date(2011,3,1), datetime.date(2011,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_b.axvspan(datetime.date(2012,3,1), datetime.date(2012,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_b.axvspan(datetime.date(2013,3,1), datetime.date(2013,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_b.axvspan(datetime.date(2014,3,1), datetime.date(2014,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_b.axvspan(datetime.date(2015,3,1), datetime.date(2015,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_b.axvspan(datetime.date(2016,3,1), datetime.date(2016,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax9_b.axvspan(datetime.date(2017,3,1), datetime.date(2017,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)

f_ax9_a.set_ylim([0,750])
f_ax9_b.set_ylim([1200,1300])
f_ax9_a.yaxis.set_major_locator(MultipleLocator(200))
f_ax9_a.tick_params(axis='y', which='both', right='True', labelleft='on')
f_ax9_b.tick_params(axis='y', which='both', right='True', labelleft='on')

# f_ax10.set_ylim(0,2000)
# rhoN
ind1 = pd.notna(saanich_integ_2010.loc[:,'rhoN (mmol/m2/d)'])
ind2 = pd.notna(saanich_integ_2011.loc[:,'rhoN (mmol/m2/d)'])
# ind3 = pd.notna(saanich_integ_2012_2013.loc[:,29])
ind4 = pd.notna(saanich_integ_2014_2015.loc[:,'rhoN (mmol/m2/d)'])
ind5 = pd.notna(saanich_integ_2016_2017.loc[:,'rhoN (mmol/m2/d)'])
f_ax10.plot(saanich_integ_2010['Date'][ind1], saanich_integ_2010.loc[:,'rhoN (mmol/m2/d)'][ind1], 'k--.', markersize=14)
f_ax10.plot(saanich_integ_2011['Date'][ind2], saanich_integ_2011.loc[:,'rhoN (mmol/m2/d)'][ind2], 'k--.', markersize=14)
# f_ax10.plot(saanich_integ_2012_2013['Date'][ind3], saanich_integ_2012_2013.loc[:,'rhoN (mmol/m2/d)'][ind3], 'k--.', markersize=14)
f_ax10.plot(saanich_integ_2014_2015['Date'][ind4], saanich_integ_2014_2015.loc[:,'rhoN (mmol/m2/d)'][ind4], 'k--.', markersize=14)
f_ax10.plot(saanich_integ_2016_2017['Date'][ind5], saanich_integ_2016_2017.loc[:,'rhoN (mmol/m2/d)'][ind5], 'k--.', markersize=14)
f_ax10.set_ylabel('(mmol m$^{-2}$ d$^{-1}$)')
# f_ax10.set_xlim(np.min(saanich_integ['Date']),np.max(saanich_integ['Date']))
f_ax10.set_xlim(datemin, datemax)
f_ax10.autoscale(enable=True, axis='y', tight=True)
f_ax10.tick_params(axis='y', which='major', length=10)
f_ax10.tick_params(axis='y', which='minor', length=6)
start, end = f_ax10.get_ylim()
f_ax10.yaxis.set_ticks(np.arange(0, 61, 10))

# highlight growing seasons:
f_ax10.axvspan(datetime.date(2010,3,1), datetime.date(2010,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2011,3,1), datetime.date(2011,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2012,3,1), datetime.date(2012,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2013,3,1), datetime.date(2013,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2014,3,1), datetime.date(2014,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2015,3,1), datetime.date(2015,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2016,3,1), datetime.date(2016,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax10.axvspan(datetime.date(2017,3,1), datetime.date(2017,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)

f_ax10.tick_params(axis='y', which='both', right='True', labelleft='on')

# f_ax11.set_ylim(0,2000)

# rhoSi
f_ax11.hlines(0, datetime.date(2010,1,1), datetime.date(2017,12,31), 'r', 'dashed', linewidth=2.5)

ind1 = pd.notna(saanich_integ_2014_2015.loc[:,'bSiO2 Precipitation'])
ind2 = pd.notna(saanich_integ_2016_2017.loc[:,'bSiO2 Precipitation'])
f_ax11.plot(saanich_integ_2014_2015['Date'][ind1], saanich_integ_2014_2015['bSiO2 Precipitation'][ind1], 'k--.', markersize=14)
f_ax11.plot(saanich_integ_2016_2017['Date'][ind2], saanich_integ_2016_2017['bSiO2 Precipitation'][ind2], 'k--.', markersize=14)
f_ax11.set_ylabel('(mmol m$^{-2}$ d$^{-1}$)')
# f_ax11.set_xlim(np.min(saanich_integ['Date']),np.max(saanich_integ['Date']))
f_ax11.set_xlim(datemin, datemax)
f_ax11.autoscale(enable=True, axis='y', tight=True)
f_ax11.tick_params(axis='y', which='major', length=10)
f_ax11.tick_params(axis='y', which='minor', length=6)
start, end = f_ax11.get_ylim()
f_ax11.yaxis.set_ticks(np.arange(0, end, 50))
start, end = f_ax11.get_ylim()
f_ax11.yaxis.set_ticks(np.append(np.arange(start,0,start),np.arange(0,81,20)))
f_ax11.set_ylim(-20,80)

# highlight growing seasons:
color_code = 'dimgray'
# f_ax11.axvspan(datetime.date(2010,3,1), datetime.date(2010,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
# f_ax11.axvspan(datetime.date(2011,3,1), datetime.date(2011,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
# f_ax11.axvspan(datetime.date(2012,3,1), datetime.date(2012,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2013,3,1), datetime.date(2013,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2014,3,1), datetime.date(2014,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2015,3,1), datetime.date(2015,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2016,3,1), datetime.date(2016,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)
f_ax11.axvspan(datetime.date(2017,3,1), datetime.date(2017,10,30), facecolor=color_code, edgecolor='k', alpha=0.2)

# add indication of different methods used:
f_ax11.axvspan(datetime.date(2013,2,1), datetime.date(2015,12,15), facecolor='blue', edgecolor='k', alpha=0.1, zorder=0, label=r'$^{32}$Si derived')
f_ax11.axvspan(datetime.date(2016,9,1), datetime.date(2017,10,15), facecolor='red', edgecolor='k', alpha=0.1, zorder=0, label=r'Net bSiO$_{2}$ derived')
# f_ax11.legend(loc='upper left', fancybox=True, shadow=True, frameon=True)
f_ax11.tick_params(axis='y', which='both', right='True', labelleft='on')
#------------------------------------------------------------------------------
#### Plot seasonally averaged data:
total_depth = np.max(production_binned['Depth'])+1

# rhoC
h16 = f_ax12.contourf(np.unique(production_binned['Month']),np.arange(0,total_depth,1),rhoC_seas_wide_interp, 100, cmap=cm.jet)

ind_grid = pd.notna(production_binned['rhoC (umol/L)'])
s1 = f_ax12.scatter(production_binned['Month'][ind_grid], production_binned['Depth'][ind_grid], 100, c=production_binned['rhoC (umol/L)'][ind_grid], edgecolor='w', cmap=cm.jet, clip_on=False)

contours1 = f_ax12.contour(np.unique(production_binned['Month']),np.arange(0,total_depth,1),rhoC_seas_wide_interp, contour_label, colors='black')
f_ax12.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

f_ax12.set_ylabel('Depth (m)')
# f_ax12.set_xlabel('Month')
f_ax12.set_xlim([1,12])
cbar = fig.colorbar(h16, ax=f_ax12, format="%1.1f", pad=-0.02, aspect=30)
cbar.set_label(r'($\mathrm{\mu}$mol L$^{-1}$ d$^{-1}$)')
f_ax12.invert_yaxis()
f_ax12.set_title(r' $\mathit{\rho}$C', loc='left', fontweight='bold')

f_ax12.set_xticks(np.arange(1,13,1))
f_ax12.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)

labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
labels[-1] = '>'+str("{:.1f}".format(rhoC_min_outlier))
cbar.ax.set_yticklabels(labels)

f_ax12.yaxis.set_minor_locator(MultipleLocator(1))
f_ax12.tick_params(axis='both', which='major', length=10)
f_ax12.tick_params(axis='y', which='minor', length=6)
f_ax12.tick_params(axis='y', which='both', right='True', labelleft='on')

# rhoN
h17 = f_ax13.contourf(np.unique(production_binned['Month']),np.arange(0,total_depth,1),rhoN_seas_wide_interp, 100, cmap=cm.jet)

ind_grid = pd.notna(production_binned['rhoN (umol/L)'])
s1 = f_ax13.scatter(production_binned['Month'][ind_grid], production_binned['Depth'][ind_grid], 100, c=production_binned['rhoN (umol/L)'][ind_grid], edgecolor='w', cmap=cm.jet, clip_on=False)

contours1 = f_ax13.contour(np.unique(production_binned['Month']),np.arange(0,total_depth,1),rhoN_seas_wide_interp, contour_label, colors='black')
f_ax13.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

f_ax13.set_ylabel('Depth (m)')
f_ax13.set_xlabel('Month')
f_ax13.set_xlim([1,12])
cbar = fig.colorbar(h17, ax=f_ax13, pad=-0.02, aspect=30)
cbar.set_label(r'($\mathrm{\mu}$mol L$^{-1}$ d$^{-1}$)')
f_ax13.invert_yaxis()
f_ax13.set_title(r' $\mathit{\rho}$N', loc='left', fontweight='bold')

f_ax13.set_xticks(np.arange(1,13,1))
f_ax13.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)

labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
labels[-1] = '>'+str("{:.2f}".format(rhoN_min_outlier))
cbar.ax.set_yticklabels(labels)

f_ax13.yaxis.set_minor_locator(MultipleLocator(1))
f_ax13.tick_params(axis='both', which='major', length=10)
f_ax13.tick_params(axis='y', which='minor', length=6)
f_ax13.tick_params(axis='y', which='both', right='True', labelleft='on')

# # rhoSi
# contour_label = 5
# h1 = f_ax14.contourf(np.unique(production_binned['Month']),np.arange(0,total_depth,1),rhoSi_seas_wide_interp, 100, cmap=cm.jet)

# ind_grid = pd.notna(production_binned['rhoSi (umol/L)'])
# s1 = f_ax14.scatter(production_binned['Month'][ind_grid], production_binned['Depth'][ind_grid], 50, c=production_binned['rhoSi (umol/L)'][ind_grid], edgecolor='k', cmap=cm.jet, clip_on=False)

# contours1 = f_ax14.contour(np.unique(production_binned['Month']),np.arange(0,total_depth,1),rhoSi_seas_wide_interp, contour_label, colors='black')
# f_ax14.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)

# f_ax14.set_ylabel('Depth (m)')
# # f_ax14.set_xlabel('Month')
# f_ax14.set_xlim([1,12])
# cbar = fig.colorbar(h1, ax=f_ax14, format="%1.1f")
# cbar.set_label(r' $\mathrm{\rho}$Si ($\mathrm{\mu}$mol L$^{-1}$ d$^{-1}$)')
# f_ax14.invert_yaxis()
# f_ax14.set_title(r' $\mathbf{\rho}$Si', loc='left', fontweight='bold')

# f_ax14.set_xticks(np.arange(1,13,1))
# f_ax14.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=90)

# labels = [cbar.get_text() for cbar in cbar.ax.get_yticklabels()]
# labels[-1] = '>'+str("{:.1f}".format(rhoSi_min_outlier))
# cbar.ax.set_yticklabels(labels)

# f_ax14.yaxis.set_minor_locator(MultipleLocator(1))
# f_ax14.tick_params(axis='both', which='major', length=10)
# f_ax14.tick_params(axis='y', which='minor', length=6)
#------------------------------------------------------------------------------

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### Collate figure handles for a vector graphic format:
h_prod2 = [h1,h2,h4,h5,h6,h7,h9,h10,h14,h15,h16,h17] # collate plot handles
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% Map/Timeline Plot
#### Set-up figure layout:
fig = plt.figure(figsize=(34,24), constrained_layout=True)
font={'family':'DejaVu Sans',
      'weight':'normal',
      'size':'22'} 
plt.rc('font', **font) # sets the specified font formatting globally

gs = fig.add_gridspec(11, 12)
# main plots
f_ax12 = fig.add_subplot(gs[0:7, 0:11])
f_ax14 = fig.add_subplot(gs[7:9, 0:12])
for axis in ['bottom']:
    f_ax14.spines[axis].set_visible(False)
f_ax13 = fig.add_subplot(gs[9:11, 0:12])
for axis in ['top']:
    f_ax13.spines[axis].set_visible(False)
#------------------------------------------------------------------------------
#### Inlet Map

# build colormap:
# c_map = cmo.topo
top = cm.get_cmap('cmo.gray', 128)
bottom = cm.get_cmap('cmo.deep_r', 128)
newcolors = np.vstack((bottom(np.linspace(0, 1, 128)),top(np.linspace(0, 1, 128))))
newcmap = ListedColormap(newcolors, name='gray_topo')

# specify contour depths:
map_contour_label = np.append(np.arange(-2000,-100,200),np.arange(-150,0,50))
map_contour_label_zoomed = np.arange(-200,0,50)

# load data:
from scipy.io import loadmat
mapdata = loadmat('BCRegion_sm.mat')
lat = pd.DataFrame(mapdata['lat'])
lon = pd.DataFrame(mapdata['lon']).T
z = pd.DataFrame(mapdata['z'])
# set coordinates of saanich inlet:
shortlat = lat.where((lat>48.5) & (lat<48.8)).dropna()
shortlon = lon.where((lon>-123.6) & (lon<-123.3)).dropna()
shortz = z.iloc[shortlat.index,shortlon.index]

# plot lower vancouver island bathymetry: 
h1 = f_ax12.contourf(np.tile(lon.T-0.03,(989,1)),np.tile(lat,(1,1321)),z, 100, cmap=newcmap)
h1.set_clim(-2000,2000)
contours1 = f_ax12.contour(np.tile(lon.T-0.03,(989,1)),np.tile(lat,(1,1321)),z, map_contour_label, colors='black')
f_ax12.clabel(contours1, inline=True, inline_spacing=0.01, fmt='%1.1f', rightside_up=True, fontsize=16)
# cbar5 = fig.colorbar(h1, ax=f_ax12)
# cbar5.set_label(r'Elevation (m)')
f_ax12.set_xlabel('Longitude ($\mathrm{^o}$W)')
# f_ax12.tick_params(labelrotation=45)
f_ax12.set_ylabel('Latitude ($\mathrm{^o}$N)')

# create inset axes & plot saanich inlet map:
# axins = f_ax12.inset_axes([0.65, 0.5, 0.25, 0.47])
axins = f_ax12.inset_axes([1.05, 0.4, 0.3, 0.6])
# axins = f_ax12.inset_axes(width=0.25, height=0.47, bbox_to_anchor=([1.65, 0.5]))

norm = cm.colors.DivergingNorm(vmin=-600, vcenter=0, vmax=2000) # scales land to match the main map, accentuate depth colors, and diverge at 0

h2 = axins.contourf(np.tile(shortlon.T-0.03,(133,1)),np.tile(shortlat,(1,86)),shortz, 200, cmap=newcmap, norm=norm)
contours1 = axins.contour(np.tile(shortlon.T-0.03,(133,1)),np.tile(shortlat,(1,86)),shortz, map_contour_label_zoomed, colors='black')
manual_labels = [(-123.504,48.616),(-123.51,48.65),(-123.5,48.68),(-123.525,48.68),(-123.45,48.7)]
axins.clabel(contours1, inline=True, inline_spacing=0.01, manual=manual_labels, fmt='%1.1f', rightside_up=True, fontsize=14)
for axis in ['top','bottom','left','right']:
    axins.spines[axis].set_color('r')
    axins.spines[axis].set_linewidth(4)
mark_inset(f_ax12, axins, loc1=2, loc2=4, lw=4, edgecolor='r', zorder=4)
axins.plot(-123.505,48.59166667, 'r*', markersize=30, markeredgecolor='k')
axins.xaxis.set_major_locator(MultipleLocator(0.25))
axins.yaxis.set_major_locator(MultipleLocator(0.25))
axins.set_xticklabels('')
axins.set_yticklabels('')
axins.grid()
axins.text(-123.52,48.785,'Saanich Inlet', backgroundcolor='w', bbox=dict(facecolor='w',edgecolor='r'))
# f_ax12.indicate_inset_zoom(axins)

# finishing touches:
f_ax12.text(-124.5,49.955,' Lower Vancouver Island ', backgroundcolor='w', bbox=dict(facecolor='w',edgecolor='k'))
f_ax12.plot(-123.505,48.59166667, 'r*', markersize=24, markeredgecolor='k')
f_ax12.tick_params(axis='both', which='major', length=10)
f_ax12.tick_params(axis='x', which='minor', length=6)
f_ax12.xaxis.set_minor_locator(MultipleLocator(0.25))
f_ax12.grid(which='both', alpha=0.5)
f_ax12.set_xticklabels(['127','126','125','124','123','122'])
#------------------------------------------------------------------------------
#### Timeline
# import timeline data:
timeline_data = pd.read_excel('C:/Users/bcamc/OneDrive/Documents/Education/UVic/Honours/Data/Saanich Data/Saanich_Timeline.xlsx', sheet_name='Timeline')
timeline_data = timeline_data.set_index('Date')
timeline_data = timeline_data.T

# create discrete colormap:
cmap = cm.colors.ListedColormap(['black', 'silver']) # 0=black (unpublished), 1=grey (published)

# plot timeline data:
f_ax13.pcolormesh(timeline_data.columns, np.arange(0,13,1), timeline_data, cmap=cmap, edgecolor='w', linewidths=4, joinstyle='bevel')

# format dates:
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

f_ax13.xaxis.set_major_locator(years)
f_ax13.xaxis.set_major_formatter(years_fmt)
f_ax13.xaxis.set_minor_locator(months)

datemin = np.datetime64(timeline_data.columns[0], 'Y')
datemax = np.datetime64(timeline_data.columns[-1], 'Y') + np.timedelta64(1, 'Y')
f_ax13.set_xlim(datemin, datemax)

f_ax13.yaxis.set_major_locator(MultipleLocator(1))
f_ax13.tick_params(axis='both', which='major', length=10)
f_ax13.tick_params(axis='both', which='minor', length=6)

# set category labels:
f_ax13.set_yticks(np.arange(0.5,12.5,step=1))
f_ax13.set_yticklabels(['CTD',
                        'NO$_{\mathrm{3}}$+NO$_{\mathrm{2}}$','NO$_{\mathrm{2}}$','PO$_{\mathrm{4}}$','Si(OH)$_{\mathrm{4}}$',
                        'Chl $\mathit{a}$','POC','PON','bSiO$_{\mathrm{2}}$',
                        r'$\mathrm{\rho}$C',r'$\mathrm{\rho}$N',r'$\mathrm{\rho}$Si'])
# f_ax13.tick_params(axis='y', which='both', length=0)
f_ax13.set_xlabel('Time (Years)')

#------------------------------------------------------------------------------
# plot sampling depths:
f_ax14.plot(x_condNO3,y_condNO3,'k.', markersize='16')
f_ax14.invert_yaxis()
f_ax14.set_ylabel('Depth (m)')

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

f_ax14.xaxis.set_major_locator(years)
f_ax14.xaxis.set_major_formatter(years_fmt)
f_ax14.xaxis.set_minor_locator(months)

datemin = np.datetime64(timeline_data.columns[0], 'Y')
datemax = np.datetime64(timeline_data.columns[-1], 'Y') + np.timedelta64(1, 'Y')
f_ax14.set_xlim(datemin, datemax)

f_ax14.yaxis.set_major_locator(MultipleLocator(5))
f_ax14.yaxis.set_minor_locator(MultipleLocator(1))
f_ax14.tick_params(axis='both', which='major', length=10)
f_ax14.tick_params(axis='both', which='minor', length=6)

plt.setp(f_ax14.get_xticklabels(), visible=False)
f_ax14.tick_params(axis='x', which='both', length=0)

f_ax14.tick_params(axis='y', which='both', right='True', labelleft='on')
#------------------------------------------------------------------------------
# add legend for timeline:
legend_elements = [Line2D([0], [0], marker='s', color='black', ms=21, linestyle='', label='Unpublished data'),
                Line2D([0], [0], marker='s', color='silver', ms=21, linestyle='', label='Previously published data')]
f_ax14.legend(handles=legend_elements, loc='lower left', fancybox=True, shadow=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### Collate figure handles for a vector graphic format:
h_map = [h1,h2] # collate plot handles
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% Save figures as vector images for publication
# these commands loop through the figure handles and smooth the edges of each to 
# remove the contourf "white lines" issue caused in the aliasing process when 
# the renderer loads each image in the vector image:
#------------------------------------------------------------------------------
# Nuts
for i,idx in enumerate(h_nuts):
    for image in h_nuts[i].collections:
        image.set_edgecolor("face")

fig.savefig("C:/Users/bcamc/OneDrive/Documents/Education/UVic/Honours/Data/Saanich Data/Saanich_Figs_Vector_Images/Nutrients.pdf")
#------------------------------------------------------------------------------
# Particulates
for i,idx in enumerate(h_part):
    for image in h_part[i].collections:
        image.set_edgecolor("face")

fig.savefig("C:/Users/bcamc/OneDrive/Documents/Education/UVic/Honours/Data/Saanich Data/Saanich_Figs_Vector_Images/Particulates.pdf")
#------------------------------------------------------------------------------
# Production (#1)
for i,idx in enumerate(h_prod1):
    for image in h_prod1[i].collections:
        image.set_edgecolor("face")

fig.savefig("C:/Users/bcamc/OneDrive/Documents/Education/UVic/Honours/Data/Saanich Data/Saanich_Figs_Vector_Images/Production.pdf")
#------------------------------------------------------------------------------
# Production (#2)
for i,idx in enumerate(h_prod2):
    for image in h_prod2[i].collections:
        image.set_edgecolor("face")

fig.savefig("C:/Users/bcamc/OneDrive/Documents/Education/UVic/Honours/Data/Saanich Data/Saanich_Figs_Vector_Images/Production_alt.pdf")
#------------------------------------------------------------------------------
# Map/timeline
for i,idx in enumerate(h_map):
    for image in h_map[i].collections:
        image.set_edgecolor("face")

fig.savefig("C:/Users/bcamc/OneDrive/Documents/Education/UVic/Honours/Data/Saanich Data/Saanich_Figs_Vector_Images/Map_Timeline.pdf")
#------------------------------------------------------------------------------

