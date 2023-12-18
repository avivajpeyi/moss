# -*- coding: utf-8 -*-
"""
Preprocess and Detrend Raw Wind Data (wind_raw.csv) to obtain wind_detrend.csv

@author: Zhixiong Hu, UCSC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mydata = pd.read_csv('data/wind_raw.csv', header = 0, index_col=0)

# datetime
mydata.valid = pd.to_datetime(mydata.valid)
mydata.valid.dt.date  # .dt is useful

# select every two hours
mydata['date'] = mydata.valid.dt.date 
mydata['date2hour'] = mydata.valid.dt.floor('2H')

# take median
median_sknt = mydata.groupby(['station', 'lon', 'lat', 'date', 'date2hour'])['sknt'].median()
median_sknt_df = median_sknt.reset_index()

# detrend by 1-st order differencing
# create sknt_diff
df = median_sknt_df
df['sknt_diff'] = np.NaN
for s in np.unique(median_sknt_df.station):
    station_sknt = df.loc[df.station == s, 'sknt']
    sknt_diff = station_sknt.values[1:] - station_sknt.values[:-1]
    df.loc[df.station == s, 'sknt_diff'] = np.concatenate([np.zeros([1, ]), sknt_diff], axis=0)

mydata = df
CA_wind_diff = mydata.loc[:, ['date2hour', 'station', 'sknt_diff']]
# change long to wide
CA_wide = CA_wind_diff.pivot(index='date2hour', columns='station', values='sknt_diff')
# check NA rows for each station
s_over = []
for s in CA_wide.columns:
    nan_v = CA_wide.loc[CA_wide[s].isna()].shape[0]
    print(s, nan_v)
    if nan_v > 5:
        s_over.append(s)
len(s_over)
# look at na rows
CA_wide.loc[CA_wide.isna().any(axis=1)]
# impute nan into 0 for columns (station)
CA_wide = CA_wide.fillna(0)
# standardize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
de_stdize = scaler.fit_transform(CA_wide.values)
CA_wide_detrend = pd.DataFrame(de_stdize, index=CA_wide.index, columns=CA_wide.columns)
# save data for spectral analysis.
CA_wide_detrend.to_csv('data/wind_detrend_.csv')

