"""
INTENSE QC Component 2 - Quality Control

This component of the INTENSE QC package reads rainfall data formatted as an INTENSE
Series and executes the flagging process by which the data is checked. 
NO DATA IS EXCLUDED IN THIS CODE!
To exclude flagged data use component 3: Rulebase. 

This QC code will use the INTENSE module to create QC object

Data is read in by the INTENSE module as 
    data = pd.Series(data, 
                     pd.date_range(start=pd.datetime.strptime(metadata['start datetime'],'%Y%m%d%H'),
                     end=pd.datetime.strptime(metadata['end datetime'],'%Y%m%d%H'),
                     freq=metadata['new timestep'][:-2]+'H'),
                     dtype=float)
    
    
The INTENSE object looks like this:
    s =  Series(station_id=metadata['station id'],
                path_to_original_data=metadata['path to original data'],
                latitude=tryFloat(metadata['latitude']),
                longitude=tryFloat(metadata['longitude']),
                original_timestep=metadata['original timestep'],
                original_units=metadata['original units'],
                new_units=metadata['new units'],
                new_timestep=metadata['new timestep'],
                elevation=metadata['elevation'],
                data=data)
    
For more details on INTENSE objects and the associated functionality refer to 
Component 1: intense_CW.py

Required packages: 
    intense
    pandas
    numpy
    rpy2
    xarray
    scipy
    datetime
    zipfile
    subprocess
    os
    
Developed by: 
    Elizabeth Lewis, PhD
    SB, RV, others...

Publication to be cited:
    Paper

June 2019 
"""

##import intense.intense as ex
import intense as ex
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import xarray as xr
import numpy.ma as ma
import scipy.interpolate
import datetime
import os
import zipfile
import math
import scipy.spatial as sp
import subprocess
import scipy.stats
from multiprocessing import Pool, Manager
import calendar
import time

trend = importr('trend')

"""
+++++++++++++++++++++++++++++++++++++++++++ Basic checks +++++++++++++++++++++++++++++++++++++++++++
"""


# Indicative check to flag years with 95th or 99th percentiles equal to zero.
def check_percentiles(its):
    perc95 = its.data.groupby(pd.Grouper(freq='A')).quantile(.95)
    perc99 = its.data.groupby(pd.Grouper(freq='A')).quantile(.99)

    return [[d.year for d in list(perc95[perc95 == 0].index)], [d.year for d in list(perc99[perc99 == 0].index)]]


# Indicative check to flag years with K-largest values equal to zero.
def check_k_largest(its):
    k1 = its.data.groupby(pd.Grouper(freq='A')).nlargest(n=1).min(level=0)
    k5 = its.data.groupby(pd.Grouper(freq='A')).nlargest(n=5).min(level=0)
    k10 = its.data.groupby(pd.Grouper(freq='A')).nlargest(n=10).min(level=0)

    return [[d.year for d in list(k1[k1 == 0].index)], [d.year for d in list(k5[k5 == 0].index)],
            [d.year for d in list(k10[k10 == 0].index)]]


# Indicative, checks if proportions of rainfall in each day is significantly different
def check_days_of_week(its):
    # 0 is monday, 1 is tuesday etc...
    days = its.data.groupby(lambda x: x.weekday).mean()
    popmean = its.data.mean()
    p = scipy.stats.ttest_1samp(days, popmean)[1]
    if p < 0.01:  # different
        return 1
    else:
        return 0


# Indicative, hourly analogue to daily check
def check_hours_of_day(its):
    # 0 is midnight, 1 is 01:00 etc...
    hours = its.data.groupby(lambda x: x.hour).mean()
    popmean = its.data.mean()
    p = scipy.stats.ttest_1samp(hours, popmean)[1]
    if p < 0.01:  # different
        return 1
    else:
        return 0


# Annual check for discontinuous records.
# Returns years where more than 5 no data periods are bounded by zeros.
# A no data period is defined as 2 or more consecutive missing values.
# Return years where more than 5 no data periods are bounded by zeros
# A no data period is defined as 2 or more consecutive missing values
# For a year to be flagged no data periods must occur in at least 5 different days

def check_intermittency(its):
    # Shift data +/- 1 hour to help identify missing data periods with vectorised approach
    df = its.data.copy().to_frame()
    df.columns = ['val']
    df['prev'] = df.shift(1)['val']
    df['next'] = df.shift(-1)['val']

    # Look for >=2 consecutive missing values (bounds by >=0 values first)
    # - find start and end indices of these periods
    start_inds = np.flatnonzero((np.isfinite(df.prev)) & (np.isnan(df.val)) &
                                (np.isnan(df.next)))
    end_inds = np.flatnonzero((np.isnan(df.prev)) & (np.isnan(df.val)) &
                              (np.isfinite(df.next)))

    # If no final non-nan value then if a missing period assign end index as 
    # end of series
    if len(start_inds) == len(end_inds):
        pass
    elif len(start_inds) == len(end_inds) + 1:
        end_inds = end_inds.tolist()
        end_inds.append(len(df['val']) - 1)
        end_inds = np.asarray(end_inds, dtype=np.int)
    else:
        print('intermittency period identification error')

    # Select only the periods preceded and followed by dry hours
    start_inds2 = []
    end_inds2 = []
    if len(start_inds) > 0:
        for si, ei in zip(start_inds, end_inds):

            # Account for case that first start index is beginning of series and
            # case that final end index is end of series
            if (si == 0) or (ei == (df['val'].shape[0] - 1)):
                start_inds2.append(si)
                end_inds2.append(ei)

            # Otherwise check if preceding/following values are dry
            else:
                if (df['prev'][si] == 0) and (df['next'][ei] == 0):
                    start_inds2.append(si)
                    end_inds2.append(ei)

    start_inds = start_inds2
    end_inds = end_inds2

    # Count missing periods by year
    # - just count year in which the missing period begins, i.e. ignore if finishes
    # in e.g. the next year for now
    if len(start_inds) > 0:
        '''
        dc = {}
        dates = []
        for si,ei in zip(start_inds, end_inds):
            start_year = df.index.year[si]
            end_year = df.index.year[ei]
            d = df.index.date[si]
            if d not in dates:
                if start_year not in dc.keys():
                    dc[start_year] = 1
                else:
                    dc[start_year] += 1
                if start_year != end_year:
                    if end_year not in dc.keys():
                        dc[end_year] = 1
                    else:
                        dc[end_year] += 1
                dates.append(d)

        # Make final list of years with >=5 missing periods
        flag_years = []
        for year,count in dc.items():
            if count >= 5:
                flag_years.append(year)
        flag_years = sorted(flag_years)
        '''

        # Alternative faster approach using dataframe operations
        df1 = df.iloc[start_inds].copy()
        df1['date'] = df1.index.date
        df1['year'] = df1.index.year
        df1.drop_duplicates('date', inplace=True)
        df2 = df1.groupby(df1['year'])['year'].agg('count')
        df2 = df2.to_frame()
        df2.columns = ['count']
        df2 = df2.loc[df2['count'] >= 5]
        flag_years = df2.index.tolist()

    else:
        flag_years = []

    return flag_years


# Indicative, Pettitt breakpoint check
def check_break_point(its):
    x = its.data.resample("D").sum().values
    x = x[~np.isnan(x)]
    x = x
    x = robjects.FloatVector(x)

    # using the pettitt test
    pettitt = robjects.r['pettitt.test']
    pet = pettitt(x)
    y = pet.rx('p.value')  # gives the p-value if p-value is below 0.05 (or 0.01) there might be a change point
    p = np.asarray(y)[0][0]
    if p < 0.01:  # different
        return 1
    else:
        return 0


"""
++++++++++++++++++++++++++++++++++ Threshold Checks +++++++++++++++++++++++++++++++++++
"""


def world_record_check(val):
    """
    1h record1 with separate flags showing exceedance by <20%, >= 20%, 33%, or 50% - 
    world record = 394.5mm in 1 hour (http://www.nws.noaa.gov/oh/hdsc/record_precip/record_precip_world.html)
    
    *** updated to use 401.0 mm in 1 hour - compare e.g.
    http://www.nws.noaa.gov/oh/hdsc/record_precip/record_precip_world.html
    http://www.bom.gov.au/water/designRainfalls/rainfallEvents/worldRecRainfall.shtml
    https://wmo.asu.edu/content/world-meteorological-organization-global-weather-climate-extremes-archive
    """

    if val >= world_records['hourly'] * 1.5:
        return 4
    elif val >= world_records['hourly'] * 1.33:
        return 3
    elif val >= world_records['hourly'] * 1.2:
        return 2
    elif val >= world_records['hourly']:
        return 1
    else:
        return 0


def world_record_check_ts(its):
    wrcts = its.data.map(lambda x: world_record_check(x))
    return list(wrcts)

# Checks against ETCCDI indices ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# We are using [ETCCDI indicies](http://etccdi.pacificclimate.org/list_27_indices.shtml)
# to act as thresholds for expected hourly values.
# In particular, we are using index 17:
# Rx1day, Monthly maximum 1-day precipitation:
# Here I'm just going to start by using the maximum of the annual maximums,
# to give us the biggest possible daily value for each available square.
# First we must read in the indicies from the netCDF file:
# We then calculate the maximum rainfall value over the whole period for each gridsquare.


# ETCCDI utility function 1
def prep_etccdi_variable(input_path, index_name, aggregation, data_source):
    ds = xr.open_dataset(input_path)

    # Omit final year (2010) of HADEX2 - suspiciously large CDD for Malaysia
    if data_source == 'HADEX2':
        ds = ds.sel(time=slice(datetime.datetime(1951, 1, 1, 0),
                               datetime.datetime(2009, 12, 31, 23)))

    # Calculate maximum rainfall value over whole period
    vals = ds[index_name].values
    if index_name in ['CWD', 'CDD']:
        vals = ds[index_name].values.astype('timedelta64[s]')
        vals = vals.astype('float32') / 86400.0
        vals[vals < 0.0] = np.nan
    vals = ma.masked_invalid(vals)
    if aggregation == 'max':
        data = ma.max(vals, axis=0)
    if aggregation == 'mean':
        data = ma.mean(vals, axis=0)

    # Convert back from to a xarray DataArray for easy plotting
    # - masked array seems to be interpreted as np array (i.e. nans are present 
    # in the xarray DataArray
    data2 = xr.DataArray(data, coords={'Latitude': ds['lat'].values,
                                       'Longitude': ds['lon'].values}, dims=('Latitude', 'Longitude'),
                         name=index_name)

    ds.close()

    return data2


def get_etccdi_value(index_name, lon, lat):
    lon = float(lon)
    lat = float(lat)

    # Check gauge longitude and convert to -180 - 180 range if necessary
    if lon > 180.0:
        lon = lon - 360.0

    # Array location indices for closest cell centre to gauge location
    location_indices = {'GHCNDEX': {}, 'HADEX2': {}}
    for data_source in location_indices.keys():
        location_indices[data_source]['lon'] = (np.argmin(
            np.abs(etccdi_data[data_source][index_name]['Longitude'].values - lon)))
        location_indices[data_source]['lat'] = (np.argmin(
            np.abs(etccdi_data[data_source][index_name]['Latitude'].values - lat)))

    # Maximum of ETCCDI index values from GHCNDEX and HADEX2 for cell
    etccdi_index_values = {}
    for data_source in location_indices.keys():
        yi = location_indices[data_source]['lat']
        xi = location_indices[data_source]['lon']
        etccdi_index_values[data_source] = etccdi_data[data_source][index_name].values[yi, xi]
    etccdi_vals = np.asarray(list(etccdi_index_values.values()))
    if np.any(np.isfinite(etccdi_vals)):
        max_index = np.max(etccdi_vals[np.isfinite(etccdi_vals)])
    else:
        max_index = np.nan

    # For cases where no value for the cell, look in 3x3 window and take the maximum
    if np.isnan(max_index):
        etccdi_index_window = {}
        for data_source in location_indices.keys():
            yi = location_indices[data_source]['lat']
            xi = location_indices[data_source]['lon']
            window = etccdi_data[data_source][index_name].values[yi - 1:yi + 2, xi - 1:xi + 2]
            if np.any(np.isfinite(window)):
                etccdi_index_window[data_source] = np.max(window[np.isfinite(window)])
            else:
                etccdi_index_window[data_source] = np.nan

        window_vals = np.asarray(list(etccdi_index_window.values()))
        if np.any(np.isfinite(window_vals)):
            max_index_window = np.max(window_vals[np.isfinite(window_vals)])
        else:
            max_index_window = np.nan

    else:
        max_index_window = np.nan

    return max_index, max_index_window

# ETCCDI utility function 3 - returns flag based on exceedence of parameters
# Replaces Rx1dayCheck, R99pTOTCheck, PRCPTOTCheck
def day_check(val, p_max, p_max_filled):
    if np.isnan(p_max):
        if val >= p_max_filled * 1.5:
            return 8
        elif val >= p_max_filled * 1.33:
            return 7
        elif val >= p_max_filled * 1.2:
            return 6
        elif val >= p_max_filled:
            return 5
        else:
            return 0
    else:
        if val >= p_max * 1.5:
            return 4
        elif val >= p_max * 1.33:
            return 3
        elif val >= p_max * 1.2:
            return 2
        elif val >= p_max:
            return 1
        else:
            return 0


# ++++++++ Rx1day: check hourly values against maximum 1-day precipitation   ++++++++


def rx_1_day_check_ts(its):
    p_max, p_max_filled = get_etccdi_value('Rx1day', its.longitude, its.latitude)
    df = its.data.to_frame("GSDR")

    ''' If you have a high density of daily gauges, you can calculate Rx1day stats from that and compare them to a daily
        total from the hourly gauges. The ETCCDI gauge density is not high enough to do this so we use it as a threshold
        check for hourly values
    
    df["roll"] = np.around(df.GSDR.rolling(window=24, center=False, min_periods=24).sum())
    df["r1dcts"] = df.roll.map(lambda x: dayCheck(x, pMax, pMaxFilled))
    '''
    if np.isfinite(p_max) or np.isfinite(p_max_filled):
        df["r1dcts"] = df.GSDR.map(lambda x: day_check(x, p_max, p_max_filled))
    else:
        df["r1dcts"] = np.nan

    return list(df.r1dcts)


# ++++++++ Other precipitation index checks ++++++++

# Indicative check against R99pTOT: R99pTOT. Annual total PRCP when RR > 99p:

def r99ptot_check_annual(its):
    p_max, p_max_filled = get_etccdi_value('R99p', its.longitude, its.latitude)

    if np.isfinite(p_max) or np.isfinite(p_max_filled):

        daily_ts = its.data.resample(
            "D").sum()  # this changes depending on which version of pandas youre using. o.14 requires how agument,
        # later requires .sum

        perc99 = daily_ts.groupby(pd.Grouper(freq='A')).quantile(.99)
        py = list(perc99.index.year)
        pv = list(perc99)
        p_dict = {}
        for p in range(len(py)):
            p_dict[py[p]] = pv[p]
        daily_df = daily_ts.to_frame("daily")
        daily_df["year"] = daily_df.index.year
        daily_df["p99"] = daily_df.apply(lambda row: p_dict[row.year], axis=1)
        daily_df["filtered"] = daily_df.daily.where(daily_df.daily >= daily_df.p99)
        perc99_tot = daily_df.groupby(pd.Grouper(freq='A')).sum()
        tots = list(perc99_tot.filtered)
        checks = [day_check(t, p_max, p_max_filled) for t in tots]

    else:
        checks = [np.nan]

    return checks


# Indicative check against annual total: PRCPTOT. Annual total precipitation in wet days:

def prcptot_check_annual(its):
    # pMax, pMaxFilled = getPRCPTOT(its.latitude, its.longitude)
    p_max, p_max_filled = get_etccdi_value('PRCPTOT', its.longitude, its.latitude)

    if np.isfinite(p_max) or np.isfinite(p_max_filled):
        ann_tots = its.data.groupby(pd.Grouper(freq='A')).sum()
        tots = list(ann_tots)
        checks = [day_check(t, p_max, p_max_filled) for t in tots]
    else:
        checks = [np.nan]

    return checks


# ++++++++ Long wet/dry spell checks ++++++++

# ETCCDI provide an index for maximum length of wet spell.
# We can use this to see if there are a suspicious number of consecutive wet hours recorded.
# Consecutive Wet Days: Maximum length of wet spell, maximum number of consecutive days with RR = 1mm: 
# Let RRij be the daily precipitation amount on day i in period j.
# Count the largest number of consecutive days where: RRij = 1mm

# Helper function, flags data based on various thresholds
def spell_check(val, longest_wet_period, longest_wet_period_filled):
    if np.isnan(longest_wet_period):
        if val >= longest_wet_period_filled * 24 * 1.5:
            return 8
        elif val >= longest_wet_period_filled * 24 * 1.33:
            return 7
        elif val >= longest_wet_period_filled * 24 * 1.2:
            return 6
        elif val >= longest_wet_period_filled * 24:
            return 5
        else:
            return 0
    else:
        if val >= longest_wet_period * 24 * 1.5:
            return 4
        elif val >= longest_wet_period * 24 * 1.33:
            return 3
        elif val >= longest_wet_period * 24 * 1.2:
            return 2
        elif val >= longest_wet_period * 24:
            return 1
        else:
            return 0


# Calculate length of consecutive wet days and their location in the rainfall series
def get_wet_periods(vals):
    daily = vals.groupby(lambda x: x.floor('1D')).aggregate(lambda x: np.sum(x))

    start_day_index_list = []
    start_index_list = []
    duration_list = []

    wet_flag = 0
    day_ticker = 0

    for i in range(len(daily)):
        v = daily.iloc[i]

        if v >= 1.0:
            if wet_flag == 0:
                start_day_index_list.append(daily.index[i])
            day_ticker += 1
            wet_flag = 1
        else:
            if wet_flag == 1:
                duration_list.append(day_ticker)
            day_ticker = 0
            wet_flag = 0

        if i == len(daily) - 1:
            if wet_flag == 1:
                duration_list.append(day_ticker)

    # Convert date list to index list
    for i in range(len(start_day_index_list)):
        if (i == 0 & (start_day_index_list[i] < vals.index[i])):
            start_index_list.append(0)
        else:
            start_index_list.append(vals.index.get_loc(start_day_index_list[i]))

    # Convert day length to hourly length:
    duration_list = list(np.dot(24, duration_list))

    return [start_index_list, duration_list]


def cwd_check(its):
    vals = its.data
    longest_wet_period, longest_wet_period_filled = get_etccdi_value('CWD', its.longitude, its.latitude)
    start_index_list, duration_list = get_wet_periods(vals)
    flags_list = [0 for i in range(len(vals))]

    if np.isfinite(longest_wet_period) or np.isfinite(longest_wet_period_filled):

        for wetPeriod in range(len(start_index_list)):
            flag = spell_check(duration_list[wetPeriod], longest_wet_period, longest_wet_period_filled)

            for j in range(start_index_list[wetPeriod],
                           min(start_index_list[wetPeriod] + duration_list[wetPeriod], (len(flags_list) - 1)), 1):
                flags_list[j] = flag

    else:
        flags_list = [np.nan for i in range(len(vals))]

    return flags_list


# ### Long dry spells

# ETCCDI provide an index for maximum length of dry spell.
# We can use this to see if there are a suspicious number of consecutive dry hours recorded.
# Consecutive Dry Days: Maximum length of dry spell, maximum number of consecutive days with RR < 1mm: 
# Let RRij be the daily precipitation amount on day i in period j.
# Count the largest number of consecutive days where: RRij < 1mm

def get_dry_periods(vals):
    start_index_list = []
    duration_list = []

    dry_flag = 0
    hours_ticker = 0

    for i in range(len(vals)):
        v = vals[i]

        if v == 0:
            if dry_flag == 0:
                start_index_list.append(i)

            hours_ticker += 1
            dry_flag = 1
            if i == len(vals) - 1:
                duration_list.append(hours_ticker)

        else:
            if dry_flag == 1:
                duration_list.append(hours_ticker)

            hours_ticker = 0
            dry_flag = 0

        if i == len(vals):
            if dry_flag == 1:
                duration_list.append(hours_ticker)

    return [start_index_list, duration_list]


def cdd_check(its):
    vals = list(its.data)
    longest_dry_period, longest_dry_period_filled = get_etccdi_value('CDD', its.longitude, its.latitude)

    start_index_list, duration_list = get_dry_periods(vals)
    flags_list = [0 for i in range(len(vals))]

    if np.isfinite(longest_dry_period) or np.isfinite(longest_dry_period_filled):

        for dryPeriod in range(len(start_index_list)):
            flag = spell_check(duration_list[dryPeriod], longest_dry_period, longest_dry_period_filled)

            for j in range(start_index_list[dryPeriod], start_index_list[dryPeriod] + duration_list[dryPeriod], 1):
                flags_list[j] = flag
    else:
        flags_list = [np.nan for i in range(len(vals))]

    return flags_list


# ++++++++++++++++++++++++++++++++++ Non-Threshold Checks +++++++++++++++++++++++++++++++++++


def get_sdii(its):
    # *** CHECK HOURLY WORLD RECORD PRECIPITATION ***
    # ?? insert a check for whether gauge SDII exceeds minimum tip / resolution/precision ??

    # Remove any hours exceeding world record in the gauge record
    df1 = its.data.copy().to_frame()
    df1.columns = ['val']
    df1['val'] = np.where(df1['val'] > world_records['hourly'], np.nan, df1['val'])

    # Aggregate gauge to daily and remove any days exceeding world record
    # - remove first and last days assuming might be incomplete
    df2 = df1.resample("D", label='left', closed='right').apply(lambda x: x.values.sum())
    df2 = df2.loc[(df2.index > df2.index.min()) & (df2.index < df2.index.max())]
    df2['val'] = np.where(df2['val'] > world_records['daily'], np.nan, df2['val'])

    # Calculate SDII from gauge
    prcp_sum = df2.loc[df2['val'] >= 1.0, 'val'].sum()
    wetday_count = df2.loc[df2['val'] >= 1.0, 'val'].count()
    sdii_gauge = prcp_sum / float(wetday_count)

    # Retrieve SDII from gridded ETCCDI datasets
    sdii_cell, sdii_filled = get_etccdi_value('SDII', its.longitude, its.latitude)
    if np.isfinite(sdii_cell):
        sdii_gridded = sdii_cell
    else:
        if np.isfinite(sdii_filled):
            sdii_gridded = sdii_filled
        else:
            sdii_gridded = np.nan

    return [sdii_gridded, sdii_gauge]


# ++++++++ Daily accumulation checks ++++++++


def daily_accums_day_check(day_list, mean_wet_day_val, mean_wet_day_val_filled):
    """
    Suspect daily accumulations flagged where a recorded rainfall amount at these times is preceded by 23 hours with no
    rain. A threshold of 2x the mean wet day amount for the corresponding month is applied to increase the chance of
    identifying accumulated values at the expense of genuine, moderate events.
    """

    if day_list[23] > 0:
        dry_hours = 0
        for i in range(23):
            if day_list[i] <= 0:
                dry_hours += 1
        if dry_hours == 23:
            if np.isnan(mean_wet_day_val):
                if day_list[23] > mean_wet_day_val_filled:
                    flag = 2
                else:
                    flag = 0
            else:
                if day_list[23] > mean_wet_day_val:
                    flag = 1
                else:
                    flag = 0
        else:
            flag = 0
    else:
        flag = 0

    return flag


def daily_accums_check(its):
    vals = list(its.data)

    mean_wet_day_val, mean_wet_day_val_filled = get_sdii(its)

    flags = [0 for i in range(len(vals))]

    for i in range(len(vals) - 24):
        day_val_list = vals[i:i + 24]
        flag = daily_accums_day_check(day_val_list, mean_wet_day_val, mean_wet_day_val_filled)
        if flag > max(flags[i:i + 24]):
            flags[i:i + 24] = [flag for j in range(24)]

    return flags


"""
++++++++ Monthly accumulation checks ++++++++
"""


def monthly_accums_day_check(month_list, mean_wet_day_val, mean_wet_day_val_filled):
    """Suspect monthly accumulations.  
    Identified where only one hourly value is reported over a period of a month
    and that value exceeds the mean wet hour amount for the corresponding month."""

    if month_list[719] > 0:
        dry_hours = 0
        for i in range(719):
            if month_list[i] <= 0:
                dry_hours += 1
        if dry_hours == 719:
            if np.isnan(mean_wet_day_val):
                if month_list[719] > mean_wet_day_val_filled * 2:
                    return 2
                else:
                    return 0
            else:
                if month_list[719] > mean_wet_day_val * 2:
                    return 1
                else:
                    return 0
        else:
            return 0
    else:
        return 0


# Flags month prior to high value

def monthly_accums_check(its):
    # Find threshold for wet hour following dry month (2 * mean wet day rainfall)
    mean_wetday_val, mean_wetday_val_filled = get_sdii(its)
    if np.isnan(mean_wetday_val):
        threshold = mean_wetday_val_filled * 2.0
    else:
        threshold = mean_wetday_val * 2.0

    # Lag values forwards and backwards to help identify consecutive value streaks
    df = its.data.copy().to_frame()
    df.columns = ['val']
    df['prev'] = df.shift(1)['val']
    df['next'] = df.shift(-1)['val']

    # Look for streaks of consecutive zeros followed by a wet hour
    start_inds = np.flatnonzero(((df.prev > 0.0) | (np.isnan(df.prev))) & (df.val == 0.0) &
                                (df.next == 0.0))
    end_inds = np.flatnonzero((df.prev == 0.0) & (df.val == 0.0) &
                              ((df.next > 0.0) | (np.isnan(df.next)))) + 1

    # Check whether any periods identified (if not then may be a very high 
    # proportion of missing data
    if start_inds.shape[0] > 0:

        # Check whether final start index has a corresponding end index
        if end_inds.shape[0] == (start_inds.shape[0] - 1):
            end_inds = end_inds.tolist()
            end_inds.append(start_inds[-1])
            end_inds = np.asarray(end_inds)

        # Check whether final end index is out of array bounds (by 1)
        # - this occurs if final period stretches to the end of the dataframe,
        # where 'next' will be a nan
        if int(end_inds[-1]) == len(df['val']):
            end_inds[-1] -= 1

        # Summary dataframe of periods
        df1 = pd.DataFrame(
            dict(start=start_inds, end=end_inds))
        df1['diff'] = df1['end'] - df1['start'] + 1

        # Subset on periods with length of >= 720 days
        df1 = df1.loc[df1['diff'] >= 720]

        # Code below would adjust any periods >720 days to be =720 days (i.e. 
        # alter period start date) - not currently used

        # Filter on periods where end wet hour exceeds threshold (greater than 
        # 2 * mean wet day rainfall)
        df1['end_rainfall'] = np.nan
        i = 0
        for row in df1.iterrows():
            idx = int(row[1].end)
            df1.iloc[i, df1.columns.get_loc('end_rainfall')] = (
                df.iloc[idx, df.columns.get_loc('val')])
            i += 1
        df2 = df1.loc[df1['end_rainfall'] > threshold].copy()

        # Find out if the 23 hours following the wet hour are wet or dry 
        # (0=dry, 1=wet (any rain))
        df2['end_nextday_wet'] = 0
        i = 0
        for row in df2.iterrows():
            idx = int(row[1].end)
            rainfall_sum = df.iloc[idx + 1:idx + 1 + 23, df.columns.get_loc('val')].sum()
            if rainfall_sum > 0.0:
                df2.iloc[i, df2.columns.get_loc('end_nextday_wet')] = 1
            i += 1

        # Define flags
        flag = 1
        if np.isnan(mean_wetday_val):
            flag = 2
        df2['flag'] = flag
        df2['flag'] = np.where(df2['end_nextday_wet'] == 1, df2['flag'] + 2,
                               df2['flag'])

        # Make list of flags
        flags = [0 for i in range(len(df['val']))]
        for row in df2.iterrows():
            for i in range(int(row[1].start), int(row[1].end + 1)):
                flags[i] = int(row[1].flag)

    # If no periods identified (e.g. lots of missing data) return nans
    else:
        flags = [np.nan for i in range(len(df['val']))]

    return flags


# ++++++++ Streak checks ++++++++

# Streaks: This is where you see the same value repeated in a run.
# Currently this records streaks of 2hrs in a row or more over 2 x Monthly mean rainfall.
# It is considered to be unlikely that you would see even 2 consecutive large rainfall amounts.
# For this code I have substituted the monthly mean rainfall for SDII as I want the thresholds
# to be independent of the rainfall time series as the global dataset is of highly variable quality.

def streaks_check(its):
    # Find wet day rainfall threshold (for streaks of any length)
    # mean_wetday_val, mean_wetday_val_filled = getSDII(its.latitude, its.longitude)
    mean_wetday_val, mean_wetday_val_filled = get_sdii(its)
    threshold = mean_wetday_val * 2.0
    if np.isnan(mean_wetday_val):
        threshold = mean_wetday_val_filled * 2.0

    # Lag values forwards and backwards to help identify consecutive value streaks
    df = its.data.copy().to_frame()
    df.columns = ['val']
    df['prev'] = df.shift(1)['val']
    df['next'] = df.shift(-1)['val']
    df['prev'] = np.where(df['prev'].isnull(), 0, df['prev'])
    df['next'] = np.where(df['next'].isnull(), 0, df['next'])

    # Look for streaks of values exceeding 2 * mean wet day rainfall
    df1 = pd.DataFrame(
        dict(start=np.flatnonzero((df.val != df.prev) & (df.val == df.next) &
                                  (df.val >= threshold)),
             end=np.flatnonzero((df.val == df.prev) & (df.val != df.next) &
                                (df.val >= threshold))))
    df1['diff'] = df1['end'] - df1['start'] + 1

    # Calculate annual minimum data value >0 in each year (added FL09)
    df99 = df[df['val'] > 0.0].groupby(df[df['val'] > 0.0].index.year)['val'].agg('min')
    df99 = df99.to_frame()
    df99.rename({'val': 'year_min'}, axis=1, inplace=True)
    df99['year'] = df99.index

    # Ensure that year_min is not too small (FL10) - set minimum as 0.1
    # - issues with small numbers, especially where minimum changes between years
    # - also ensure not too large (<= 2.6, i.e. US tip resolution)
    df99['year_min'] = np.where(df99['year_min'] < 0.1, 0.1, df99['year_min'])
    df99['year_min'] = np.where(df99['year_min'] > 2.6, 2.6, df99['year_min'])

    # Add annual minimum data value >0 as column (added FL09)
    df['year'] = df.index.year
    df['datetime'] = df.index
    df = df.merge(df99, how='left', left_on='year', right_on='year')
    df.drop('year', axis=1, inplace=True)
    df.set_index('datetime', inplace=True)

    # Look for streaks of consecutive values (changed from >0 to any 
    # magnitude > year minimum data value above 0 in FL09)
    try:
        df2 = pd.DataFrame(
            dict(start=np.flatnonzero((df.val != df.prev) & (df.val == df.next) &
                                      (df.val > df.year_min)),
                 end=np.flatnonzero((df.val == df.prev) & (df.val != df.next) &
                                    (df.val > df.year_min))))

    # If above fails then use one value for all years as threshold, based on 
    # maximum of annual minima, ensuring >= 0.1 and <= 2.6 (done above) (FL10)
    except:
        min_threshold = np.max(df99['year_min'])
        df2 = pd.DataFrame(
            dict(start=np.flatnonzero((df.val != df.prev) & (df.val == df.next) &
                                      (df.val > min_threshold)),
                 end=np.flatnonzero((df.val == df.prev) & (df.val != df.next) &
                                    (df.val > min_threshold))))

    # Subset on periods of >= 12 consecutive values
    df2['diff'] = df2['end'] - df2['start'] + 1
    df2 = df2.loc[df2['diff'] >= 12]

    flag = 1
    if np.isnan(mean_wetday_val):
        flag = 2
    df1['flag'] = flag
    df2['flag'] = 3
    df3 = df1.append(df2)

    # Make list of flags
    flags = [0 for i in range(len(df['val']))]
    for row in df3.iterrows():
        for i in range(row[1].start, row[1].end + 1):
            flags[i] = row[1].flag

    return flags

# ++++++++ Change in minimum value check ++++++++


# Change in minimum value: This is an homogeneity check to see if the resolution of the data has changed.
# Currently, I am including a flag if there is a year of no data as that seems pretty bad to me.

# Alternative implementation to return list of years where the minimum value >0
# differs from the data precision/resolution identified in the raw (pre-QC) files
def change_in_min_val_check(its):
    # Filter on values >0
    df = its.data[its.data > 0.0].to_frame()
    df.columns = ['val']

    # Find minimum by year
    df = df.groupby(df.index.year).min()

    # List of years differing from inferred precision in raw (pre-QC) data files
    df = df.loc[df['val'] != its.resolution]
    flag_years = df.index.tolist()
    if len(flag_years) > 0:
        change_flag = 1
    else:
        change_flag = 0

    return [change_flag, flag_years]


"""
++++++++++++++++++++++++++ Neighbour Checks - Basic functions +++++++++++++++++++++++++++
"""


# Coordinate system conversion
def geodetic_to_ecef(lat, lon, h):
    a = 6378137
    b = 6356752.3142
    f = (a - b) / a
    e_sq = f * (2 - f)
    lamb = math.radians(lat)
    phi = math.radians(lon)
    s = math.sin(lamb)
    n = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x = (h + n) * cos_lambda * cos_phi
    y = (h + n) * cos_lambda * sin_phi
    z = (h + (1 - e_sq) * n) * sin_lambda

    return x, y, z


# Helper function, finds hourly neighbour stations ++++++++++++++++++++++++++++
def find_hourly_neighbours(target):
    # float("nan") returns np.nan so needs to be handled separately (occurs in some Italy (Sicily) files)
    # whereas float("NA") returns value error (i.e. convention in most raw/formatted files)
    try:
        if elv != "nan":
            elv = float(target.elevation)
        else:
            elv = 100
    except:
        elv = 100

    converted_hourly_coords = geodetic_to_ecef(target.latitude, target.longitude, elv)
    dist, index = hourly_n_tree.query(converted_hourly_coords,
                                      k=30)
    # K needs to be equal or less than the number
    # of stations available in the database
    overlap = []
    paired_stations = []
    distance = []
    paths = []

    hourly_dates = (target.start_datetime, target.end_datetime)

    counter = 0
    for i in range(len(dist)):
        dci = index[i]
        pol, ol = calculate_overlap(hourly_dates, hourly_n_dates[dci])
        ps = hourly_n_names[dci]
        di = dist[i]
        pa = hourly_n_paths[dci]

        if di < 50000:  # must be within 50km
            if ol > 365 * 3:  # must have at least 3 years overlap
                if counter < 11:  # want to select the closest 10, but the first one is always the target itself
                    overlap.append(ol)
                    paired_stations.append(ps)
                    distance.append(di)
                    paths.append(pa)
                    counter += 1

    if len(paired_stations) >= 3:
        return [paired_stations, paths]
    else:
        return [[], []]


# Helper function, finds daily neighbour stations +++++++++++++++++++++++++++++
def find_daily_neighbours(target):
    try:
        elv = float(target.elevation)
    except:
        elv = 100

    converted_hourly_coords = geodetic_to_ecef(target.latitude, target.longitude, elv)

    dist, index = tree.query(converted_hourly_coords, k=30)

    overlap = []
    paired_stations = []
    distance = []

    hourly_dates = (target.start_datetime, target.end_datetime)

    counter = 0
    for i in range(len(dist)):
        dci = index[i]
        pol, ol = calculate_overlap(hourly_dates, daily_dates[dci])
        ps = daily_names[dci]
        di = dist[i]

        if di < 50000:  # must be within 50km
            if ol > 365 * 3:  # must have at least 3 years overlap
                if counter < 10:  # want to select the closest 10
                    overlap.append(ol)
                    paired_stations.append(ps)
                    distance.append(di)
                    counter += 1

    if len(paired_stations) >= 3:
        return paired_stations
    else:
        return []


# Helper function, finds daily neighbour stations +++++++++++++++++++++++++++++
def find_monthly_neighbours(target):
    try:
        elv = float(target.elevation)
    except:
        elv = 100

    converted_hourly_coords = geodetic_to_ecef(target.latitude, target.longitude, elv)

    dist, index = monthly_tree.query(converted_hourly_coords, k=30)

    overlap = []
    paired_stations = []
    distance = []

    hourly_dates = (target.start_datetime, target.end_datetime)

    counter = 0
    for i in range(len(dist)):
        mci = index[i]
        pol, ol = calculate_overlap(hourly_dates, monthly_dates[mci])
        ps = monthly_names[mci]
        di = dist[i]

        if di < 50000:  # must be within 50km
            if ol > 365 * 3:  # must have at least 3 years overlap
                if counter < 10:  # want to select the closest 10
                    overlap.append(ol)
                    paired_stations.append(ps)
                    distance.append(di)
                    counter += 1

    if len(paired_stations) >= 3:
        return paired_stations
    else:
        return None


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def calculate_overlap(period1, period2):
    # taking period2 as the reference (daily) period
    r1, r2 = period2
    t1, t2 = period1

    if t1 >= r1 and t2 <= r2:
        p = 100
        rd = (t2 - t1).days

    elif t1 < r1 and t2 < r1:
        p = 0
        rd = 0

    elif t1 > r2 and t2 > r2:
        p = 0
        rd = 0

    elif t1 <= r1 and t2 >= r2:
        rd = (r2 - r1).days
        td = (t2 - t1).days

        try:
            p = rd * 100 / td
        except:
            p = -999

    elif t1 < r1 and t2 >= r1 and t2 < r2:
        rd = (t2 - r1).days
        td = (t2 - t1).days

        try:
            p = rd * 100 / td
        except:
            p = -999

    elif t1 > r1 and t1 <= r2 and t2 > r2:
        rd = (r2 - t1).days
        td = (t2 - t1).days

        try:
            p = rd * 100 / td
        except:
            p = -999

    else:
        p = -999

    return p, rd


# ++++++++++++++++++++++++++++++++++ GPCC functions +++++++++++++++++++++++++++++++++++ LIZ WILL CHANGE THIS


def get_gpcc(start_year, end_year, gpcc_id):
    gpcc_filename = "tw_" + gpcc_id + ".dat"
    dat_path = "/media/nas/x21971/GPCC_daily2/tw_" + gpcc_id + ".dat"
    zip_path = "/media/nas/x21971/GPCC_daily2/tw_" + gpcc_id + ".zip"
    if not os.path.exists(zip_path):
        if not os.path.exists(dat_path):
            p = subprocess.Popen(["get_zeitreihe_tw_by_id.sh", str(start_year), str(end_year), gpcc_id],
                                 cwd="/media/nas/x21971/GPCC_daily2")
            p.wait()
            time.sleep(0.1)

            # Move retrieved .dat file to its own .zip folder
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.write(dat_path, arcname=gpcc_filename, compress_type=zipfile.ZIP_DEFLATED)
            time.sleep(0.1)
            os.remove(dat_path)

    zf = zipfile.ZipFile(zip_path, "r")
    f = zf.open(gpcc_filename, "r")
    f.readline()

    dates = []
    vals = []

    file_format_error = 0
    try_again = 1
    for line in f:
        line_list = line.rstrip().split()
        # inserted try as at least one file has a dubious second header line
        try:
            dates.append(datetime.date(int(line_list[2]), int(line_list[1]), int(line_list[0])))
            read_values = 1
        except:
            if try_again == 1:
                try_again = 0
                read_values = 0
            elif try_again == 0:
                file_format_error = 1
                read_values = 0
                break
        if read_values == 1:
            got_one = 0
            for v in line_list[3:]:
                if float(v) >= 0:
                    vals.append(float(v))
                    got_one = 1
                    break
            if got_one == 0:
                vals.append(np.nan)

    f.close()
    zf.close()

    if file_format_error == 0:
        ddf = pd.DataFrame(vals, index=dates, columns=["GPCC"])

        # Remove duplicate rows if present and check against unique index values
        ddf['date_tmp'] = ddf.index.copy()
        idx = ddf.index.drop_duplicates().copy()
        ddf.drop_duplicates(inplace=True)
        ddf.sort_index(inplace=True)
        ddf.drop(['date_tmp'], axis=1, inplace=True)
        if ddf.shape[0] != idx.shape[0]:
            vals = []
            dates = []
            ddf = pd.DataFrame(vals, index=dates, columns=["GPCC"])

    elif file_format_error == 1:
        vals = []
        dates = []
        ddf = pd.DataFrame(vals, index=dates, columns=["GPCC"])

    return ddf


# Helper function to access Global Sub Daily Rainfall database (a.k.a. Intense Database)
def get_gsdr(gsdr_id, path):
    zfh = zipfile.ZipFile(path, "r")

    d = zfh.open(gsdr_id + ".txt", mode="r")
    df = ex.read_intense(d, only_metadata=False, opened=True).data.to_frame("GSDR")
    d.close()
    zfh.close()

    # convert hourly to daily 7am-7am
    df["roll"] = np.around(df.GSDR.rolling(window=24, center=False, min_periods=24).sum(), 1)

    dfd = df[df.index.hour == 7]
    dts = list(dfd.index)
    daily_vals = list(dfd.roll)
    dday = []
    for hday in dts:
        s0 = hday - datetime.timedelta(days=1)
        dday.append(datetime.date(s0.year, s0.month, s0.day))

    gsdr = pd.Series(daily_vals, index=dday).to_frame("ts2")

    return gsdr


# Helper function to access Global Precipitation Climatology Centre monthly data...
# Ask Liz what is needed to implement this!
def get_monthly_gpcc(start_year, end_year, gpcc_id):  # Hey liz! Check this once you have access to monthly!

    gpcc_filename = "mw_" + gpcc_id + ".dat"
    dat_path = "/media/nas/x21971/GPCC_monthly2/mw_" + gpcc_id + ".dat"
    zip_path = "/media/nas/x21971/GPCC_monthly2/mw_" + gpcc_id + ".zip"
    if not os.path.exists(zip_path):
        if not os.path.exists(dat_path):
            p = subprocess.Popen(["get_zeitreihe_mw_by_id.sh", str(start_year), str(end_year), gpcc_id],
                                 cwd="/media/nas/x21971/GPCC_monthly2")
            p.wait()
            time.sleep(0.1)

            # Move retrieved .dat file to its own .zip folder
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.write(dat_path, arcname=gpcc_filename, compress_type=zipfile.ZIP_DEFLATED)
            time.sleep(0.1)
            os.remove(dat_path)

    zf = zipfile.ZipFile(zip_path, "r")
    f = zf.open(gpcc_filename, "r")
    f.readline()

    dates = []
    vals = []

    file_format_error = 0
    for line in f:
        line_list = line.rstrip().split()
        try:
            year = int(line_list[1])
            month = int(line_list[0])
            read_values = 1
        except:
            file_format_error = 1
            read_values = 0
            break
        if read_values == 1:
            day = calendar.monthrange(year, month)[1]
            dates.append(datetime.date(year, month, day))
            got_one = 0
            for v in line_list[2:]:
                if float(v) >= 0:
                    vals.append(float(v))
                    got_one = 1
                    break
            if got_one == 0:
                vals.append(np.nan)

    f.close()
    zf.close()

    # ddf = pd.DataFrame(vals, index=dates, columns=["GPCC"])

    if file_format_error == 0:
        ddf = pd.DataFrame(vals, index=dates, columns=["GPCC"])

        # Remove duplicate rows if present and check against unique index values
        ddf['date_tmp'] = ddf.index.copy()
        idx = ddf.index.drop_duplicates().copy()
        ddf.drop_duplicates(inplace=True)
        ddf.sort_index(inplace=True)
        ddf.drop(['date_tmp'], axis=1, inplace=True)
        if ddf.shape[0] != idx.shape[0]:
            vals = []
            dates = []
            ddf = pd.DataFrame(vals, index=dates, columns=["GPCC"])
            print(file_format_error, ddf.shape[0])

    elif file_format_error == 1:
        vals = []
        dates = []
        ddf = pd.DataFrame(vals, index=dates, columns=["GPCC"])
        print(file_format_error, ddf.shape[0])

    return ddf


# Match station id helper function ++++++++++++++++++++++++++++++++++++++++++++
def find_identical_by_id(target, neighbour):  # this probably needs refining...
    match = 0
    if target.station_id[3:] in neighbour.name:
        match = 1
    if target.station_id[3:] in neighbour.station_id:
        match = 1
    if target.station_id[3:] in neighbour.wmo_id:
        match = 1
    if target.original_station_number in neighbour.name:
        match = 1
    if target.original_station_number in neighbour.station_id:
        match = 1
    if target.original_station_number in neighbour.wmo_id:
        match = 1
    if target.original_station_name in neighbour.name:
        match = 1
    if target.original_station_name in neighbour.station_id:
        match = 1
    if target.original_station_name in neighbour.wmo_id:
        match = 1

    return match


"""
++++++++++++++++++++++++++++++++++ GPCC functions -end +++++++++++++++++++++++++++++++++++
"""

"""
++++++++++++++++++++++++++++++++++ Neighbour Checks +++++++++++++++++++++++++++++++++++
"""


def calculate_affinity_index_and_pearson(df1, df2):  # done

    df = pd.concat([df1, df2], axis=1, join='inner')
    df.columns = ["ts1", "ts2"]
    df = df.dropna()

    # dp 29/11/2019 - extra check to make sure there is some overlap between
    # hourly target and neighbour - implemented because at least one error in
    # GPCC statlex_daily
    # - also possibility that dropping nans from df reduces overlap
    if df.shape[0] >= 365:

        # p = 0.1 # 0.0 # 1.0
        a = np.around(df.loc[df['ts1'] >= 0.1, 'ts1'].min(), 1)
        b = np.around(df.loc[df['ts2'] >= 0.1, 'ts2'].min(), 1)
        p = max(a, b, 0.1)
        conditions = [
            (df['ts1'] > p) & (df['ts2'] > p),
            (df['ts1'] == p) & (df['ts2'] == p),
            (df['ts1'] == p) & (df['ts2'] > p),
            (df['ts1'] > p) & (df['ts2'] == p)]
        choices = [1, 1, 0, 0]

        df['duplicate'] = np.select(conditions, choices, default=np.nan)

        try:
            match = df["duplicate"].value_counts()[1]
        except:
            match = 0
        try:
            diff = df["duplicate"].value_counts()[0]
        except:
            diff = 0

        if (match > 0) or (diff > 0):
            perc = match / (match + diff)
            p_corr = df.ts1.corr(df.ts2)
            df["factor"] = df.ts1 / df.ts2
            f = np.mean(df.loc[(df.ts1 > 0) & (df.ts2 > 0), "factor"].values)
        else:
            perc = 0
            p_corr = 0
            f = 0

    else:
        perc = 0
        p_corr = 0
        f = 0

    return perc, p_corr, f


# Main helper function, used by check_neighbour and check_neighbourDry
def compare_target_to_neighbour(target, neighbour, high_or_dry, station=None, check_type=None,
                                neighbour_id=None):

    """ After Upton and Rahimi 2003 https://www.sciencedirect.com/science/article/pii/S0022169403001422

    last three args for output for
    normalised difference distribution checks
    """

    check_flag = 1  # default setting is to check

    # dp 31/12/2019 - this AI check should be redundant, because neighbours with
    # AI < 0.9 are filtered out before this function is called...
    if high_or_dry == "high":
        ai = calculate_affinity_index_and_pearson(target, neighbour)[0]
        if ai >= 0.9:
            check_flag = 1
        else:
            check_flag = 0

    if check_flag == 1:

        df = pd.concat([target, neighbour], axis=1, join='inner')
        df.columns = ["ts1", "ts2"]
        df = df.dropna()

        # There are cases where intermittent records cause problems in 15-days
        # windows for dry spell stuff, so try to ensure reasonable overlap
        if df.shape[0] >= 365:

            # Separate out high and dry checks as using slightly different approaches now
            if high_or_dry == "high":

                # Normalise target and neighbour series by their respective min/max and
                # find differences
                # - retained actual amounts too to help filter on wet days
                df['ts1n'] = (df['ts1'] - df['ts1'].min()) / (df['ts1'].max() - df['ts1'].min())
                df['ts2n'] = (df['ts2'] - df['ts2'].min()) / (df['ts2'].max() - df['ts2'].min())
                df["nd"] = df['ts1n'] - df['ts2n']

                # ----------
                # *** dp 31/12/2019 - rather than using a median/standard deviation approach to
                # identifying outlying differences, switching to an approach based on presumption
                # that normalised differences for wet day amounts are roughly exponentially
                # distributed (based on testing)

                # Filter for target wet days, no NAs and positive differences
                df1 = df.loc[(df['ts1'] >= 1.0) & (np.isfinite(df['ts1'])) &
                             (np.isfinite(df['ts2'])) & (df['nd'] > 0.0)]

                # Ensure still some data left to fit a distribution etc
                if df1.shape[0] >= 30:

                    # Fit exponential distribution
                    params = scipy.stats.expon.fit(df1['nd'])

                    # Calculate thresholds at key percentiles of fitted distribution
                    q95 = scipy.stats.expon.ppf(0.95, params[0], params[1])
                    q99 = scipy.stats.expon.ppf(0.99, params[0], params[1])
                    q999 = scipy.stats.expon.ppf(0.999, params[0], params[1])

                    # Assign flags
                    # - no need for an additional condition that target exceeds wet day threshold
                    # because the percentiles are defined based on just positive differences?
                    # -- left in for now...
                    conditions = [
                        (df['ts1'] >= 1.0) & (df['nd'] <= q95),
                        (df['ts1'] >= 1.0) & (df['nd'] > q95) & (df['nd'] <= q99),
                        (df['ts1'] >= 1.0) & (df['nd'] > q99) & (df['nd'] <= q999),
                        (df['ts1'] >= 1.0) & (df['nd'] > q999)]
                    choices = [0, 1, 2, 3]

                    df['temp_flags'] = np.select(conditions, choices, default=0)

                    temp_flags = df['temp_flags']
                    return temp_flags

                else:
                    return pd.Series([])

            elif high_or_dry == "dry":

                # Assign flags
                # - consider only whether dry 15-day periods at the target are
                # corroborated as dry by neighbours
                # - check based on whether 0, 1, 2 or >= 3 wet days are recorded at the
                # neighbour when the target is dry over the 15-day period
                # - dry flag works on the basis of fraction of dry days within 15-day
                # moving window, so 1 = all dry, 0 = all wet
                # -- truncating these fractions to 2 dp below and manipulating equalities
                # to work with these fractions, but could work in days not fractions if
                # change the convertToDrySpell function
                # - in dry day fraction calcs a threshold of 0 mm is currently used to
                # identify days as wet (i.e. any rainfall)
                frac_drydays = {}
                for d in range(1, 3 + 1):
                    frac_drydays[d] = np.trunc((1.0 - (float(d) / 15.0)) * 10 ** 2) / (10 ** 2)
                conditions = [
                    (df['ts1'] == 1.0) & (df['ts2'] == 1.0),
                    (df['ts1'] == 1.0) & (df['ts2'] < 1.0) & (df['ts2'] >= frac_drydays[1]),
                    (df['ts1'] == 1.0) & (df['ts2'] < frac_drydays[1]) & (df['ts2'] >= frac_drydays[2]),
                    (df['ts1'] == 1.0) & (df['ts2'] < frac_drydays[2])]  # & (df['ts2'] >= frac_drydays[3])
                choices = [0, 1, 2, 3]

                # *** dp 27/11/2019 *** - commented out line below so changed to default=0
                # normalized_df['temp_flags'] = np.select(conditions, choices, default=np.nan)
                # normalized_df['temp_flags'] = np.select(conditions, choices, default=0)
                df['temp_flags'] = np.select(conditions, choices, default=0)

                # tempFlags = normalized_df['temp_flags']
                temp_flags = df['temp_flags']
                return temp_flags

        else:
            return pd.Series([])

    else:
        return pd.Series([])


# Monthly checks ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def compare_target_to_neighbour_monthly(target, neighbour):
    df = pd.concat([target, neighbour], axis=1, join='inner')

    df = df.dropna().rename(columns={"GPCC": "ts2", "target": "ts1"})

    df["perc_diff"] = (df.ts1 - df.ts2) * 100. / df.ts2
    df["factor_diff"] = df.ts1 / df.ts2
    conditions = [
        (df['perc_diff'] <= -100.0),
        (df['perc_diff'] <= -50.0) & (df['perc_diff'] > -100.0),
        (df['perc_diff'] <= -25.0) & (df['perc_diff'] > -50),
        (df['perc_diff'] < 25.0) & (df['perc_diff'] > -25.0),
        (df['perc_diff'] >= 25.0) & (df['perc_diff'] < 50.0),
        (df['perc_diff'] >= 50.0) & (df['perc_diff'] < 100.0),
        (df['perc_diff'] >= 100.0)]

    choices = [-3, -2, -1, 0, 1, 2, 3]

    df['temp_flags'] = np.select(conditions, choices, default=np.nan)

    df.loc[np.isnan(df['ts1']), 'temp_flags'] = np.nan
    temp_flags = df['temp_flags']

    conditions_f = [
        (df['factor_diff'] < 11) & (df['factor_diff'] > 9),  # hourly is approx 10x greater than monthly
        (df['factor_diff'] < 26) & (df['factor_diff'] > 24),  # hourly is approx 25.4x greater than monthly
        (df['factor_diff'] < 3) & (df['factor_diff'] > 2),  # hourly is approx 2.45x greater than monthly
        (df['factor_diff'] > 1 / 11) & (df['factor_diff'] < 1 / 9),
        (df['factor_diff'] > 1 / 26) & (df['factor_diff'] < 1 / 24),
        (df['factor_diff'] > 1 / 3) & (df['factor_diff'] < 1 / 2)]

    choices_f = [1, 2, 3, 4, 5, 6]

    df['factor_flags'] = np.select(conditions_f, choices_f, default=0)

    df.loc[np.isnan(df['ts1']), 'factor_flags'] = np.nan
    factor_flags = df['factor_flags']

    return [temp_flags, factor_flags]  # Hey Liz! Make sure theres a new thing for factor flags!


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Uses compare_target_to_neighbour function based on Upton and Rahimi(2003)
def check_neighbours(target, neighbours, station=None,
                     check_type=None):  # temporary extra args for checking neighbour stuff

    df = target
    concat_list = [df]

    nid = 1  # temporary for neighbours checking
    for n in neighbours:
        dfn = compare_target_to_neighbour(df, n, "high", station, check_type, nid)
        if dfn.empty:
            pass
        else:
            concat_list.append(dfn)
        nid += 1
    df = pd.concat(concat_list, axis=1, join='outer')

    cols = ["n" + str(i + 1) for i in range(len(concat_list) - 1)]
    cols2 = ["target"]
    cols2.extend(cols)

    df.columns = cols2

    # dp 28/11/2019 - changed assuming that looking for number of neighbours online
    # on any given day (similar to monthly neighbours)
    df["online"] = len(concat_list) - df[cols].isnull().sum(axis=1) - 1

    conditions = [
        ((df[cols] == 3).T.sum() == df.online),
        ((df[cols] >= 2).T.sum() == df.online),
        ((df[cols] >= 1).T.sum() == df.online)]

    choices = [3, 2, 1]
    df["flags"] = np.select(conditions, choices, default=0)

    df.loc[df.online < 3, "flags"] = np.nan
    dfr = df.flags

    return dfr


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Uses compareTargetToNeighbour function based on Upton and Rahimi(2003)
def check_neighbours_dry(target, neighbours):  # Liz check this

    df = convert_to_dry_spell(target)  # Liz, check column names

    concat_list = [df]
    for n in neighbours:
        nn = convert_to_dry_spell(n)  # Liz check column names
        dfn = compare_target_to_neighbour(df, nn, "dry")
        if dfn.empty:
            pass
        else:
            concat_list.append(dfn)

    df = pd.concat(concat_list, axis=1, join='outer')

    cols = ["n" + str(i + 1) for i in range(len(concat_list) - 1)]
    cols2 = ["target"]
    cols2.extend(cols)
    df.columns = cols2

    # dp 28/11/2019 - again assumed want count of number of neighbours online
    df["online"] = len(concat_list) - df[cols].isnull().sum(axis=1) - 1

    df["flags"] = np.floor(df[cols].sum(axis=1) / df.online)

    # *** dp 29/11/2019 - in last section below, why set flags to -999 when target equals 0? in this
    # case target is between 0 and 1 because it is the fraction of dry days in the 15-day period
    # ending on each date. 1 means all dry days in the period, so it should be values less than
    # 1 that are set to -999 i think - changed to that effect for now... ***
    # - may also be better to give a different flag than nan to show that the check has been done and
    #   not failed e.g. due to lack of neighbours?

    # *** dp 01/01/2020 - changed so that no adjustment to flags where period is
    # not totally dry (i.e. df.target < 1) on the basis that this is now handled in
    # the revised dry neighbours check (so just commented out line below)

    df.flags = df.flags.replace([np.inf, -np.inf], -999)
    df.loc[df.online < 3, "flags"] = -999
    df.flags = df.flags.astype(int)
    df.flags = df.flags.replace(-999, np.nan)
    dfr = df.flags

    return dfr


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def check_m_neighbours(target, neighbours):
    df = target
    concat_list = [df]
    ticker = 0
    for n in neighbours:
        dfn, dff = compare_target_to_neighbour_monthly(df, n)
        concat_list.append(dfn)
        if ticker == 0:
            df_factor = dff.copy()
        ticker += 1

    df = pd.concat(concat_list, axis=1, join='outer')

    cols = ["n" + str(i + 1) for i in range(len(concat_list) - 1)]
    cols2 = ["target"]
    cols2.extend(cols)

    df.columns = cols2

    df["online"] = len(concat_list) - df[cols].isnull().sum(axis=1) - 1
    conditions = [
        ((df[cols] == -3).T.sum() == df.online),
        ((df[cols] <= -2).T.sum() == df.online),
        ((df[cols] <= -1).T.sum() == df.online),
        ((df[cols] == 3).T.sum() == df.online),
        ((df[cols] >= 2).T.sum() == df.online),
        ((df[cols] >= 1).T.sum() == df.online)]

    choices = [-3, -2, -1, 3, 2, 1]

    df["flags"] = np.select(conditions, choices, default=0)
    df.loc[np.isnan(df['target']), 'flags'] = np.nan

    # Add additional checks in relation to monthly neighbours climatology

    # Calculate neighbour monthly climatology (monthly min/max across all neighbours)
    tmp = []
    for n in neighbours:
        tmp.append(n)
    df_mc = pd.concat(tmp, axis=1, join='outer')
    cols3 = ["n" + str(i + 1) for i in range(len(tmp))]
    df_mc.columns = cols3
    df_mc.index = pd.to_datetime(df_mc.index, format='%Y-%m-%d')
    df_mc_max = df_mc.groupby(df_mc.index.month).max()
    df_mc_max['max'] = df_mc_max.max(axis=1)
    df_mc_min = df_mc.groupby(df_mc.index.month).min()
    df_mc_min['min'] = df_mc_min.min(axis=1)
    df_mc2 = pd.concat(
        [df_mc_min.drop(cols3, axis=1), df_mc_max.drop(cols3, axis=1)], axis=1,
        join='outer')
    df_mc2['month'] = df_mc2.index

    # Join monthly climatology to target series
    df['month'] = df.index.month
    df['date'] = df.index
    df2 = df.merge(df_mc2, left_on='month', right_on='month')
    df2.set_index('date', inplace=True)
    df2.sort_index(inplace=True)

    # Adjust flag where -3 or 3 and rainfall is outside neighbours min/max range
    df2.loc[(df2['flags'] == -3) & (df2['online'] >= 3) &
            ((df2['target'] < (0.75 * df2['min'])) | (df2['target'] > (1.25 * df2['max']))),
            'flags'] = -4
    df2.loc[(df2['flags'] == 3) & (df2['online'] >= 3) &
            ((df2['target'] < (0.75 * df2['min'])) | (df2['target'] > (1.25 * df2['max']))),
            'flags'] = 4

    # Flag where less than 3 neighbours but value > 2 * neighbour max or
    # < 0.5 * neighbour min
    df2.loc[(df2['online'] < 3) & (df2['target'] > (2.0 * df2['max'])),
            'flags'] = 5
    df2.loc[(df2['online'] < 3) & (df2['target'] < (0.5 * df2['min'])),
            'flags'] = -5

    # If less than 3 stations online than flag check as incomplete unless flag
    # has a value of -5/5 (i.e. outside climatology range)
    df2.loc[(df2.online < 3) & (np.absolute(df2['flags']) != 5), "flags"] = np.nan
    dfr = df2.flags

    return [dfr, df_factor]


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def check_hourly_neighbours(target):
    df = target.data.to_frame("target")

    # convert hourly to daily 7am-7am
    df["roll"] = np.around(df.target.rolling(window=24, center=False, min_periods=24).sum(), 1)
    dfd = df[df.index.hour == 7]
    dts = list(dfd.index)
    daily_vals = list(dfd.roll)

    dts0 = []
    for dt in dts:
        s0 = dt - datetime.timedelta(days=1)
        dts0.append(datetime.date(s0.year, s0.month, s0.day))
    ts0 = pd.Series(daily_vals, index=dts0)

    neighbours, paths = find_hourly_neighbours(target)

    # dp 30/11/2019 - assuming neighbours[0] is the target
    if len(neighbours) > 1:

        # More GSDR bits here Liz: -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

        # get GSDR
        neighbour_dfs = []
        for nId in range(len(neighbours)):
            if nId == 0:
                pass
            else:
                neighbour_dfs.append(get_gsdr(neighbours[nId], paths[nId]))
        # get matching stats for nearest gauge and offset calculateAffinityIndexAndPearson(ts1, ts2) -> returns a flag

        # do neighbour check

        # filter out gauges with AI < 0.9
        neighbour_dfs2 = []
        for ndf in neighbour_dfs:
            nai, nr2, nf = calculate_affinity_index_and_pearson(ts0.to_frame("ts1"), ndf)
            if nai > 0.9:
                neighbour_dfs2.append(ndf)
            else:
                pass

        flags_df = check_neighbours(ts0.to_frame("ts1"), neighbour_dfs2, target.station_id, 'hourly')

        flags_dates = list(flags_df.index.values)
        flags_vals = list(flags_df)

        # do neighbour check for dry periods and flag the whole 15 day period
        dry_flags_df = check_neighbours_dry(ts0.to_frame("ts1"), neighbour_dfs2)
        dry_flags_dates = list(dry_flags_df.index.values)
        dry_flags_vals = list(dry_flags_df)

        i1 = []
        i2 = []
        i3 = []

        for i in range(len(dry_flags_vals)):
            if dry_flags_vals[i] == 1:
                for j in range(15):
                    i1.append(i - j)
            elif dry_flags_vals[i] == 2:
                for j in range(15):
                    i2.append(i - j)
            elif dry_flags_vals[i] == 3:
                for j in range(15):
                    i3.append(i - j)
            else:
                pass

        for i in i1:
            dry_flags_vals[i] = 1
        for i in i2:
            dry_flags_vals[i] = 2
        for i in i3:
            dry_flags_vals[i] = 3

        # add daily flags back onto hourly
        flags_dt = [datetime.datetime(d.year, d.month, d.day, 7) for d in flags_dates]
        flags_df = pd.Series(flags_vals, index=flags_dt).to_frame("flags")
        dry_flags_dt = [datetime.datetime(d.year, d.month, d.day, 7) for d in dry_flags_dates]
        dry_flags_df = pd.Series(dry_flags_vals, index=dry_flags_dt).to_frame("dryFlags")

        df = pd.concat([df, flags_df, dry_flags_df], axis=1, join_axes=[df.index])
        df.flags = df.flags.fillna(method="ffill", limit=23)
        df.dryFlags = df.dryFlags.fillna(method="ffill", limit=23)
        df.fillna(-999, inplace=True)

        return [list(df.flags.astype(int)), list(df.dryFlags.astype(int))]

    else:
        tmp = [-999 for i in range(df['roll'].shape[0])]
        return [tmp, tmp]


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def check_daily_neighbours(target):
    df = target.data.to_frame("target")
    # convert hourly to daily 7am-7am
    df["roll"] = np.around(df.target.rolling(window=24, center=False, min_periods=24).sum(), 1)
    dfd = df[df.index.hour == 7]
    dts = list(dfd.index)
    daily_vals = list(dfd.roll)

    # offset by one day in either direction
    dtsm1 = []
    dts0 = []
    dtsp1 = []
    for dt in dts:
        sm1 = dt - datetime.timedelta(days=2)
        s0 = dt - datetime.timedelta(days=1)
        sp1 = dt

        dtsm1.append(datetime.date(sm1.year, sm1.month, sm1.day))
        dts0.append(datetime.date(s0.year, s0.month, s0.day))
        dtsp1.append(datetime.date(sp1.year, sp1.month, sp1.day))

    tsm1 = pd.Series(daily_vals, index=dtsm1)
    ts0 = pd.Series(daily_vals, index=dts0)
    tsp1 = pd.Series(daily_vals, index=dtsp1)

    # find neighbours
    neighbours = find_daily_neighbours(target)

    # Check for duplicate neighbours
    if len(neighbours) > 0:
        tmp = []
        for n in neighbours:
            if n not in tmp:
                tmp.append(n)
        neighbours = tmp.copy()

    # dp 30/11/2019
    if len(neighbours) > 0:

        # get gpcc
        neighbour_dfs = []
        for nId in neighbours:
            neighbour_start_year = daily_dates[daily_names.index(nId)][0].year
            neighbour_end_year = daily_dates[daily_names.index(nId)][1].year
            neighbour_dfs.append(get_gpcc(neighbour_start_year, neighbour_end_year, nId))

        # get matching stats for nearest gauge and offset calculateAffinityIndexAndPearson(ts1, ts2) -> returns a flag
        nearest = neighbour_dfs[0].rename(columns={"GPCC": "ts2"})
        sm1ai, sm1r2, sm1f = calculate_affinity_index_and_pearson(tsm1.to_frame("ts1"), nearest)
        s0ai, s0r2, s0f = calculate_affinity_index_and_pearson(ts0.to_frame("ts1"), nearest)
        sp1ai, sp1r2, sp1f = calculate_affinity_index_and_pearson(tsp1.to_frame("ts1"), nearest)

        ais = [sm1ai, s0ai, sp1ai]
        r2s = [sm1r2, s0r2, sp1r2]

        if ais.index(max(ais)) == r2s.index(max(r2s)):
            offset_flag = ais.index(max(ais)) - 1
        else:
            offset_flag = 0

        # do neighbour check
        # print("doing neighbour check")

        # dp 29/11/2019 - check that there is indeed some overlap between the hourly and GPCC
        # daily gauge - for DE_02483 one neighbour (3798) ends in 1972 in the data file but
        # statlex_daily says it continues until 2018, which results in no overlap and
        # a divide by zero error when trying to calculate the percentage matching
        # - for now check placed in calculate AI etc function

        # filter out gauges with AI < 0.9
        neighbour_dfs2 = []
        for nId, ndf in zip(neighbours, neighbour_dfs):
            ndf2 = ndf.rename(columns={"GPCC": "ts2"})
            nai, nr2, nf = calculate_affinity_index_and_pearson(ts0.to_frame("ts1"), ndf2)
            if nai > 0.9:
                neighbour_dfs2.append(ndf)
            else:
                pass

        flags_df = check_neighbours(ts0.to_frame("ts1"), neighbour_dfs2, target.station_id, 'daily')
        flags_dates = list(flags_df.index.values)
        flags_vals = list(flags_df)

        # do neighbour check for dry periods and flag the whole 15 day period
        dry_flags_df = check_neighbours_dry(ts0.to_frame("ts1"), neighbour_dfs2)
        dry_flags_dates = list(dry_flags_df.index.values)
        dry_flags_vals = list(dry_flags_df)

        i1 = []
        i2 = []
        i3 = []

        for i in range(len(dry_flags_vals)):
            if dry_flags_vals[i] == 1:
                for j in range(15):
                    i1.append(i - j)
            elif dry_flags_vals[i] == 2:
                for j in range(15):
                    i2.append(i - j)
            elif dry_flags_vals[i] == 3:
                for j in range(15):
                    i3.append(i - j)
            else:
                pass

        for i in i1:
            dry_flags_vals[i] = 1
        for i in i2:
            dry_flags_vals[i] = 2
        for i in i3:
            dry_flags_vals[i] = 3

        # add daily flags back onto hourly
        flags_dt = [datetime.datetime(d.year, d.month, d.day, 7) for d in flags_dates]
        flags_df = pd.Series(flags_vals, index=flags_dt).to_frame("flags")
        dry_flags_dt = [datetime.datetime(d.year, d.month, d.day, 7) for d in dry_flags_dates]
        dry_flags_df = pd.Series(dry_flags_vals, index=dry_flags_dt).to_frame("dryFlags")

        df = pd.concat([df, flags_df, dry_flags_df], axis=1, join_axes=[df.index])
        df.flags = df.flags.fillna(method="ffill", limit=23)
        df.dryFlags = df.dryFlags.fillna(method="ffill", limit=23)
        df.fillna(-999, inplace=True)
        return [list(df.flags.astype(int)), offset_flag, s0ai, s0r2, s0f, list(df.dryFlags.astype(int))]

    # -999 if no neighbours
    else:
        tmp = [-999 for i in range(df['roll'].shape[0])]
        return [tmp, -999, -999, -999, -999, tmp]


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def convert_to_dry_spell(daily_df):
    # dp 29/11/2019 - it would make sense to remove np.around so fractional not binary,
    # but this will require a change to how the flagging is done for it to make sense
    # i think (i.e. do all stations agree the period is wet when the target is dry?)
    # should the threshold for dry be larger than just zero?

    daily_df["fracDryDays"] = daily_df.rolling(15, min_periods=15).apply(lambda window: (window == 0).sum() / 15)

    converted_df = daily_df["fracDryDays"]
    converted_df.columns = ["ts1"]
    return converted_df


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def check_monthly_neighbours(
        target):  # Hey Liz! check this when you have access to monthly, esp mapping back onto hourly.
    df = target.data.to_frame("target")

    # convert hourly to daily 7am-7am
    dfm = df.resample("M", label='right', closed='right').apply(lambda x: x.values.sum())
    # find neighbours
    neighbours = find_monthly_neighbours(target)

    # Check for duplicate neighbours
    if neighbours is not None:
        tmp = []
        for n in neighbours:
            if n not in tmp:
                tmp.append(n)
        neighbours = tmp.copy()

    if neighbours is None:
        hourly_flags_s = df.copy()
        hourly_flags_s['flags'] = -999
        hourly_factor_flags_s = df.copy()
        hourly_factor_flags_s['factor_flags'] = -999
    else:

        # get gpcc
        neighbour_dfs = []
        for n_id in neighbours:
            neighbour_start_year = monthly_dates[monthly_names.index(n_id)][0].year
            neighbour_end_year = monthly_dates[monthly_names.index(n_id)][1].year
            neighbour_dfs.append(get_monthly_gpcc(neighbour_start_year, neighbour_end_year, n_id))
        # get matching stats for nearest gauge and offset calculateAffinityIndexAndPearson(ts1, ts2) -> returns a flag

        # do neighbour check

        flags_df, factor_flags_df = check_m_neighbours(dfm, neighbour_dfs)

        # set dates to be at 2300 (rather than 0000) so bfill works
        flags_df.index += datetime.timedelta(hours=23)
        factor_flags_df.index += datetime.timedelta(hours=23)

        orig_dates = list(df.index.values)
        hourly_flags_s = flags_df.reindex(orig_dates, method="bfill")
        hourly_factor_flags_s = factor_flags_df.reindex(orig_dates, method="bfill")

        # count valid values within month and set flag as nan if more than 5% of data is missing
        # - hourly percentage differences
        hourly_flags_s = hourly_flags_s.to_frame()
        hourly_flags_s['count'] = hourly_flags_s.groupby(
            [hourly_flags_s.index.year, hourly_flags_s.index.month]).transform('count')
        hourly_flags_s['expected'] = hourly_flags_s.index.days_in_month * 24
        hourly_flags_s['frac_complete'] = hourly_flags_s['count'] / hourly_flags_s['expected']
        hourly_flags_s.loc[hourly_flags_s['frac_complete'] < 0.95, 'flags'] = np.nan
        hourly_flags_s.drop(['count', 'expected', 'frac_complete'], axis=1, inplace=True)
        # - hourly factor differences
        hourly_factor_flags_s = hourly_factor_flags_s.to_frame()
        hourly_factor_flags_s['count'] = hourly_factor_flags_s.groupby(
            [hourly_factor_flags_s.index.year, hourly_factor_flags_s.index.month]).transform('count')
        hourly_factor_flags_s['expected'] = hourly_factor_flags_s.index.days_in_month * 24
        hourly_factor_flags_s['frac_complete'] = hourly_factor_flags_s['count'] / hourly_factor_flags_s['expected']
        hourly_factor_flags_s.loc[hourly_factor_flags_s['frac_complete'] < 0.95, 'factor_flags'] = np.nan
        hourly_factor_flags_s.drop(['count', 'expected', 'frac_complete'], axis=1, inplace=True)

        hourly_flags_s.fillna(-999, inplace=True)
        hourly_factor_flags_s.fillna(-999, inplace=True)

    return [list(hourly_flags_s['flags'].astype(int)), list(hourly_factor_flags_s['factor_flags'].astype(int))]


# +++++++++++++++++++++++++++++++ MAIN FUNCTION, CALLS CHECKS ++++++++++++++++++++++++++++++++


def get_flags(ito):  # pass intense object

    # Ensure non-nan lat/lon before neighbour checks (issue for some Sicily stations)
    if np.isfinite(ito.latitude) and np.isfinite(ito.longitude):
        ito.QC_hourly_neighbours, ito.QC_hourly_neighbours_dry = check_hourly_neighbours(ito)

        ito.QC_daily_neighbours, ito.QC_offset, ito.QC_preQC_affinity_index, ito.QC_preQC_pearson_coefficient, ito.QC_factor_daily, ito.QC_daily_neighbours_dry = check_daily_neighbours(ito)

        ito.QC_monthly_neighbours, ito.QC_factor_monthly = check_monthly_neighbours(ito)

    ito.QC_world_record = world_record_check_ts(ito)

    ito.QC_Rx1day = rx_1_day_check_ts(ito)

    ito.QC_CDD = cdd_check(ito)

    ito.QC_daily_accumualtions = daily_accums_check(ito)

    ito.QC_monthly_accumulations = monthly_accums_check(ito)

    ito.QC_streaks = streaks_check(ito)

    ito.QC_percentiles = check_percentiles(ito)

    ito.QC_k_largest = check_k_largest(ito)

    ito.QC_days_of_week = check_days_of_week(ito)

    ito.QC_hours_of_day = check_hours_of_day(ito)

    ito.QC_intermittency = check_intermittency(ito)

    ito.QC_breakpoint = check_break_point(ito)

    ito.QC_R99pTOT = r99ptot_check_annual(ito)

    ito.QC_PRCPTOT = prcptot_check_annual(ito)

    ito.QC_change_min_value = change_in_min_val_check(ito)

    return ito


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Process a folder (country) - q argument used in multiprocessing

def process_folder(folder_to_check, q=None):
    print(folder_to_check)
    if not os.path.exists(qc_folder + "/" + folder_to_check[:-4]):
        os.makedirs(qc_folder + "/" + folder_to_check[:-4])

    error_path = qc_folder + '/' + folder_to_check[:-4] + "/error_" + folder_to_check[:-4] + ".txt"
    if not os.path.exists(error_path):
        error_file = open(error_path, "w")
    else:
        error_file = open(error_path, "a")
    existing_files = os.listdir(qc_folder + "/" + folder_to_check[:-4])
    zf = zipfile.ZipFile(orig_folder + "/" + folder_to_check, "r")
    files_list = zf.namelist()
    for file in files_list:
        if file == "France/":
            pass
        elif file[:-4] + "_QC.txt" in existing_files:
            print("already done")
        else:
            try:
                d = zf.open(file, mode="r")
                qc = ex.read_intense(d, only_metadata=False, opened=True)
                qc = get_flags(qc)

            except:
                error_file.write(file + "\n")
    error_file.close()
    zf.close()


def find_files_to_process(folders_to_check):
    files_to_process = []
    file_folders = []
    for folderToCheck in folders_to_check:

        # Check for existence of output folder - make if need be
        if not os.path.exists(qc_folder + "/" + folderToCheck[:-4] + "/Flags"):
            os.makedirs(qc_folder + "/" + folderToCheck[:-4] + "/Flags")
        existing_files = os.listdir(qc_folder + "/" + folderToCheck[:-4] + "/Flags")

        # Get list of raw (formatted) files to process
        zf = zipfile.ZipFile(orig_folder + "/" + folderToCheck, "r")
        files_list = zf.namelist()
        for file in files_list:

            if file[:-4] + "_QC.txt" in existing_files:
                pass
            else:
                files_to_process.append(file)
                file_folders.append(folderToCheck)
        zf.close()
    return files_to_process, file_folders


def process_file(file, q=None):
    # work out file index with a counter and pass as argument

    folder_to_check = file_folders[files_to_process.index(file)]
    zf = zipfile.ZipFile(orig_folder + "/" + folder_to_check, "r")

    d = zf.open(file, mode="r")

    try:
        qc = ex.read_intense(d, only_metadata=False, opened=True)
        print(file)
        successful_read = True
    except:
        print(file, "- read failed")
        successful_read = False

    if successful_read:
        qc = get_flags(qc)

        # for global run
        ex.Series.write_qc(qc, qc_folder + "/" + folder_to_check[:-4] + "/Flags")

    d.close()
    zf.close()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# prepare ETCCDI variables

etccdi_data = {"GHCNDEX": {}, "HADEX2": {}}
etccdi_data_folder = '/media/nas/x21971/ETCCDI_02/'
etccdi_indices = ['CWD', 'CDD', 'R99p', 'PRCPTOT', 'SDII', 'Rx1day']
periods = {"GHCNDEX": '1951-2018', "HADEX2": '1951-2010'}
aggregations = {}
for index in etccdi_indices:
    aggregations[index] = 'max'
aggregations['SDII'] = 'mean'
for data_source in etccdi_data.keys():
    for index in etccdi_indices:
        etccdi_data_path = (etccdi_data_folder + 'RawData_' + data_source +
                            '_' + index + '_' + periods[data_source] +
                            '_ANN_from-90to90_from-180to180.nc')
        etccdi_data[data_source][index] = prep_etccdi_variable(etccdi_data_path,
                                                               index, aggregations[index], data_source)

# create kd tree of monthly gauges ++++++++++++++++++++++++++++++++++++++

# THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
THIS_FOLDER = '/media/nas/x21971/PythonLessons/Python_3'
my_file = os.path.join(THIS_FOLDER, 'statlex_monthly.dat')

monthly_info = open(my_file, "r")

monthly_names = []
monthly_dates = []
monthly_coords = []

monthly_info = open(my_file, "r")
monthly_info.readline()

for line in monthly_info:
    line_list = [line[0:10], line[11:54], line[54:62], line[62:73], line[73:79], line[125:135], line[137:147]]
    sid, name, lat, lon, elv, sd, ed = [l.strip() for l in line_list]
    try:
        sd = datetime.datetime.strptime(sd, "%d.%m.%Y")
        ed = datetime.datetime.strptime(ed, "%d.%m.%Y")
    except:
        sd = None
        ed = None

    if elv == "-999":
        elv = 100  # use 100m above seal level as a default elevation
    else:
        elv = float(elv)

    if sd is None or ed is None:
        pass
    else:
        monthly_names.append(sid)
        monthly_dates.append((sd, ed))
        monthly_coords.append((float(lat), float(lon), elv))

converted_monthly_coords = [geodetic_to_ecef(a, b, c) for a, b, c in monthly_coords]
monthly_tree = sp.KDTree(converted_monthly_coords)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# create kd tree of daily gauges ++++++++++++++++++++++++++++++++++++++

THIS_FOLDER = '/media/nas/x21971/PythonLessons/Python_3'
my_file = os.path.join(THIS_FOLDER, 'statlex_daily')
daily_info = open(my_file, "r")

daily_names = []
daily_dates = []
daily_coords = []

daily_info.readline()

for line in daily_info:
    line_list = [line[0:10], line[11:54], line[54:62], line[62:73], line[73:79], line[125:135], line[137:147]]
    sid, name, lat, lon, elv, sd, ed = [l.strip() for l in line_list]
    try:
        sd = datetime.datetime.strptime(sd, "%d.%m.%Y")
        ed = datetime.datetime.strptime(ed, "%d.%m.%Y")
    except:
        sd = None
        ed = None

    if elv == "-999":
        elv = 100  # use 100m above sea level as a default elevation
    else:
        elv = float(elv)

    if sd is None or ed is None:
        pass
    else:
        daily_names.append(sid)
        daily_dates.append((sd, ed))
        daily_coords.append((float(lat), float(lon), elv))

converted_dailyCoords = [geodetic_to_ecef(a, b, c) for a, b, c in daily_coords]
tree = sp.KDTree(converted_dailyCoords)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# create kd tree of hourly gauges ++++++++++++++++++++++++++++++++++++++

THIS_FOLDER = '/media/nas/x21971/PythonLessons/Python_3'
my_file = os.path.join(THIS_FOLDER, 'statlex_hourly_200108.dat')
hourlyn_info = open(my_file, "r")

hourly_n_names = []
hourly_n_dates = []
hourly_n_coords = []
hourly_n_paths = []
converted_hourly_n_coords = []

hourly_n_names_t = []
hourly_n_dates_t = []
hourly_n_coords_t = []
hourly_n_paths_t = []

hourlyn_info.readline()

for line in hourlyn_info:
    sid, lat, lon, sd, ed, elv, hpath = line.rstrip().split(",")

    try:
        sd = datetime.datetime.strptime(sd, "%Y%m%d%H")
        ed = datetime.datetime.strptime(ed, "%Y%m%d%H")
    except:
        sd = None
        ed = None

    if elv.lower() == "na" or elv == "m" or elv == "nam" or elv == "nan":
        elv = 100  # use 100m above sea level as a default elevation
    else:
        if elv.endswith("m"):
            elv = elv[:-1]
        try:
            elv = float(elv)
        except:
            elv = 100

    if sd is None or ed is None:
        pass
    else:

        # Only append if >=3 years of record (no point having potential neighbours
        # without substantial data)
        # - Also ensure that no duplicates arising from e.g. duplicates in Australia1min.zip
        date_diff = ed - sd
        if date_diff.days >= 3 * 365:
            if sid not in hourly_n_names_t:
                hourly_n_names_t.append(sid)
                hourly_n_dates_t.append((sd, ed))
                hourly_n_coords_t.append((float(lat), float(lon), elv))
                hourly_n_paths_t.append(hpath)

converted_hourly_n_coords_t = [geodetic_to_ecef(a, b, c) for a, b, c in hourly_n_coords_t]

for i in range(len(converted_hourly_n_coords_t)):
    addIt = 1
    for j in converted_hourly_n_coords_t[i]:
        if np.isnan(j):
            addIt = 0

    if addIt == 1:
        hourly_n_names.append(hourly_n_names_t[i])
        hourly_n_dates.append(hourly_n_dates_t[i])
        hourly_n_coords.append(hourly_n_coords_t[i])
        hourly_n_paths.append(hourly_n_paths_t[i])
        converted_hourly_n_coords.append(converted_hourly_n_coords_t[i])

hourly_n_tree = sp.KDTree(converted_hourly_n_coords)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Selected upper bound for hourly world record - see:
# http://www.nws.noaa.gov/oh/hdsc/record_precip/record_precip_world.html
# http://www.bom.gov.au/water/designRainfalls/rainfallEvents/worldRecRainfall.shtml
# https://wmo.asu.edu/content/world-meteorological-organization-global-weather-climate-extremes-archive
world_records = {'hourly': 401.0, 'daily': 1825.0}  # mm

orig_folder = "/media/nas/x21971/QualityControlledData"
qc_folder = "/media/nas/x21971/QC_10"
num_processes = 4

folders_to_check = []
for file in os.listdir(orig_folder):
    if file.endswith(".zip"):
        folders_to_check.append(file)

# Multiprocessing by file (gauge)
# - first get lists of files to process and corresponding folder (country)
files_to_process, file_folders = find_files_to_process(folders_to_check)
if __name__ == '__main__':
    pool = Pool(processes=num_processes)
    m = Manager()
    q = m.Queue()
    for file in files_to_process:
        pool.apply_async(process_file, [file, q])
    pool.close()
    pool.join()

    results = []
    while not q.empty():
        try:
            results.append(q.get())
        except:
            pass

# Additional sweep(s) with serial processing
time.sleep(60)
files_to_process, file_folders = find_files_to_process(folders_to_check)
for file in files_to_process:
    process_file(file)
