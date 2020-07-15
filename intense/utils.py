import xarray as xr
import numpy.ma as ma
import calendar
import time
import math
import scipy.spatial as sp
import subprocess

from rpy2.robjects import StrVector
from rpy2.robjects.packages import importr

from intense import gauge as ex
import pandas as pd
import numpy as np
import scipy.interpolate
import datetime
import os
import zipfile
import scipy.stats

# Selected upper bound for hourly world record - see:
# http://www.nws.noaa.gov/oh/hdsc/record_precip/record_precip_world.html
# http://www.bom.gov.au/water/designRainfalls/rainfallEvents/worldRecRainfall.shtml
# https://wmo.asu.edu/content/world-meteorological-organization-global-weather-climate-extremes-archive
world_records = {'hourly': 401.0, 'daily': 1825.0}  # mm


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

def get_etccdi_value(etccdi_data, index_name, lon, lat):
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


def get_daily_gpcc(path, start_year, end_year, gpcc_id):
    gpcc_filename = "tw_" + gpcc_id + ".dat"
    dat_path = os.path.join(path, "tw_" + gpcc_id + ".dat")
    zip_path = os.path.join(path, "tw_" + gpcc_id + ".zip")
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

    filename = gsdr_id + '.txt'

    if path.endswith('.zip'):
        f = zipfile.ZipFile(path).open(filename)
    else:
        f = os.path.join(path, filename)

    df = ex.read_intense(f, only_metadata=False).data.to_frame("GSDR")
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
def get_monthly_gpcc(path, start_year, end_year, gpcc_id):  # Hey liz! Check this once you have access to monthly!

    gpcc_filename = "mw_" + gpcc_id + ".dat"
    dat_path = os.path.join(path, "mw_" + gpcc_id + ".dat")
    zip_path = os.path.join(path, "mw_" + gpcc_id + ".zip")
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

    dfr.index = pd.to_datetime(dfr.index) + datetime.timedelta(hours=8)

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
    # needs to be at hour=0800 to reconcile GPCC vs GSDR aggregation definitions
    df.index = pd.to_datetime(df.index) + datetime.timedelta(hours=8)
    return propagate_flags(df.flags)


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

def convert_to_dry_spell(daily_df):
    # dp 29/11/2019 - it would make sense to remove np.around so fractional not binary,
    # but this will require a change to how the flagging is done for it to make sense
    # i think (i.e. do all stations agree the period is wet when the target is dry?)
    # should the threshold for dry be larger than just zero?

    dry_periods = daily_df.copy()
    dry_periods[dry_periods > 0] = -1
    dry_periods += 1

    daily_df["fracDryDays"] = dry_periods.rolling(15, min_periods=15).sum() / 15

    converted_df = daily_df["fracDryDays"]
    converted_df.columns = ["ts1"]
    return converted_df


def propagate_flags(series, days=14):
    series = series.copy()
    # flag preceding periods, prioritising higher flag values
    for flag, series_filtered in [(flag, series[series == flag]) for flag in [1, 2, 3]]:
        for idx, value in series_filtered.iteritems():
            series[idx - datetime.timedelta(days=days):idx] = flag

    return series

# Prepare ETCCDI variables
def read_etccdi_data(etccdi_data_folder):
    etccdi_data = {"GHCNDEX": {}, "HADEX2": {}}
    etccdi_indices = ['CWD', 'CDD', 'R99p', 'PRCPTOT', 'SDII', 'Rx1day']
    periods = {"GHCNDEX": '1951-2018', "HADEX2": '1951-2010'}
    aggregations = {}
    for index in etccdi_indices:
        aggregations[index] = 'max'
    aggregations['SDII'] = 'mean'
    for data_source in etccdi_data.keys():
        for index in etccdi_indices:
            etccdi_data_path = (etccdi_data_folder + '/RawData_' + data_source +
                                '_' + index + '_' + periods[data_source] +
                                '_ANN_from-90to90_from-180to180.nc')
            etccdi_data[data_source][index] = prep_etccdi_variable(etccdi_data_path,
                                                                   index, aggregations[index], data_source)
    return etccdi_data

# create kd tree of monthly gauges ++++++++++++++++++++++++++++++++++++++
def create_kdtree_monthly_data(path):

    with open(path, "r") as monthly_info:

        monthly_names = []
        monthly_dates = []
        monthly_coords = []
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
    
    return monthly_names, monthly_dates, monthly_coords, monthly_tree


# create kd tree of daily gauges ++++++++++++++++++++++++++++++++++++++
def create_kdtree_daily_data(path):
    with open(path, "r") as daily_info:

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

    return daily_names, daily_dates, daily_coords, tree

# create kd tree of hourly gauges ++++++++++++++++++++++++++++++++++++++
def create_kdtree_hourly_data(path):
    with open(path, "r") as hourlyn_info:

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

    return hourly_n_names, hourly_n_dates, hourly_n_coords, hourly_n_paths, hourly_n_tree


def try_float(test_val):
    try:
        v = float(test_val)
    except:
        v = np.nan
    return v


def try_strptime(test_val):
    try:
        v = datetime.datetime.strptime(test_val, '%Y%m%d%H')
    except:
        v = np.nan
    return v


def try_int(test_val):
    try:
        v = int(test_val)
    except:
        v = np.nan
    return v


def try_list(test_list):
    try:
        v = [try_int(i) for i in test_list[1:-1].split(", ")]
    except:
        v = np.nan
    return v


def install_r_package(package_name):
    """Installs """
    utils = importr('utils')
    utils.install_packages(StrVector([package_name]), repos='http://cran.us.r-project.org')