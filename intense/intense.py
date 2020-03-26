"""
INTENSE QC Component 1 - Series class definition

This component of the INTENSE QC package defines the Series class and multiple 
utility functions required by the other components of the QC package to read and write
rainfall timeseries objects in the standard INTENSE format. 

Required packages: 
    pandas
    numpy
    os
    
Developed by: 
    Elizabeth Lewis, PhD
    SB, RV, others...

Publication to be cited:
    Paper

June 2019
"""

from __future__ import division
import os
import numpy as np
import pandas as pd
import netCDF4
from datetime import datetime

"""
------------------------------- INTENSE Series definition -------------------------------
"""


class Series:
    def __init__(self,
                 station_id,
                 path_to_original_data,
                 latitude,
                 longitude,
                 original_timestep,
                 original_units,
                 new_units,
                 new_timestep,
                 data=None,
                 elevation='NA',
                 country='NA',
                 original_station_number='NA',
                 original_station_name='NA',
                 time_zone='NA',
                 daylight_saving_info='NA',
                 other='',
                 QC_hourly_neighbours=None,
                 QC_hourly_neighbours_dry=None,
                 QC_daily_neighbours=None,
                 QC_daily_neighbours_dry=None,
                 QC_monthly_neighbours=None,
                 QC_world_record=None,
                 QC_Rx1day=None,
                 QC_CWD=None,
                 QC_CDD=None,
                 QC_daily_accumualtions=None,
                 QC_monthly_accumulations=None,
                 QC_streaks=None,
                 QC_percentiles=["NA", "NA"],
                 QC_k_largest=["NA", "NA", "NA"],
                 QC_days_of_week="NA",
                 QC_hours_of_day="NA",
                 QC_intermittency="NA",
                 QC_breakpoint="NA",
                 QC_R99pTOT="NA",
                 QC_PRCPTOT="NA",
                 QC_change_min_value="NA",
                 QC_offset="NA",
                 QC_preQC_affinity_index="NA",
                 QC_preQC_pearson_coefficient="NA",
                 QC_factor_daily="NA",
                 QC_factor_monthly=None
                 ):
        self.station_id = station_id
        self.country = country
        self.elevation = elevation
        self.original_station_number = original_station_number
        self.original_station_name = original_station_name
        self.path_to_original_data = path_to_original_data
        self.latitude = latitude
        self.longitude = longitude
        self.data = data
        self.no_data_value = -999
        self.masked = None
        self.start_datetime = None
        self.end_datetime = None
        self.number_of_records = None
        self.percent_missing_data = None
        self.original_timestep = original_timestep
        self.original_units = original_units
        self.new_units = new_units
        self.time_zone = time_zone
        self.daylight_saving_info = daylight_saving_info
        self.resolution = None
        self.new_timestep = new_timestep
        self.other = other
        self.QC_hourly_neighbours = QC_hourly_neighbours
        self.QC_hourly_neighbours_dry = QC_hourly_neighbours_dry
        self.QC_daily_neighbours = QC_daily_neighbours
        self.QC_daily_neighbours_dry = QC_daily_neighbours_dry
        self.QC_monthly_neighbours = QC_monthly_neighbours
        self.QC_world_record = QC_world_record
        self.QC_Rx1day = QC_Rx1day
        self.QC_CWD = QC_CWD
        self.QC_CDD = QC_CDD
        self.QC_daily_accumualtions = QC_daily_accumualtions
        self.QC_monthly_accumulations = QC_monthly_accumulations
        self.QC_streaks = QC_streaks
        self.QC_percentiles = QC_percentiles
        self.QC_k_largest = QC_k_largest
        self.QC_days_of_week = QC_days_of_week
        self.QC_hours_of_day = QC_hours_of_day
        self.QC_intermittency = QC_intermittency
        self.QC_breakpoint = QC_breakpoint
        self.QC_R99pTOT = QC_R99pTOT
        self.QC_PRCPTOT = QC_PRCPTOT
        self.QC_change_min_value = QC_change_min_value
        self.QC_offset = QC_offset
        self.QC_preQC_affinity_index = QC_preQC_affinity_index
        self.QC_preQC_pearson_coefficient = QC_preQC_pearson_coefficient
        self.QC_factor_daily = QC_factor_daily
        self.QC_factor_monthly = QC_factor_monthly
        if self.data is not None:
            self.get_info()

    def get_info(self):
        self.masked = self.data[self.data != self.no_data_value]
        self.start_datetime = min(self.data.index)
        self.end_datetime = max(self.data.index)
        self.number_of_records = len(self.data)
        self.percent_missing_data = len(self.data[self.data == self.no_data_value]) * 100 / self.number_of_records
        self.resolution = self.data[self.data != self.no_data_value].diff()[
            self.data[self.data != self.no_data_value].diff() > 0].abs().min()

    def write(self, directory):
        if not os.path.exists(directory):
            os.mkdir(directory)
        with open(os.path.join(directory, self.station_id) + '.txt', 'w') as o:
            o.write(
                "Station ID: {self.station_id}\n"
                "Country: {self.country}\n"
                "Original Station Number: {self.original_station_number}\n"
                "Original Station Name: {self.original_station_name}\n"
                "Path to original data: {self.path_to_original_data}\n"
                "Latitude: {self.latitude}\n"
                "Longitude: {self.longitude}\n"
                "Start datetime: {self.start_datetime:%Y%m%d%H}\n"
                "End datetime: {self.end_datetime:%Y%m%d%H}\n"
                "Elevation: {self.elevation}\n"
                "Number of records: {self.number_of_records}\n"
                "Percent missing data: {self.percent_missing_data:.2f}\n"
                "Original Timestep: {self.original_timestep}\n"
                "New Timestep: {self.new_timestep}\n"
                "Original Units: {self.original_units}\n"
                "New Units: {self.new_units}\n"
                "Time Zone: {self.time_zone}\n"
                "Daylight Saving info: {self.daylight_saving_info}\n"
                "No data value: {self.no_data_value}\n"
                "Resolution: {self.resolution:.2f}\n"
                "Other: {self.other}\n".format(self=self))

            try:
                output_data = self.data.vals.values.copy()
            except:
                output_data = self.data.values.copy()
            output_data[np.isnan(output_data)] = -999
            o.writelines([('%f' % v).rstrip('0').rstrip('.') + '\n' for v in output_data])

    def write_qc(self, directory):
        if not os.path.exists(directory):
            os.mkdir(directory)
        with open(os.path.join(directory, self.station_id) + '_QC.txt', 'w') as o:
            o.write(
                "Station ID: {self.station_id}\n"
                "Country: {self.country}\n"
                "Original Station Number: {self.original_station_number}\n"
                "Original Station Name: {self.original_station_name}\n"
                "Path to original data: {self.path_to_original_data}\n"
                "Latitude: {self.latitude}\n"
                "Longitude: {self.longitude}\n"
                "Start datetime: {self.start_datetime:%Y%m%d%H}\n"
                "End datetime: {self.end_datetime:%Y%m%d%H}\n"
                "Elevation: {self.elevation}\n"
                "Number of records: {self.number_of_records}\n"
                "Percent missing data: {self.percent_missing_data:.2f}\n"
                "Original Timestep: {self.original_timestep}\n"
                "New Timestep: {self.new_timestep}\n"
                "Original Units: {self.original_units}\n"
                "New Units: {self.new_units}\n"
                "Time Zone: {self.time_zone}\n"
                "Daylight Saving info: {self.daylight_saving_info}\n"
                "No data value: {self.no_data_value}\n"
                "Resolution: {self.resolution:.2f}\n"
                "Other: {self.other}\n"
                "Years where Q95 equals 0: {self.QC_percentiles[0]}\n"
                "Years where Q99 equals 0: {self.QC_percentiles[1]}\n"
                "Years where k1 equals 0: {self.QC_k_largest[0]}\n"
                "Years where k5 equals 0: {self.QC_k_largest[1]}\n"
                "Years where k10 equals 0: {self.QC_k_largest[2]}\n"
                "Uneven distribution of rain over days of the week: {self.QC_days_of_week}\n"
                "Uneven distribution of rain over hours of the day: {self.QC_hours_of_day}\n"
                "Years with intermittency issues: {self.QC_intermittency}\n"
                "Break point detected: {self.QC_breakpoint}\n"
                "R99pTOT checks: {self.QC_R99pTOT}\n"
                "PRCPTOT checks: {self.QC_PRCPTOT}\n"
                "Years where min value changes: {self.QC_change_min_value}\n"
                "Optimum offset: {self.QC_offset}\n"
                "Pre QC Affinity Index: {self.QC_preQC_affinity_index}\n"
                "Pre QC Pearson coefficient: {self.QC_preQC_pearson_coefficient}\n"
                "Factor against nearest daily gauge: {self.QC_factor_daily}\n".format(self=self))

            empty_series = np.full(len(self.data), self.no_data_value, dtype=int)

            if self.QC_hourly_neighbours is None:
                self.QC_hourly_neighbours = empty_series

            if self.QC_hourly_neighbours_dry is None:
                self.QC_hourly_neighbours_dry = empty_series

            if self.QC_daily_neighbours is None:
                self.QC_daily_neighbours = empty_series

            if self.QC_daily_neighbours_dry is None:
                self.QC_daily_neighbours_dry = np.full(len(self.data), self.no_data_value, dtype=int)

            if self.QC_monthly_neighbours is None:
                self.QC_monthly_neighbours = np.full(len(self.data), self.no_data_value, dtype=int)

            if self.QC_world_record is None:
                self.QC_world_record = empty_series

            if self.QC_Rx1day is None:
                self.QC_Rx1day = empty_series

            if self.QC_CWD is None:
                self.QC_CWD = empty_series

            if self.QC_CDD is None:
                self.QC_CDD = empty_series

            if self.QC_daily_accumualtions is None:
                self.QC_daily_accumualtions = empty_series

            if self.QC_monthly_accumulations is None:
                self.QC_monthly_accumulations = empty_series

            if self.QC_streaks is None:
                self.QC_streaks = empty_series

            if self.QC_factor_monthly is None:
                self.QC_factor_monthly = empty_series

            self.data.fillna(self.no_data_value, inplace=True)
            vals_flags = zip([float(format(v, '.3f')) for v in self.data.values],
                             self.QC_hourly_neighbours,
                             self.QC_hourly_neighbours_dry,
                             self.QC_daily_neighbours,
                             self.QC_daily_neighbours_dry,
                             self.QC_monthly_neighbours,
                             self.QC_world_record,
                             self.QC_Rx1day,
                             self.QC_CWD,
                             self.QC_CDD,
                             self.QC_daily_accumualtions,
                             self.QC_monthly_accumulations,
                             self.QC_streaks,
                             self.QC_factor_monthly)
            print(vals_flags)
            o.writelines(str(a)[1:-1] + "\n" for a in vals_flags)

    def monthly_max(self):
        return self.masked.groupby(pd.TimeGrouper('M')).max()


def try_float(test_val):
    try:
        v = float(test_val)
    except:
        v = np.nan
    return v


def try_int(test_val):
    try:
        v = int(test_val)
    except:
        v = np.nan
    return v


def try_strptime(test_val):
    try:
        v = datetime.strptime(test_val, '%Y%m%d%H')
    except:
        v = np.nan
    return v


def try_list(test_list):
    try:
        v = [try_int(i) for i in test_list[1:-1].split(", ")]
    except:
        v = np.nan
    return v


def read_intense_qc(path, only_metadata=False, opened=False):
    metadata = []
    if not opened:
        try:
            with open(path, 'rb') as f:
                while True:
                    try:
                        key, val = f.readline().strip().split(':', maxsplit=1)
                        key = key.lower()
                        metadata.append((key.strip(), val.strip()))
                    except:
                        key = "other"
                        val = ""
                    if 'factor against nearest daily gauge' in metadata[-1][0].lower():
                        break
                if only_metadata:
                    data = None
                else:
                    data = f.readlines()
        except:
            with open(path, 'r') as f:
                while True:
                    try:
                        key, val = f.readline().strip().split(':', maxsplit=1)
                        key = key.lower()
                        metadata.append((key.strip(), val.strip()))
                    except:
                        key = "other"
                        val = ""
                    if 'factor against nearest daily gauge' in metadata[-1][0].lower():
                        break
                if only_metadata:
                    data = None
                else:
                    data = f.readlines()

    else:
        f = path
        while True:
            try:
                key, val = str(f.readline().strip())[2:-1].split(':', maxsplit=1)
                key = key.lower()
                metadata.append((key.strip(), val.strip()))
            except:
                key = "other"
                val = ""
            if 'factor against nearest daily gauge' in metadata[-1][0].lower():
                break
        if only_metadata:
            data = None
        else:
            data = f.readlines()
    metadata = dict(metadata)

    for variable in ['country', 'elevation', 'time zone', 'daylight saving info', 'original station name',
                     'original station number']:
        if variable not in metadata.keys():
            metadata[variable] = 'NA'
    if data is not None:
        try:
            data = [i.rstrip().split(", ") for i in data]
        except:
            # working on files written from linux (DWD server), it seems to work 
            # without specifying "utf-8" as argument for decode...
            data = [i.rstrip().decode().split(", ") for i in data]

        data = np.array(data)
        data = pd.DataFrame(data, pd.date_range(start=datetime.strptime(metadata['start datetime'], '%Y%m%d%H'),
                                                end=datetime.strptime(metadata['end datetime'], '%Y%m%d%H'),
                                                freq=metadata['new timestep'][:-2] + 'H'), dtype=float,
                            columns=["vals", "QC_hourly_neighbours", "QC_hourly_neighbours_dry", "QC_daily_neighbours",
                                     "QC_daily_neighbours_dry", "QC_monthly_neighbours", "QC_world_record", "QC_Rx1day",
                                     "QC_CWD", "QC_CDD", "QC_daily_accumualtions", "QC_monthly_accumulations",
                                     "QC_streaks", "QC_factor_monthly"])

        data = data.where(data != -999)

    s = Series(station_id=metadata['station id'],
               path_to_original_data=metadata['path to original data'],
               latitude=try_float(metadata['latitude']),
               longitude=try_float(metadata['longitude']),
               original_timestep=metadata['original timestep'],
               original_units=metadata['original units'],
               new_units=metadata['new units'],
               new_timestep=metadata['new timestep'],
               data=data,
               elevation=metadata['elevation'],
               country=metadata['country'],
               original_station_number=metadata['original station number'],
               original_station_name=metadata['original station name'],
               time_zone=metadata['time zone'])

    s.number_of_records = int(metadata['number of records'])
    s.percent_missing_data = try_float(metadata['percent missing data'])
    s.resolution = try_float(metadata['resolution'])
    s.start_datetime = try_strptime(metadata['start datetime'])
    s.end_datetime = try_strptime(metadata['end datetime'])
    s.QC_percentiles = [try_list(metadata['years where q95 equals 0']), try_list(metadata['years where q99 equals 0'])]
    s.QC_k_largest = [try_list(metadata['years where k1 equals 0']), try_list(metadata['years where k5 equals 0']),
                      try_list(metadata['years where k10 equals 0'])]
    s.QC_days_of_week = try_int(metadata['uneven distribution of rain over days of the week'])
    s.QC_hours_of_day = try_int(metadata['uneven distribution of rain over hours of the day'])
    s.QC_intermittency = try_list(metadata['years with intermittency issues'])
    s.QC_breakpoint = try_int(metadata['break point detected'])
    s.QC_R99pTOT = try_list(metadata['r99ptot checks'])
    s.QC_PRCPTOT = try_list(metadata['prcptot checks'])

    tmp = metadata['years where min value changes']
    change_flag = try_int(tmp.split(", ")[0][1:])
    if change_flag == 0:
        change_list = [np.nan]
    elif change_flag == 1:
        years = tmp[5:-2]
        years = years.split(", ")
        change_list = [int(y) for y in years]
    s.QC_change_min_value = [change_flag, change_list]

    s.QC_offset = try_int(metadata['optimum offset'])
    s.QC_preQC_affinity_index = try_float(metadata['pre qc affinity index'])
    s.QC_preQC_pearson_coefficient = try_float(metadata['pre qc pearson coefficient'])
    s.QC_factor_daily = try_float(metadata['factor against nearest daily gauge'])
    return s


def read_intense(path, only_metadata=False, opened=False):
    metadata = []
    if not opened:
        try:
            with open(path, 'rb') as f:
                while True:
                    key, val = f.readline().strip().split(':', maxsplit=1)
                    key = key.lower()
                    metadata.append((key.strip(), val.strip()))
                    if 'other' in metadata[-1][0].lower():
                        break
                if only_metadata:
                    data = None
                else:
                    data = f.readlines()
        except:
            with open(path, 'r') as f:
                while True:
                    key, val = f.readline().strip().split(':', maxsplit=1)
                    key = key.lower()
                    metadata.append((key.strip(), val.strip()))
                    if 'other' in metadata[-1][0].lower():
                        break
                if only_metadata:
                    data = None
                else:
                    data = f.readlines()

    else:
        f = path
        while True:
            key, val = str(f.readline().strip())[2:-1].split(':', maxsplit=1)
            key = key.lower()
            metadata.append((key.strip(), val.strip()))
            if 'other' in metadata[-1][0].lower():
                break
        if only_metadata:
            data = None
        else:
            data = f.readlines()
    metadata = dict(metadata)

    for variable in ['country', 'elevation', 'time zone', 'daylight saving info', 'original station name',
                     'original station number']:
        if variable not in metadata.keys():
            metadata[variable] = 'NA'
    if data is not None:
        try:  # Exception built for when a nan is added to end of data after applying rulebase
            data = pd.Series(data,
                             pd.date_range(start=datetime.strptime(metadata['start datetime'], '%Y%m%d%H'),
                                           end=datetime.strptime(metadata['end datetime'], '%Y%m%d%H'),
                                           freq=metadata['new timestep'][:-2] + 'H'),
                             dtype=float)
        except:  # Modification adds extra hour at end of series to accomodate nan value
            # Drop nan alternative: (keeps all series same length)
            data = pd.Series(data[:-1],
                             pd.date_range(start=datetime.strptime(metadata['start datetime'], '%Y%m%d%H'),
                                           end=datetime.strptime(metadata['end datetime'], '%Y%m%d%H'),
                                           freq=metadata['new timestep'][:-2] + 'H'),
                             dtype=float)
        data = data.where(data >= 0)

    s = Series(station_id=metadata['station id'],
               path_to_original_data=metadata['path to original data'],
               latitude=try_float(metadata['latitude']),
               longitude=try_float(metadata['longitude']),
               original_timestep=metadata['original timestep'],
               original_units=metadata['original units'],
               new_units=metadata['new units'],
               new_timestep=metadata['new timestep'],
               data=data,
               elevation=metadata['elevation'],
               country=metadata['country'],
               original_station_number=metadata['original station number'],
               original_station_name=metadata['original station name'],
               time_zone=metadata['time zone'])
    try:
        s.number_of_records = int(metadata['number of records'])
    except:
        s.number_of_records = int(metadata['number of hours'])
    s.percent_missing_data = try_float(metadata['percent missing data'])
    s.resolution = try_float(metadata['resolution'])
    s.start_datetime = try_strptime(metadata['start datetime'])
    s.end_datetime = try_strptime(metadata['end datetime'])
    return s


def convert_isd(in_path, out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    f = netCDF4.Dataset(in_path)
    time = f.variables['time'][:]
    precip = f.variables['precip1_depth'][:]
    periods = f.variables['precip1_period'][:]
    if isinstance(periods, np.ma.MaskedArray):
        precip = precip.data
        periods = periods.data

    for period in np.unique(periods)[np.unique(periods) > 0]:
        mask = np.logical_and(periods == period, precip >= 0)

        if len(precip[mask]) > 1:
            times = netCDF4.num2date(time[mask], f.variables['time'].units)
            datetimes = pd.date_range(start=min(times), end=max(times), freq=str(period) + 'H')

            data = pd.Series(precip[mask],
                             index=times)
            data = data.reindex(datetimes)
            data = data[data.first_valid_index():data.last_valid_index()]
            data[pd.isnull(data)] = -999

            series = Series(station_id='ISD_%s' % f.station_id,
                            path_to_original_data=in_path,
                            latitude=f.latitude,
                            longitude=f.longitude,
                            original_timestep='%shr' % period,
                            original_units='mm',
                            new_units='mm',
                            new_timestep='%shr' % period,
                            data=data,
                            elevation='%sm' % f.elevation,
                            original_station_number=f.station_id,
                            time_zone='UTC')
            path = os.path.join(out_path, '%shr' % period)
            if not os.path.exists(path):
                os.mkdir(path)
            series.write(path)
