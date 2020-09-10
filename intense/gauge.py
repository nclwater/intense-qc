from __future__ import division
import os
import numpy as np
import pandas as pd
from . import utils
from datetime import datetime
from typing import IO, Union


class Gauge:
    """
    Args:
        station_id: The ID of the station
        path_to_original_data: path to the file which held the data in its original format
        latitude: the latitude of the station
        longitude: the longitude of the station
        original_timestep: the time between each original observation
        original_units: the units of the original observations
        new_units: the units that the original observations were converted into
        new_timestep: the time between each resampled observation
        data: a time series of rainfall depth observations
        elevation: the elevation of the station
        country: the country code of the station,
        original_station_number: the ID number from the original observations
        original_station_name: the original name of the station
        time_zone: the time zone that the observations were recorded in
        daylight_saving_info: if the observations include daylight savings
        other: any other relevant details

    Attributes:
        station_id (str): The ID of the station
        country (str): the country code of the station
        elevation (str): the elevation of the station
        original_station_number (str): the ID number from the original observations
        original_station_name (str): the original name of the station
        path_to_original_data (str): path to the file which held the data in its original format
        latitude (float): the latitude of the station
        longitude (float): the longitude of the station
        data (pd.Series): a time series of rainfall depth observations
        no_data_value (int): the value used to represent missing data
        masked: a subset of data where values are not missing
        start_datetime: the time at which the first observation was recorded
        end_datetime: the time at which the most recent obseration was recorded
        number_of_records: the number of records (including missing data)
        percent_missing_data: the percentage of all records that are missing
        original_timestep: the time between each original observation
        original_units: the units of the original observations
        new_units: the units that the original observations were converted into
        time_zone: the time zone that the observations were recorded in
        daylight_saving_info: if the observations include daylight savings
        resolution: the pandas offset alias describing the frequency of observations
        new_timestep: the time between each resampled observation
        other: any other relevant details

    """
    def __init__(self,
                 station_id: str,
                 path_to_original_data: str,
                 latitude: float,
                 longitude: float,
                 original_timestep: str,
                 original_units: str,
                 new_units: str,
                 new_timestep: str,
                 data: pd.Series = None,
                 elevation: str = 'NA',
                 country: str = 'NA',
                 original_station_number: str = 'NA' ,
                 original_station_name: str = 'NA',
                 time_zone: str = 'NA',
                 daylight_saving_info: str = 'NA',
                 other: str = ''
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
        if self.data is not None:
            self.data.loc[(self.data == self.no_data_value) | (self.data < 0)] = np.nan
            self.get_info()

    def get_info(self):
        """Updates masked, start and end times, number of records, percent missing and resolution based on data"""
        self.masked = self.data[np.isfinite(self.data)]
        self.start_datetime = min(self.data.index)
        self.end_datetime = max(self.data.index)
        self.number_of_records = len(self.data)
        self.percent_missing_data = len(self.data[np.isnan(self.data)]) * 100 / self.number_of_records
        self.resolution = self.data[self.data != self.no_data_value].diff()[
            self.data[self.data != self.no_data_value].diff() > 0].abs().min()

    def write(self, directory: str):
        """Writes data and metadata to a file named using the station ID

        Args:
            directory: the directory in which to create the text file
        """

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

    def monthly_max(self) -> pd.Series:
        """Returns the monthly maximum value from masked"""
        return self.masked.groupby(pd.TimeGrouper('M')).max()


def read_intense(path_or_stream: Union[str, IO], only_metadata: bool = False) -> Gauge:
    """Reads data and metadata from a text file

    Args:
        path_or_stream: a path of the file to read from or an I/O stream
        only_metadata: if True then stop reading after metadata

    Returns:
        An intense.Gauge object
    """
    metadata = []
    if type(path_or_stream) == str:
        try:
            with open(path_or_stream, 'rb') as f:
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
            with open(path_or_stream, 'r') as f:
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
        f = path_or_stream
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

    gauge = Gauge(station_id=metadata['station id'],
                  path_to_original_data=metadata['path to original data'],
                  latitude=utils.try_float(metadata['latitude']),
                  longitude=utils.try_float(metadata['longitude']),
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
        gauge.number_of_records = int(metadata['number of records'])
    except:
        gauge.number_of_records = int(metadata['number of hours'])
    gauge.percent_missing_data = utils.try_float(metadata['percent missing data'])
    gauge.resolution = utils.try_float(metadata['resolution'])
    gauge.start_datetime = utils.try_strptime(metadata['start datetime'])
    gauge.end_datetime = utils.try_strptime(metadata['end datetime'])
    return gauge
