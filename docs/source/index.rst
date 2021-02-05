.. INTENSE documentation master file, created by
   sphinx-quickstart on Sun Apr 19 12:13:35 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

INTENSE Documentation
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Quickstart
----------
INTENSE files can be read using the :py:meth:`read_intense <intense.gauge.read_intense>` function:

.. code-block:: python

   from intense.gauge import read_intense
   gauge = read_intense('path/to/intense/data.txt')

This returns a :py:class:`Gauge <intense.gauge.Gauge>` object.

If your file is not in INTENSE format, you will need to manually create the :py:class:`Gauge <intense.gauge.Gauge>` object:

.. code-block:: python

   from intense.gauge import Gauge
   import pandas as pd

   path = 'path/to/data.txt'
   data = pd.read_csv(path, index_col='time', squeeze=True)

   gauge = Gauge(
      station_id='STATION_ID',
      path_to_original_data=path,
      latitude=55,
      longitude=-1,
      original_timestep='1hr',
      original_units='mm',
      new_units='mm',
      new_timestep='1hr',
      data=data,
      elevation='NA',
      country='NA',
      original_station_number='NA',
      original_station_name='NA',
      time_zone='NA',
      daylight_saving_info='NA',
      other='')


To run quality control, create a :py:class:`Qc <intense.qc.Qc>` object using a :py:class:`Gauge <intense.gauge.Gauge>` and call
the :py:meth:`get_flags <intense.qc.Qc.get_flags>` method:

.. code-block:: python

   from intense.qc import Qc
   qc = Qc(gauge, use_hourly_neighbours=False)
   qc.get_flags()

The results can be written to a file. This file will be named with the gauge ID followed by "_QC":

.. code-block:: python

   qc.write('/path/to/output/directory')

To apply the rulebase, call :py:meth:`apply_rulebase <intense.rulebase.apply_rulebase>` on the path to the QC file
created using :py:meth:`Qc.write <intense.qc.Qc.write>`.
This will write a file in the specified output directory containing the QC'd data and named using the station ID.

.. code-block:: python

   from intense.rulebase import apply_rulebase
   apply_rulebase('/path/to/qc/file.txt', 'path/to/output/directory')



intense.gauge module
--------------------
.. automodule:: intense.gauge
   :members:
   :undoc-members:

intense.qc module
-----------------
.. automodule:: intense.qc
   :members:
   :undoc-members:

intense.rulebase module
-----------------------
.. automodule:: intense.rulebase
   :members:
   :undoc-members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
