from setuptools import setup

setup(
    name='intense',
    version='0.3.0.dev1',
    description='',
    long_description='',
    url='https://www.github.com/nclwater/intense-qc',
    author='Elizabeth Lewis, David Pritchard, Roberto Villalobos-Herrera, Fergus McClean',
    author_email='fergus.mcclean@newcastle.ac.uk',
    packages=['intense'],
    install_requires=['pandas>=1', 'rpy2>=2.9', 'xarray>=0.15', 'scipy>=1.4', 'netCDF4>=1.5'],
    classifiers=['Development Status :: 4 - Beta'],
)
