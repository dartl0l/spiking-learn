from setuptools import setup, find_packages
from os.path import join, dirname

import spiking_network_learning_algorithm


setup(
    name='spilearn',
    version=spiking_network_learning_algorithm.__version__,
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'README.txt')).read(),
)
