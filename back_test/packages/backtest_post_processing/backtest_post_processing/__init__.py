# this code will setup a version equal to the one present in setup.py
from pkg_resources import get_distribution
__version__ = get_distribution('backtest_post_processing').version


# import here the function that you want to be available to the external world

from .backtest_post_processing_wrapper import generate_tables, generate_plots
