"""Top-level package for Tinto."""

from importlib.metadata import metadata

from tinto.experiments import Experiment, Iterator, ProductIterator

infos = metadata(__name__)
__version__ = infos["Version"]
__author__ = "Fabien Mathieu"
__email__ = "fabien.mathieu@normalesup.org"

__all__ = ["Experiment", "Iterator", "ProductIterator"]
