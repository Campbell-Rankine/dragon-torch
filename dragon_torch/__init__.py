"""dragon-torch: PyTorch deep learning research framework, focused on RL."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('dragon-torch')
except PackageNotFoundError:  # running from a source checkout that isn't installed
    __version__ = '0.0.0'
