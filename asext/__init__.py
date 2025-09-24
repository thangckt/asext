"""
<img src="./1images/kde_color_128x128.png" style="float: left; margin-right: 20px" width="120" />

Python package extends functions of ASE (Atomic Simulation Environment).

Developed and maintained by [C.Thang Nguyen](https://thangckt.github.io)
"""

# from .check_installation import *
# from .general_utils import *

from pathlib import Path

THKIT_ROOT = Path(__file__).parent

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.1_fallback"

__author__ = "C.Thang Nguyen"
__contact__ = "http://thangckt.github.io/email"
