# In a mixed package maturin does not modify the
# __init__.py file, so we need to import the 
# Rust module here.
from .rust import Disc, Donor, Brightspot, Whitedwarf
from .rust import findi, findq, findphi

from .cv import CV