import numpy as np
from scipy.constants import Stefan_Boltzmann as sbc

from ..utilities.performance.numba import njit
from ..utilities.types import FloatArray

INNER_EDGE_TEMP = 273.15
OUTER_EDGE_TEMP = 373.15
INNER_EDGE_TEMP_4 = INNER_EDGE_TEMP**4
OUTER_EDGE_TEMP_4 = OUTER_EDGE_TEMP**4
