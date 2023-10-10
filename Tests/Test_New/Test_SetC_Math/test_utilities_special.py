import warnings
from math import isclose
from scipy.special import factorial2

import pytest
import numpy as np

import TidalPy
TidalPy.test_mode()

from TidalPy.utilities.math.special_x import double_factorial_

@pytest.mark.parametrize('l', (0, 2, 3, 4, 5, 10, 20, 100))
def test_double_factorial(l):

    tpy_value = double_factorial_(l)
    ref_value = factorial2(l)

    assert isclose(tpy_value, ref_value)
