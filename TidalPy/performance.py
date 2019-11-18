import numpy as np
from scipy.special import gamma
from .types import FloatArray

from .configurations import use_numba


if use_numba:
    import numba
    from numba.types import UniTuple as nbTuple
    njit = numba.njit
    vectorize = numba.vectorize
    float64 = numba.float64
    int64 = numba.int64

else:
    vectorize = np.vectorize
    float64 = np.float64
    int64 = np.int64

    def njit(func):
        return func

    def nbTuple(*args):
        return None

# Special functions that work with njit
# njit does not support wrapping around scipy's gamma function. Instead we use a lookup table technique. This actually
#     ends up being faster than the gamma function, but it will give incorrect values if you are:
#     1). Outside its defined range
#     2). Taking a factorial of a number that is 0.01 / 2. away from a predefined number.
#     To counter this we add a list of commonly used values of the Andrade alpha parameter, as it is usually what we are
#     taking the factorial of.
_predefined_range = np.linspace(0., 1., 1000)
_common_alphas = np.asarray([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
                             0.9, 0.95])
_predefined_inputs = np.concatenate((_predefined_range, _common_alphas))
_factorials = gamma(_predefined_inputs + 1.0)


@njit
def find_factorial(number: FloatArray) -> FloatArray:
    """ Find's the factorial of a number based on a look-up table. 'number' must be between 0 and 1.

    The lookup table approach allows this function to be njit'd and used in other njit functions.

    Parameters
    ----------
    number : FloatArray
        Number to be factorialized. Must be between 0 and 1

    Returns
    -------
    factorial : FloatArray
        Factorial of 'number'
    """

    index = np.abs(_predefined_inputs - number).argmin()
    factorial = _factorials[index]

    return factorial
