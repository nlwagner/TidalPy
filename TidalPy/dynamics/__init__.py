from ..exceptions import ImplementationException
from ..performance import njit

from .duel_dissipation import (eccentricity_derivative as eccentricity_derivative_duel,
                               semi_major_axis_derivative as semi_major_axis_derivative_duel)
from .single_dissipation import eccentricity_derivative, semi_major_axis_derivative, spin_rate_derivative

# Minimum tidal mode in [rad s-1]
MODE_ZERO_TOL = 1.e-12

diff_eqs_duel_dissipation = {
    'eccentricity'   : eccentricity_derivative_duel,
    'semi_major_axis': semi_major_axis_derivative_duel,
    # Spin-rate is the same for both duel and single dissipation
    'spin_rate'      : spin_rate_derivative
}

diff_eqs_single_dissipation = {
    'eccentricity'   : eccentricity_derivative,
    'semi_major_axis': semi_major_axis_derivative,
    'spin_rate'      : spin_rate_derivative
}

from .modes_l2 import nsr_modes as nsr_modes_l2
from .modes_l2 import nsr_modes_4 as nsr_modes_l2_t4
from .modes_l2 import nsr_modes_6 as nsr_modes_l2_t6
from .modes_l2 import nsr_modes_8 as nsr_modes_l2_t8
from .modes_l2 import nsr_modes_10 as nsr_modes_l2_t10
from .modes_l2 import nsr_modes_12 as nsr_modes_l2_t12
from .modes_l2 import nsr_modes_16 as nsr_modes_l2_t16
from .modes_l2 import nsr_modes_20 as nsr_modes_l2_t20
from .modes_l2 import spin_sync_modes as spin_sync_l2
from .modes_l2 import spin_sync_modes_4 as spin_sync_l2_t4
from .modes_l2 import spin_sync_modes_6 as spin_sync_l2_t6
from .modes_l2 import spin_sync_modes_8 as spin_sync_l2_t8

from .modes_l3 import nsr_modes as nsr_modes_l3
from .modes_l3 import nsr_modes_4 as nsr_modes_l3_t4
from .modes_l3 import nsr_modes_6 as nsr_modes_l3_t6
from .modes_l3 import spin_sync_modes as spin_sync_l3
from .modes_l3 import spin_sync_modes_4 as spin_sync_l3_t4
from .modes_l3 import spin_sync_modes_6 as spin_sync_l3_t6

max_implemented_order_l = 2

mode_types = {
    # Is NSR or not
    True: {
        # order_l
        2: {
            # Truncation level
            2: nsr_modes_l2,
            4: nsr_modes_l2_t4,
            6: nsr_modes_l2_t6,
            8: nsr_modes_l2_t8,
            10: nsr_modes_l2_t10,
            12: nsr_modes_l2_t12,
            16: nsr_modes_l2_t16,
            20: nsr_modes_l2_t20,
        },
        3: {
            # Truncation level
            2: nsr_modes_l3,
            4: nsr_modes_l3_t4,
            6: nsr_modes_l3_t6
        }
    },
    False: {
        2: {
            2: spin_sync_l2,
            4: spin_sync_l2_t4,
            6: spin_sync_l2_t6,
            8: spin_sync_l2_t8
        },
        3: {
            2: spin_sync_l3,
            4: spin_sync_l3_t4,
            6: spin_sync_l3_t6
        }
    }
}
from functools import partial

from .mode_finder import nsr_mode_finder, sync_mode_finder


def get_mode_finder(order_l: int = 2, truncation_level: int = 2, use_nsr: bool = True):

    try:
        mode_func = mode_types[use_nsr][order_l][truncation_level]
    except KeyError:
        raise ImplementationException('Mode has not been implemented for: '
                                      'NSR={use_nsr}, order-l={order_l}, and truncation level={truncation_level}.')

    if use_nsr:
        mode_finder_func = partial(nsr_mode_finder, nsr_mode_function=mode_func)
    else:
        mode_finder_func = partial(nsr_mode_finder, sync_mode_function=mode_func)

    return mode_finder_func
