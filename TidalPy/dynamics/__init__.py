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
            20: nsr_modes_l2_t20
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




#
# # Build function lists
# mode_dict = dict()
# for
# nsr_order_l2_dict = dict()
# nsr_order_l2_max_trunc = 0
# for _name, _func in nsr_modes_l2_NEW.__dict__.items():
#     if 'nsr_modes_t' in _name:
#         trunc_lvl = int(_name.split('nsr_modes_t')[1])
#         nsr_order_l2_dict[trunc_lvl] = _func
#         if trunc_lvl > nsr_order_l2_max_trunc:
#             nsr_order_l2_max_trunc = trunc_lvl
# nsr_order_l2_list = list()
# for i in range(nsr_order_l2_max_trunc+1):
#     func_ = nsr_order_l2_dict[i]
#     nsr_order_l2_list.append(func_)
