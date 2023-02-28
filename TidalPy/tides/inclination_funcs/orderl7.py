""" Inclination functions (squared) for tidal order-l = 7. These are exact (no truncation on I)
"""

from typing import TYPE_CHECKING

import numpy as np

from . import InclinOutput
from ...utilities.performance.numba import njit

if TYPE_CHECKING:
    from ...utilities.types import FloatArray


@njit(cacheable=True, parallel=True)
def calc_inclination_off(inclination: 'FloatArray') -> 'InclinOutput':
    """Calculate F^2_lmp (assuming I=0) for l = 7"""

    # Inclination Functions Calculated for l = 7, Inclination == off.
    ones_ = np.ones_like(inclination)

    inclination_results = {
        (1, 3): 4.78515625 * ones_,
        (3, 2): 13953.515625 * ones_,
        (5, 1): 27014006.25 * ones_,
        (7, 0): 18261468225. * ones_,
        }

    return inclination_results


@njit(cacheable=True, parallel=True)
def calc_inclination(inclination: 'FloatArray') -> 'InclinOutput':
    """Calculate F^2_lmp for l = 7"""

    # Inclination Functions Calculated for l = 7.
    # Optimizations
    i = inclination
    i_half = i / 2.
    i_double = 2. * i
    i_triple = 3. * i
    sin_i = np.sin(i)
    cos_i = np.cos(i)
    sin_i_half = np.sin(i_half)
    cos_i_half = np.cos(i_half)
    cos_i_double = np.cos(i_double)
    cos_i_triple = np.cos(i_triple)

    inclination_results = {
        (0, 0) : 718.91015625*sin_i_half**14*cos_i_half**14,
        (0, 1) : 35226.59765625*(-sin_i_half**4 + sin_i_half**2 - 0.2307692307692307692307692)**2*sin_i_half**10*cos_i_half**10,
        (0, 2) : 401876.75390625*(-0.6397515527950310559006211*sin_i_half**8 + sin_i_half**6 - 0.3913043478260869565217391*sin_i_half**4 - 0.248447204968944099378882*cos_i_half**8 + 0.2173913043478260869565217*cos_i_half**6)**2*sin_i_half**6*cos_i_half**6,
        (0, 3) : 146545.41015625*(-0.00004464285714285714285714286*(cos_i + 1.0)**6*sin_i - 0.005714285714285714285714286*sin_i_half**13*cos_i_half + 0.12*sin_i_half**11*cos_i_half**3 - 0.6*sin_i_half**9*cos_i_half**5 + sin_i_half**7*cos_i_half**7 - 0.6*sin_i_half**5*cos_i_half**9 + 0.12*sin_i_half**3*cos_i_half**11)**2,
        (0, 4) : 146545.41015625*(0.00004464285714285714285714286*(cos_i + 1.0)**6*sin_i + 0.005714285714285714285714286*sin_i_half**13*cos_i_half - 0.12*sin_i_half**11*cos_i_half**3 + 0.6*sin_i_half**9*cos_i_half**5 - sin_i_half**7*cos_i_half**7 + 0.6*sin_i_half**5*cos_i_half**9 - 0.12*sin_i_half**3*cos_i_half**11)**2,
        (0, 5) : 401876.75390625*(0.6397515527950310559006211*sin_i_half**8 - sin_i_half**6 + 0.3913043478260869565217391*sin_i_half**4 + 0.248447204968944099378882*cos_i_half**8 - 0.2173913043478260869565217*cos_i_half**6)**2*sin_i_half**6*cos_i_half**6,
        (0, 6) : 35226.59765625*(sin_i_half**4 - sin_i_half**2 + 0.2307692307692307692307692)**2*sin_i_half**10*cos_i_half**10,
        (0, 7) : 718.91015625*sin_i_half**14*cos_i_half**14,
        (1, 0) : 35226.59765625*sin_i_half**12*cos_i_half**16,
        (1, 1) : 105.3529837131500244140625*(0.7142857142857142857142857*sin_i**2 - cos_i**3 + 0.3406593406593406593406593*cos_i - 0.6593406593406593406593407)**2*sin_i**8,
        (1, 2) : 21441530.25*(-0.702380952380952380952381*sin_i_half**8 + sin_i_half**6 - 0.3571428571428571428571429*sin_i_half**4 - 0.1488095238095238095238095*cos_i_half**8 + 0.1360544217687074829931973*cos_i_half**6)**2*sin_i_half**4*cos_i_half**8,
        (1, 3) : 70927978.515625*(-0.3672727272727272727272727*sin_i_half**14 + sin_i_half**12 - 0.9054545454545454545454545*sin_i_half**10 + 0.2727272727272727272727273*sin_i_half**8 - 0.2909090909090909090909091*sin_i_half**6*cos_i_half**8 + 0.1090909090909090909090909*sin_i_half**4*cos_i_half**10 + 0.01272727272727272727272727*cos_i_half**14 - 0.01246753246753246753246753*cos_i_half**12)**2,
        (1, 4) : 6002500.0*(-0.41875*sin_i_half**12 + 0.7928571428571428571428571*sin_i_half**10 - 0.375*sin_i_half**8 + sin_i_half**6*cos_i_half**6 - 0.9375*sin_i_half**4*cos_i_half**8 - 0.325*cos_i_half**12 + 0.3*cos_i_half**10)**2*sin_i_half**4,
        (1, 5) : 39442325.09765625*(0.3730407523510971786833856*sin_i_half**10 - sin_i_half**8 + 0.8902821316614420062695925*sin_i_half**6 - 0.2633228840125391849529781*sin_i_half**4 - 0.2545454545454545454545455*cos_i_half**10 + 0.2106583072100313479623824*cos_i_half**8)**2*sin_i_half**8,
        (1, 6) : 105.3529837131500244140625*(0.7142857142857142857142857*sin_i**2 + cos_i**3 - 0.3406593406593406593406593*cos_i - 0.6593406593406593406593407)**2*sin_i**8,
        (1, 7) : 35226.59765625*sin_i_half**16*cos_i_half**12,
        (2, 0) : 1268157.515625*sin_i_half**10*cos_i_half**18,
        (2, 1) : 62139718.265625*(sin_i_half**4 - 0.7142857142857142857142857*sin_i_half**2 + 0.1098901098901098901098901)**2*sin_i_half**6*cos_i_half**14,
        (2, 2) : 98456006.25*(0.00009300595238095238095238095*(cos_i + 1.0)**6*sin_i + 0.3*sin_i_half**9*cos_i_half**5 - sin_i_half**7*cos_i_half**7 + 0.8571428571428571428571429*sin_i_half**5*cos_i_half**9 - 0.2142857142857142857142857*sin_i_half**3*cos_i_half**11)**2,
        (2, 3) : 273488906.25*(-0.00005580357142857142857142857*(cos_i + 1.0)**6*sin_i + 0.06666666666666666666666667*sin_i_half**11*cos_i_half**3 - 0.5*sin_i_half**9*cos_i_half**5 + sin_i_half**7*cos_i_half**7 - 0.6666666666666666666666667*sin_i_half**5*cos_i_half**9 + 0.1428571428571428571428571*sin_i_half**3*cos_i_half**11)**2,
        (2, 4) : 595970156.25*(0.5532258064516129032258064*sin_i_half**10 - sin_i_half**8 + 0.4516129032258064516129032*sin_i_half**6 - 0.6774193548387096774193548*sin_i_half**4*cos_i_half**6 - 0.3838709677419354838709677*cos_i_half**10 + 0.3387096774193548387096774*cos_i_half**8)**2*sin_i_half**6*cos_i_half**2,
        (2, 5) : 366196064.0625*(-0.5617283950617283950617284*sin_i_half**8 + sin_i_half**6 - 0.4444444444444444444444444*sin_i_half**4 - 0.6740740740740740740740741*cos_i_half**8 + 0.5185185185185185185185185*cos_i_half**6)**2*sin_i_half**10*cos_i_half**2,
        (2, 6) : 102720758.765625*(-0.7777777777777777777777778*sin_i_half**4 + sin_i_half**2 - 0.3076923076923076923076923)**2*sin_i_half**14*cos_i_half**6,
        (2, 7) : 1268157.515625*sin_i_half**18*cos_i_half**10,
        (3, 0) : 31703937.890625*sin_i_half**8*cos_i_half**20,
        (3, 1) : 283131.2933698296546936035*(cos_i + 1.0)**8*(-cos_i + 0.5373608903020667726550079*cos_i_double - 0.1446740858505564387917329*cos_i_triple + 0.607313195548489666136725)**2,
        (3, 2) : 14517237656.25*(0.9411764705882352941176471*sin_i_half**8 - sin_i_half**6 + 0.2647058823529411764705882*sin_i_half**4 + 0.04019607843137254901960784*cos_i_half**8 - 0.03921568627450980392156863*cos_i_half**6)**2*cos_i_half**12,
        (3, 3) : 53603825625.0*(0.702380952380952380952381*sin_i_half**8 - sin_i_half**6 + 0.3571428571428571428571429*sin_i_half**4 + 0.1488095238095238095238095*cos_i_half**8 - 0.1360544217687074829931973*cos_i_half**6)**2*sin_i_half**4*cos_i_half**8,
        (3, 4) : 98605812744.140625*(-0.3730407523510971786833856*sin_i_half**10 + sin_i_half**8 - 0.8902821316614420062695925*sin_i_half**6 + 0.2633228840125391849529781*sin_i_half**4 + 0.2545454545454545454545455*cos_i_half**10 - 0.2106583072100313479623824*cos_i_half**8)**2*sin_i_half**8,
        (3, 5) : 6643268789.0625*(0.4507246376811594202898551*sin_i_half**8 - 0.8405797101449275362318841*sin_i_half**6 + 0.3913043478260869565217391*sin_i_half**4 + cos_i_half**8 - 0.6956521739130434782608696*cos_i_half**6)**2*sin_i_half**12,
        (3, 6) : 30496280.4931640625*(cos_i + 1.0)**2*(-0.8921568627450980392156863*sin_i**2 + 0.7647058823529411764705882*cos_i + 1)**2*sin_i_half**16,
        (3, 7) : 31703937.890625*sin_i_half**20*cos_i_half**8,
        (4, 0) : 507263006.25*sin_i_half**6*cos_i_half**22,
        (4, 1) : 9079707656.25*(-0.0004261363636363636363636364*(cos_i + 1.0)**6*sin_i - sin_i_half**5*cos_i_half**9 + 0.6*sin_i_half**3*cos_i_half**11)**2,
        (4, 2) : 54703362656.25*(0.0001736111111111111111111111*(cos_i + 1.0)**6*sin_i - 0.6666666666666666666666667*sin_i_half**7*cos_i_half**7 + sin_i_half**5*cos_i_half**9 - 0.3333333333333333333333333*sin_i_half**3*cos_i_half**11)**2,
        (4, 3) : 780704780625.0*(-0.8431372549019607843137255*sin_i_half**6 + sin_i_half**4 - 0.2941176470588235294117647*sin_i_half**2 + 0.04901960784313725490196078*cos_i_half**6)**2*sin_i_half**6*cos_i_half**10,
        (4, 4) : 975205625625.0*(-0.6754385964912280701754386*sin_i_half**6 + sin_i_half**4 - 0.368421052631578947368421*sin_i_half**2 + 0.122807017543859649122807*cos_i_half**6)**2*sin_i_half**10*cos_i_half**6,
        (4, 5) : 297829418906.25*(-0.5809523809523809523809524*sin_i_half**6 + sin_i_half**4 - 0.4285714285714285714285714*sin_i_half**2 + 0.2857142857142857142857143*cos_i_half**6)**2*sin_i_half**14*cos_i_half**2,
        (4, 6) : 61378823756.25*(0.6363636363636363636363636*sin_i_half**4 - sin_i_half**2 + 0.3846153846153846153846154)**2*sin_i_half**18*cos_i_half**2,
        (4, 7) : 507263006.25*sin_i_half**22*cos_i_half**6,
        (5, 0) : 4565367056.25*sin_i_half**4*cos_i_half**24,
        (5, 1) : 29605926.28326416015625*(cos_i + 1.0)**10*(0.6791044776119402985074627*sin_i**2 + 0.9701492537313432835820895*cos_i - 1)**2,
        (5, 2) : 366938156.2072992324829102*(cos_i + 1.0)**8*(cos_i - 0.5373608903020667726550079*cos_i_double + 0.1446740858505564387917329*cos_i_triple - 0.607313195548489666136725)**2,
        (5, 3) : 341343667.2306060791015625*(0.7142857142857142857142857*sin_i**2 - cos_i**3 + 0.3406593406593406593406593*cos_i - 0.6593406593406593406593407)**2*sin_i**8,
        (5, 4) : 341343667.2306060791015625*(0.7142857142857142857142857*sin_i**2 + cos_i**3 - 0.3406593406593406593406593*cos_i - 0.6593406593406593406593407)**2*sin_i**8,
        (5, 5) : 39523179519.140625*(cos_i + 1.0)**2*(-0.8921568627450980392156863*sin_i**2 + 0.7647058823529411764705882*cos_i + 1)**2*sin_i_half**16,
        (5, 6) : 30316468514.0625*(0.6791044776119402985074627*sin_i**2 - 0.9701492537313432835820895*cos_i - 1)**2*sin_i_half**20,
        (5, 7) : 4565367056.25*sin_i_half**24*cos_i_half**4,
        (6, 0) : 1114591.56646728515625*(cos_i + 1.0)**12*sin_i**2,
        (6, 1) : 657412856100.0*(-0.001302083333333333333333333*(cos_i + 1.0)**6*sin_i + sin_i_half**3*cos_i_half**11)**2,
        (6, 2) : 2013326871806.25*(0.4285714285714285714285714 - cos_i)**2*sin_i_half**6*cos_i_half**18,
        (6, 3) : 5592574643906.25*(0.1428571428571428571428571 - cos_i)**2*sin_i_half**10*cos_i_half**14,
        (6, 4) : 22370298575625.0*(0.4285714285714285714285714 - cos_i_half**2)**2*sin_i_half**14*cos_i_half**10,
        (6, 5) : 8053307487225.0*(0.2857142857142857142857143 - cos_i_half**2)**2*sin_i_half**18*cos_i_half**6,
        (6, 6) : 54614986.75689697265625*(cos_i - 1.0)**10*(cos_i + 0.7142857142857142857142857)**2*sin_i**2,
        (6, 7) : 18261468225.0*sin_i_half**26*cos_i_half**2,
        (7, 0) : 18261468225.0*cos_i_half**28,
        (7, 1) : 894811943025.0*sin_i_half**4*cos_i_half**24,
        (7, 2) : 8053307487225.0*sin_i_half**8*cos_i_half**20,
        (7, 3) : 22370298575625.0*sin_i_half**12*cos_i_half**16,
        (7, 4) : 22370298575625.0*sin_i_half**16*cos_i_half**12,
        (7, 5) : 8053307487225.0*sin_i_half**20*cos_i_half**8,
        (7, 6) : 894811943025.0*sin_i_half**24*cos_i_half**4,
        (7, 7) : 18261468225.0*sin_i_half**28
    }

    return inclination_results
