""" Eccentricity functions (squared) for various truncations of e at tidal order-l = 4
"""

from typing import Tuple, Dict

import numpy as np

from TidalPy.performance import njit

@njit
def eccentricity_funcs_trunc2(eccentricity: np.ndarray) -> Dict[Tuple[int, int, int], np.ndarray]:
    """ Calculates the eccentricity functions (by mode) truncated to e^10 for order-l = 3
    Parameters
    ----------
    eccentricity : np.ndarray
        Orbital Eccentricity
    Returns
    -------
    eccentricity_results_bymode : Dict[Tuple[int, int int], np.ndarray]
    """
    # Eccentricity functions calculated at truncation level 14. Reduced to 2.

    # Performance and readability improvements
    e = eccentricity
    e2  = e * e

    eccentricity_results_bymode = {
        (4, 0, -1): 2.25*e2,
        (4, 0, 0):  -22.0*e2 + 1.0,
        (4, 0, 1):  42.25*e2,
        (4, 1, -1): 0.25*e2,
        (4, 1, 0):  2.0*e2 + 1.0,
        (4, 1, 1):  20.25*e2,
        (4, 2, -1): 6.25*e2,
        (4, 2, 0):  -2.25*(e2 + 0.666666666666667)**2/(e2 - 1.0)**7,
        (4, 2, 1):  6.25*e2,
        (4, 3, -1): 20.25*e2,
        (4, 3, 0):  2.0*e2 + 1.0,
        (4, 3, 1):  0.25*e2,
        (4, 4, -1): 42.25*e2,
        (4, 4, 0):  -22.0*e2 + 1.0,
        (4, 4, 1):  2.25*e2
    }

    return eccentricity_results_bymode


@njit
def eccentricity_funcs_trunc10(eccentricity: np.ndarray) -> Dict[Tuple[int, int, int], np.ndarray]:
    """ Calculates the eccentricity functions (by mode) truncated to e^10 for order-l = 3
    Parameters
    ----------
    eccentricity : np.ndarray
        Orbital Eccentricity
    Returns
    -------
    eccentricity_results_bymode : Dict[Tuple[int, int int], np.ndarray]
    """
    # Eccentricity functions calculated at truncation level 14. Reduced to 10.

    # Performance and readability improvements
    e = eccentricity
    e2  = e * e
    e4  = e * e * e * e
    e6  = e * e * e * e * e * e
    e8  = e**8
    e10 = e**10

    eccentricity_results_bymode = {
        (4, 0, -5): 6.78168402777778e-8*e10,
        (4, 0, -3): 0.000366550021701389*e10 + 0.000379774305555556*e8 + 0.000434027777777778*e6,
        (4, 0, -2): -0.0305555555555556*e10 + 0.111111111111111*e8 - 0.333333333333333*e6 + 0.25*e4,
        (4, 0, -1): 15.6179992675781*e10 - 30.61669921875*e8 + 31.18359375*e6 - 14.0625*e4 + 2.25*e2,
        (4, 0, 0):  -1034.92486111111*e10 + 1030.61024305556*e8 - 583.638888888889*e6 + 170.75*e4 - 22.0*e2 + 1.0,
        (4, 0, 1):  17934.7833421495*e10 - 10497.2230360243*e8 + 3569.95442708333*e6 - 621.5625*e4 + 42.25*e2,
        (4, 0, 2):  -120013.70625*e10 + 42418.125*e8 - 8185.5*e6 + 650.25*e4,
        (4, 0, 3):  359793.938875665*e10 - 71820.7014431424*e8 + 6106.77126736111*e6,
        (4, 0, 4):  -485876.304166667*e10 + 42418.8350694444*e8,
        (4, 0, 5):  239949.195562134*e10,
        (4, 1, -5): 3.66662658691406*e10,
        (4, 1, -4): 11.6203125*e10 + 1.94835069444444*e8,
        (4, 1, -3): 25.5508222791884*e10 + 6.76567925347222*e8 + 1.04210069444444*e6,
        (4, 1, -2): -0.5625*e4/(e2 - 1.0)**7,
        (4, 1, -1): 76.7125413682726*e10 + 29.2154405381944*e8 + 9.11067708333333*e6 + 2.0625*e4 + 0.25*e2,
        (4, 1, 0):  129.900034722222*e10 + 58.2599826388889*e8 + 23.5694444444444*e6 + 9.125*e4 + 2.0*e2 + 1.0,
        (4, 1, 1):  213.248858642578*e10 + 97.09716796875*e8 + 67.74609375*e6 - 1.6875*e4 + 20.25*e2,
        (4, 1, 2):  147.488194444444*e10 + 460.165798611111*e8 - 197.645833333333*e6 + 175.5625*e4,
        (4, 1, 3):  3201.53101433648*e10 - 1962.95241970486*e8 + 1030.67751736111*e6,
        (4, 1, 4):  -12399.046875*e10 + 4812.890625*e8,
        (4, 1, 5):  19310.6487826199*e10,
        (4, 2, -5): 655.906780666775*e10,
        (4, 2, -4): 764.401041666667*e10 + 240.896267361111*e8,
        (4, 2, -3): 966.631546020508*e10 + 333.82568359375*e8 + 82.12890625*e6,
        (4, 2, -2): 1123.84548611111*e10 + 427.777777777778*e8 + 129.166666666667*e6 + 25.0*e4,
        (4, 2, -1): 1233.00030178494*e10 + 494.570583767361*e8 + 166.048177083333*e6 + 42.1875*e4 + 6.25*e2,
        (4, 2, 0):  -2.25*(e2 + 0.666666666666667)**2/(e2 - 1.0)**7,
        (4, 2, 1):  1233.00030178494*e10 + 494.570583767361*e8 + 166.048177083333*e6 + 42.1875*e4 + 6.25*e2,
        (4, 2, 2):  1123.84548611111*e10 + 427.777777777778*e8 + 129.166666666667*e6 + 25.0*e4,
        (4, 2, 3):  966.631546020508*e10 + 333.82568359375*e8 + 82.12890625*e6,
        (4, 2, 4):  764.401041666667*e10 + 240.896267361111*e8,
        (4, 2, 5):  655.906780666775*e10,
        (4, 3, -5): 19310.6487826199*e10,
        (4, 3, -4): -12399.046875*e10 + 4812.890625*e8,
        (4, 3, -3): 3201.53101433648*e10 - 1962.95241970486*e8 + 1030.67751736111*e6,
        (4, 3, -2): 147.488194444444*e10 + 460.165798611111*e8 - 197.645833333333*e6 + 175.5625*e4,
        (4, 3, -1): 213.248858642578*e10 + 97.09716796875*e8 + 67.74609375*e6 - 1.6875*e4 + 20.25*e2,
        (4, 3, 0):  129.900034722222*e10 + 58.2599826388889*e8 + 23.5694444444444*e6 + 9.125*e4 + 2.0*e2 + 1.0,
        (4, 3, 1):  76.7125413682726*e10 + 29.2154405381944*e8 + 9.11067708333333*e6 + 2.0625*e4 + 0.25*e2,
        (4, 3, 2):  -0.5625*e4/(e2 - 1.0)**7,
        (4, 3, 3):  25.5508222791884*e10 + 6.76567925347222*e8 + 1.04210069444444*e6,
        (4, 3, 4):  11.6203125*e10 + 1.94835069444444*e8,
        (4, 3, 5):  3.66662658691406*e10,
        (4, 4, -5): 239949.195562134*e10,
        (4, 4, -4): -485876.304166667*e10 + 42418.8350694444*e8,
        (4, 4, -3): 359793.938875665*e10 - 71820.7014431424*e8 + 6106.77126736111*e6,
        (4, 4, -2): -120013.70625*e10 + 42418.125*e8 - 8185.5*e6 + 650.25*e4,
        (4, 4, -1): 17934.7833421495*e10 - 10497.2230360243*e8 + 3569.95442708333*e6 - 621.5625*e4 + 42.25*e2,
        (4, 4, 0):  -1034.92486111111*e10 + 1030.61024305556*e8 - 583.638888888889*e6 + 170.75*e4 - 22.0*e2 + 1.0,
        (4, 4, 1):  15.6179992675781*e10 - 30.61669921875*e8 + 31.18359375*e6 - 14.0625*e4 + 2.25*e2,
        (4, 4, 2):  -0.0305555555555556*e10 + 0.111111111111111*e8 - 0.333333333333333*e6 + 0.25*e4,
        (4, 4, 3):  0.000366550021701389*e10 + 0.000379774305555556*e8 + 0.000434027777777778*e6,
        (4, 4, 5):  6.78168402777778e-8*e10,
    }

    return eccentricity_results_bymode


@njit
def eccentricity_funcs_trunc12(eccentricity: np.ndarray) -> Dict[Tuple[int, int, int], np.ndarray]:
    """ Calculates the eccentricity functions (by mode) truncated to e^12 for order-l = 3
    Parameters
    ----------
    eccentricity : np.ndarray
        Orbital Eccentricity
    Returns
    -------
    eccentricity_results_bymode : Dict[Tuple[int, int int], np.ndarray]
    """
    # Eccentricity functions calculated at truncation level 14. Reduced to 12.

    # Performance and readability improvements
    e = eccentricity
    e2  = e * e
    e4  = e * e * e * e
    e6  = e * e * e * e * e * e
    e8  = e**8
    e10 = e**10
    e12 = e**12

    eccentricity_results_bymode = {
        (4, 0, -6): 1.92901234567901e-6*e12,
        (4, 0, -5): 1.75193504050926e-7*e12 + 6.78168402777778e-8*e10,
        (4, 0, -3): 0.000341919322072724*e12 + 0.000366550021701389*e10 + 0.000379774305555556*e8 + 0.000434027777777778*e6,
        (4, 0, -2): -0.000998263888888889*e12 - 0.0305555555555556*e10 + 0.111111111111111*e8 - 0.333333333333333*e6 + 0.25*e4,
        (4, 0, -1): -5.07659271240234*e12 + 15.6179992675781*e10 - 30.61669921875*e8 + 31.18359375*e6 - 14.0625*e4 + 2.25*e2,
        (4, 0, 0):  646.549230324074*e12 - 1034.92486111111*e10 + 1030.61024305556*e8 - 583.638888888889*e6 + 170.75*e4 - 22.0*e2 + 1.0,
        (4, 0, 1):  -19313.339785258*e12 + 17934.7833421495*e10 - 10497.2230360243*e8 + 3569.95442708333*e6 - 621.5625*e4 + 42.25*e2,
        (4, 0, 2):  209706.022265625*e12 - 120013.70625*e10 + 42418.125*e8 - 8185.5*e6 + 650.25*e4,
        (4, 0, 3):  -1022671.96280935*e12 + 359793.938875665*e10 - 71820.7014431424*e8 + 6106.77126736111*e6,
        (4, 0, 4):  2428954.85556713*e12 - 485876.304166667*e10 + 42418.8350694444*e8,
        (4, 0, 5):  -2737716.23672287*e12 + 239949.195562134*e10,
        (4, 0, 6):  1168816.25004823*e12,
        (4, 1, -6): 6.92530394000772*e12,
        (4, 1, -5): 19.8938053894043*e12 + 3.66662658691406*e10,
        (4, 1, -4): 41.270521556713*e12 + 11.6203125*e10 + 1.94835069444444*e8,
        (4, 1, -3): 73.2350860124753*e12 + 25.5508222791884*e10 + 6.76567925347222*e8 + 1.04210069444444*e6,
        (4, 1, -2): -0.5625*e4/(e2 - 1.0)**7,
        (4, 1, -1): 175.357567647298*e12 + 76.7125413682726*e10 + 29.2154405381944*e8 + 9.11067708333333*e6 + 2.0625*e4 + 0.25*e2,
        (4, 1, 0):  266.809137008102*e12 + 129.900034722222*e10 + 58.2599826388889*e8 + 23.5694444444444*e6 + 9.125*e4 + 2.0*e2 + 1.0,
        (4, 1, 1):  397.592113952637*e12 + 213.248858642578*e10 + 97.09716796875*e8 + 67.74609375*e6 - 1.6875*e4 + 20.25*e2,
        (4, 1, 2):  644.418093532986*e12 + 147.488194444444*e10 + 460.165798611111*e8 - 197.645833333333*e6 + 175.5625*e4,
        (4, 1, 3):  -1290.54145761184*e12 + 3201.53101433648*e10 - 1962.95241970486*e8 + 1030.67751736111*e6,
        (4, 1, 4):  20109.588046875*e12 - 12399.046875*e10 + 4812.890625*e8,
        (4, 1, 5):  -61669.7520816323*e12 + 19310.6487826199*e10,
        (4, 1, 6):  69524.1394102045*e12,
        (4, 2, -6): 1693.8369140625*e12,
        (4, 2, -5): 1570.4741746408*e12 + 655.906780666775*e10,
        (4, 2, -4): 1995.2357494213*e12 + 764.401041666667*e10 + 240.896267361111*e8,
        (4, 2, -3): 2301.10216140747*e12 + 966.631546020508*e10 + 333.82568359375*e8 + 82.12890625*e6,
        (4, 2, -2): 2545.29351128472*e12 + 1123.84548611111*e10 + 427.777777777778*e8 + 129.166666666667*e6 + 25.0*e4,
        (4, 2, -1): 2711.33776262071*e12 + 1233.00030178494*e10 + 494.570583767361*e8 + 166.048177083333*e6 + 42.1875*e4 + 6.25*e2,
        (4, 2, 0):  -2.25*(e2 + 0.666666666666667)**2/(e2 - 1.0)**7,
        (4, 2, 1):  2711.33776262071*e12 + 1233.00030178494*e10 + 494.570583767361*e8 + 166.048177083333*e6 + 42.1875*e4 + 6.25*e2,
        (4, 2, 2):  2545.29351128472*e12 + 1123.84548611111*e10 + 427.777777777778*e8 + 129.166666666667*e6 + 25.0*e4,
        (4, 2, 3):  2301.10216140747*e12 + 966.631546020508*e10 + 333.82568359375*e8 + 82.12890625*e6,
        (4, 2, 4):  1995.2357494213*e12 + 764.401041666667*e10 + 240.896267361111*e8,
        (4, 2, 5):  1570.4741746408*e12 + 655.906780666775*e10,
        (4, 2, 6):  1693.8369140625*e12,
        (4, 3, -6): 69524.1394102045*e12,
        (4, 3, -5): -61669.7520816323*e12 + 19310.6487826199*e10,
        (4, 3, -4): 20109.588046875*e12 - 12399.046875*e10 + 4812.890625*e8,
        (4, 3, -3): -1290.54145761184*e12 + 3201.53101433648*e10 - 1962.95241970486*e8 + 1030.67751736111*e6,
        (4, 3, -2): 644.418093532986*e12 + 147.488194444444*e10 + 460.165798611111*e8 - 197.645833333333*e6 + 175.5625*e4,
        (4, 3, -1): 397.592113952637*e12 + 213.248858642578*e10 + 97.09716796875*e8 + 67.74609375*e6 - 1.6875*e4 + 20.25*e2,
        (4, 3, 0):  266.809137008102*e12 + 129.900034722222*e10 + 58.2599826388889*e8 + 23.5694444444444*e6 + 9.125*e4 + 2.0*e2 + 1.0,
        (4, 3, 1):  175.357567647298*e12 + 76.7125413682726*e10 + 29.2154405381944*e8 + 9.11067708333333*e6 + 2.0625*e4 + 0.25*e2,
        (4, 3, 2):  -0.5625*e4/(e2 - 1.0)**7,
        (4, 3, 3):  73.2350860124753*e12 + 25.5508222791884*e10 + 6.76567925347222*e8 + 1.04210069444444*e6,
        (4, 3, 4):  41.270521556713*e12 + 11.6203125*e10 + 1.94835069444444*e8,
        (4, 3, 5):  19.8938053894043*e12 + 3.66662658691406*e10,
        (4, 3, 6):  6.92530394000772*e12,
        (4, 4, -6): 1168816.25004823*e12,
        (4, 4, -5): -2737716.23672287*e12 + 239949.195562134*e10,
        (4, 4, -4): 2428954.85556713*e12 - 485876.304166667*e10 + 42418.8350694444*e8,
        (4, 4, -3): -1022671.96280935*e12 + 359793.938875665*e10 - 71820.7014431424*e8 + 6106.77126736111*e6,
        (4, 4, -2): 209706.022265625*e12 - 120013.70625*e10 + 42418.125*e8 - 8185.5*e6 + 650.25*e4,
        (4, 4, -1): -19313.339785258*e12 + 17934.7833421495*e10 - 10497.2230360243*e8 + 3569.95442708333*e6 - 621.5625*e4 + 42.25*e2,
        (4, 4, 0):  646.549230324074*e12 - 1034.92486111111*e10 + 1030.61024305556*e8 - 583.638888888889*e6 + 170.75*e4 - 22.0*e2 + 1.0,
        (4, 4, 1):  -5.07659271240234*e12 + 15.6179992675781*e10 - 30.61669921875*e8 + 31.18359375*e6 - 14.0625*e4 + 2.25*e2,
        (4, 4, 2):  -0.000998263888888889*e12 - 0.0305555555555556*e10 + 0.111111111111111*e8 - 0.333333333333333*e6 + 0.25*e4,
        (4, 4, 3):  0.000341919322072724*e12 + 0.000366550021701389*e10 + 0.000379774305555556*e8 + 0.000434027777777778*e6,
        (4, 4, 5):  1.75193504050926e-7*e12 + 6.78168402777778e-8*e10,
        (4, 4, 6):  1.92901234567901e-6*e12,
    }

    return eccentricity_results_bymode


@njit
def eccentricity_funcs_trunc14(eccentricity: np.ndarray) -> Dict[Tuple[int, int, int], np.ndarray]:
    """ Calculates the eccentricity functions (by mode) truncated to e^14 for order-l = 3
    Parameters
    ----------
    eccentricity : np.ndarray
        Orbital Eccentricity
    Returns
    -------
    eccentricity_results_bymode : Dict[Tuple[int, int int], np.ndarray]
    """
    # Eccentricity functions calculated at truncation level 14.

    # Performance and readability improvements
    e = eccentricity
    e2  = e * e
    e4  = e * e * e * e
    e6  = e * e * e * e * e * e
    e8  = e**8
    e10 = e**10
    e12 = e**12
    e14 = e**14

    eccentricity_results_bymode = {
        (4, 0, -7): 1.14925540223414e-5*e14,
        (4, 0, -6): 5.51146384479718e-6*e14 + 1.92901234567901e-6*e12,
        (4, 0, -5): 2.88440226667562e-7*e14 + 1.75193504050926e-7*e12 + 6.78168402777778e-8*e10,
        (4, 0, -3): 0.000316369310678418*e14 + 0.000341919322072724*e12 + 0.000366550021701389*e10 + 0.000379774305555556*e8 + 0.000434027777777778*e6,
        (4, 0, -2): -0.00308807319223986*e14 - 0.000998263888888889*e12 - 0.0305555555555556*e10 + 0.111111111111111*e8 - 0.333333333333333*e6 + 0.25*e4,
        (4, 0, -1): 1.05155106272016*e14 - 5.07659271240234*e12 + 15.6179992675781*e10 - 30.61669921875*e8 + 31.18359375*e6 - 14.0625*e4 + 2.25*e2,
        (4, 0, 0):  -274.421647415911*e14 + 646.549230324074*e12 - 1034.92486111111*e10 + 1030.61024305556*e8 - 583.638888888889*e6 + 170.75*e4 - 22.0*e2 + 1.0,
        (4, 0, 1):  14027.1271612972*e14 - 19313.339785258*e12 + 17934.7833421495*e10 - 10497.2230360243*e8 + 3569.95442708333*e6 - 621.5625*e4 + 42.25*e2,
        (4, 0, 2):  -244265.454441964*e14 + 209706.022265625*e12 - 120013.70625*e10 + 42418.125*e8 - 8185.5*e6 + 650.25*e4,
        (4, 0, 3):  1864648.11796509*e14 - 1022671.96280935*e12 + 359793.938875665*e10 - 71820.7014431424*e8 + 6106.77126736111*e6,
        (4, 0, 4):  -7073073.58282352*e14 + 2428954.85556713*e12 - 485876.304166667*e10 + 42418.8350694444*e8,
        (4, 0, 5):  13885275.9401448*e14 - 2737716.23672287*e12 + 239949.195562134*e10,
        (4, 0, 6):  -13459442.5123663*e14 + 1168816.25004823*e12,
        (4, 0, 7):  5080086.80850045*e14,
        (4, 1, -7): 13.1045189610835*e14,
        (4, 1, -6): 33.824509118028*e14 + 6.92530394000772*e12,
        (4, 1, -5): 66.2469145815713*e14 + 19.8938053894043*e12 + 3.66662658691406*e10,
        (4, 1, -4): 112.829789220955*e14 + 41.270521556713*e12 + 11.6203125*e10 + 1.94835069444444*e8,
        (4, 1, -3): 176.401245009929*e14 + 73.2350860124753*e12 + 25.5508222791884*e10 + 6.76567925347222*e8 + 1.04210069444444*e6,
        (4, 1, -2): -0.5625*e4/(e2 - 1.0)**7,
        (4, 1, -1): 361.787971839249*e14 + 175.357567647298*e12 + 76.7125413682726*e10 + 29.2154405381944*e8 + 9.11067708333333*e6 + 2.0625*e4 + 0.25*e2,
        (4, 1, 0):  510.991944606836*e14 + 266.809137008102*e12 + 129.900034722222*e10 + 58.2599826388889*e8 + 23.5694444444444*e6 + 9.125*e4 + 2.0*e2 + 1.0,
        (4, 1, 1):  715.117373326165*e14 + 397.592113952637*e12 + 213.248858642578*e10 + 97.09716796875*e8 + 67.74609375*e6 - 1.6875*e4 + 20.25*e2,
        (4, 1, 2):  958.088995001791*e14 + 644.418093532986*e12 + 147.488194444444*e10 + 460.165798611111*e8 - 197.645833333333*e6 + 175.5625*e4,
        (4, 1, 3):  2405.01878383467*e14 - 1290.54145761184*e12 + 3201.53101433648*e10 - 1962.95241970486*e8 + 1030.67751736111*e6,
        (4, 1, 4):  -15810.425859375*e14 + 20109.588046875*e12 - 12399.046875*e10 + 4812.890625*e8,
        (4, 1, 5):  109198.707912504*e14 - 61669.7520816323*e12 + 19310.6487826199*e10,
        (4, 1, 6):  -262758.011185585*e14 + 69524.1394102045*e12,
        (4, 1, 7):  230739.513376018*e14,
        (4, 2, -7): 4203.9590960638*e14,
        (4, 2, -6): 2851.90764508929*e14 + 1693.8369140625*e12,
        (4, 2, -5): 3856.25669750377*e14 + 1570.4741746408*e12 + 655.906780666775*e10,
        (4, 2, -4): 4371.62514812059*e14 + 1995.2357494213*e12 + 764.401041666667*e10 + 240.896267361111*e8,
        (4, 2, -3): 4829.9199915954*e14 + 2301.10216140747*e12 + 966.631546020508*e10 + 333.82568359375*e8 + 82.12890625*e6,
        (4, 2, -2): 5187.27980668541*e14 + 2545.29351128472*e12 + 1123.84548611111*e10 + 427.777777777778*e8 + 129.166666666667*e6 + 25.0*e4,
        (4, 2, -1): 5426.83725947299*e14 + 2711.33776262071*e12 + 1233.00030178494*e10 + 494.570583767361*e8 + 166.048177083333*e6 + 42.1875*e4 + 6.25*e2,
        (4, 2, 0):  -2.25*(e2 + 0.666666666666667)**2/(e2 - 1.0)**7,
        (4, 2, 1):  5426.83725947299*e14 + 2711.33776262071*e12 + 1233.00030178494*e10 + 494.570583767361*e8 + 166.048177083333*e6 + 42.1875*e4 + 6.25*e2,
        (4, 2, 2):  5187.27980668541*e14 + 2545.29351128472*e12 + 1123.84548611111*e10 + 427.777777777778*e8 + 129.166666666667*e6 + 25.0*e4,
        (4, 2, 3):  4829.9199915954*e14 + 2301.10216140747*e12 + 966.631546020508*e10 + 333.82568359375*e8 + 82.12890625*e6,
        (4, 2, 4):  4371.62514812059*e14 + 1995.2357494213*e12 + 764.401041666667*e10 + 240.896267361111*e8,
        (4, 2, 5):  3856.25669750377*e14 + 1570.4741746408*e12 + 655.906780666775*e10,
        (4, 2, 6):  2851.90764508929*e14 + 1693.8369140625*e12,
        (4, 2, 7):  4203.9590960638*e14,
        (4, 3, -7): 230739.513376018*e14,
        (4, 3, -6): -262758.011185585*e14 + 69524.1394102045*e12,
        (4, 3, -5): 109198.707912504*e14 - 61669.7520816323*e12 + 19310.6487826199*e10,
        (4, 3, -4): -15810.425859375*e14 + 20109.588046875*e12 - 12399.046875*e10 + 4812.890625*e8,
        (4, 3, -3): 2405.01878383467*e14 - 1290.54145761184*e12 + 3201.53101433648*e10 - 1962.95241970486*e8 + 1030.67751736111*e6,
        (4, 3, -2): 958.088995001791*e14 + 644.418093532986*e12 + 147.488194444444*e10 + 460.165798611111*e8 - 197.645833333333*e6 + 175.5625*e4,
        (4, 3, -1): 715.117373326165*e14 + 397.592113952637*e12 + 213.248858642578*e10 + 97.09716796875*e8 + 67.74609375*e6 - 1.6875*e4 + 20.25*e2,
        (4, 3, 0):  510.991944606836*e14 + 266.809137008102*e12 + 129.900034722222*e10 + 58.2599826388889*e8 + 23.5694444444444*e6 + 9.125*e4 + 2.0*e2 + 1.0,
        (4, 3, 1):  361.787971839249*e14 + 175.357567647298*e12 + 76.7125413682726*e10 + 29.2154405381944*e8 + 9.11067708333333*e6 + 2.0625*e4 + 0.25*e2,
        (4, 3, 2):  -0.5625*e4/(e2 - 1.0)**7,
        (4, 3, 3):  176.401245009929*e14 + 73.2350860124753*e12 + 25.5508222791884*e10 + 6.76567925347222*e8 + 1.04210069444444*e6,
        (4, 3, 4):  112.829789220955*e14 + 41.270521556713*e12 + 11.6203125*e10 + 1.94835069444444*e8,
        (4, 3, 5):  66.2469145815713*e14 + 19.8938053894043*e12 + 3.66662658691406*e10,
        (4, 3, 6):  33.824509118028*e14 + 6.92530394000772*e12,
        (4, 3, 7):  13.1045189610835*e14,
        (4, 4, -7): 5080086.80850045*e14,
        (4, 4, -6): -13459442.5123663*e14 + 1168816.25004823*e12,
        (4, 4, -5): 13885275.9401448*e14 - 2737716.23672287*e12 + 239949.195562134*e10,
        (4, 4, -4): -7073073.58282352*e14 + 2428954.85556713*e12 - 485876.304166667*e10 + 42418.8350694444*e8,
        (4, 4, -3): 1864648.11796509*e14 - 1022671.96280935*e12 + 359793.938875665*e10 - 71820.7014431424*e8 + 6106.77126736111*e6,
        (4, 4, -2): -244265.454441964*e14 + 209706.022265625*e12 - 120013.70625*e10 + 42418.125*e8 - 8185.5*e6 + 650.25*e4,
        (4, 4, -1): 14027.1271612972*e14 - 19313.339785258*e12 + 17934.7833421495*e10 - 10497.2230360243*e8 + 3569.95442708333*e6 - 621.5625*e4 + 42.25*e2,
        (4, 4, 0):  -274.421647415911*e14 + 646.549230324074*e12 - 1034.92486111111*e10 + 1030.61024305556*e8 - 583.638888888889*e6 + 170.75*e4 - 22.0*e2 + 1.0,
        (4, 4, 1):  1.05155106272016*e14 - 5.07659271240234*e12 + 15.6179992675781*e10 - 30.61669921875*e8 + 31.18359375*e6 - 14.0625*e4 + 2.25*e2,
        (4, 4, 2):  -0.00308807319223986*e14 - 0.000998263888888889*e12 - 0.0305555555555556*e10 + 0.111111111111111*e8 - 0.333333333333333*e6 + 0.25*e4,
        (4, 4, 3):  0.000316369310678418*e14 + 0.000341919322072724*e12 + 0.000366550021701389*e10 + 0.000379774305555556*e8 + 0.000434027777777778*e6,
        (4, 4, 5):  2.88440226667562e-7*e14 + 1.75193504050926e-7*e12 + 6.78168402777778e-8*e10,
        (4, 4, 6):  5.51146384479718e-6*e14 + 1.92901234567901e-6*e12,
        (4, 4, 7):  1.14925540223414e-5*e14,
    }


    return eccentricity_results_bymode