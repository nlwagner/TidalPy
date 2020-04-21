
import numpy as np
import pytest
import numba

from TidalPy.configurations import use_numba

def test_eccentricity_func_load():

    from TidalPy.tides.eccentricityFuncs import eccentricity_truncations

    # As of version 0.2, TidalPy should contain truncations up to e^20 (only even truncations)
    assert len(eccentricity_truncations) == 10

    # As of version 0.2, each truncation level should contain l = 2 ... 7 harmonics
    for funcs_by_orderl in eccentricity_truncations:

        # FIXME: for now we don't have l=5,6,7
        # assert len(funcs_by_orderl) == 6
        assert len(funcs_by_orderl) >= 3


def test_eccentricity_func():

    from TidalPy.tides.eccentricityFuncs import eccentricity_truncations

    print('Testing Eccentricity Functions. Compile times for njit may take a long time.')
    # Each eccentricity function should return a nested dictionary setup like:
    #   {p_0: {q_0: <result>, q_1: <result>, ...}, p_1: {...}, ...}

    for funcs_by_orderl in eccentricity_truncations:

        # As of 0.2, the max l = 7
        # FIXME: for now we don't have l=5,6,7
        for order_l in range(2, 4+1):

            # Pull out function. Index 0 corresponds to order-l = 2.
            e_func = funcs_by_orderl[order_l - 2]

            # Perform float calculation
            e_result_float = e_func(0.3)
            for p, p_result in e_result_float.items():
                for q, q_result in p_result.items():
                    assert type(q_result) is float

            # Perform array calculation
            e_result_array = e_func(np.linspace(0.1, 0.4, 4))
            for p, p_result in e_result_array.items():
                for q, q_result in p_result.items():
                    assert type(q_result) is np.ndarray


def test_inclination_func_load():

    from TidalPy.tides.inclinationFuncs import inclination_functions, inclination_functions_on,\
        inclination_functions_off

    # inclination_functions should contain two options: one for obliquity tides on and one for them being off.
    assert len(inclination_functions) == 2
    assert inclination_functions[True] is inclination_functions_on
    assert inclination_functions[False] is inclination_functions_off

    # As of version 0.2, there should be l = 2 ... 7 inclination functions predefined
    assert len(inclination_functions_on) == 6
    assert len(inclination_functions_off) == 6


def test_inclination_func():

    from TidalPy.tides.inclinationFuncs import inclination_functions_on, inclination_functions_off

    # Each inclination function should return a dictionary setup like:
    #   {(p_0, m_0): <result>, (p_0, m_1): <result>, ... (p_1, m_0): <result>, ...}

    for order_l in range(2, 7+1):

        # Pull out obliquity function
        obliqu_func_on = inclination_functions_on[order_l - 2]
        obliqu_func_off = inclination_functions_off[order_l - 2]

        # Perform float calculation
        obliqu_result_on_float = obliqu_func_on(0.1)
        obliqu_result_off_float = obliqu_func_off(0.1)

        if use_numba:
            assert isinstance(obliqu_result_on_float, numba.typed.typeddict.Dict)
            assert isinstance(obliqu_result_off_float, numba.typed.typeddict.Dict)
        else:
            assert type(obliqu_result_on_float) is dict
            assert type(obliqu_result_off_float) is dict

        for (p, m), result_off in obliqu_result_on_float.items():
            assert type(p) is int
            assert type(m) is int
            assert type(result_off) is float
        for (p, m), result_on in obliqu_result_off_float.items():
            assert type(p) is int
            assert type(m) is int
            assert type(result_on) is float

        # Perform array calculation
        obliqu_result_on_array = obliqu_func_on(np.linspace(0.1, 0.4, 4))
        obliqu_result_off_array = obliqu_func_off(np.linspace(0.1, 0.4, 4))

        if use_numba:
            assert isinstance(obliqu_result_on_float, numba.typed.typeddict.Dict)
            assert isinstance(obliqu_result_off_float, numba.typed.typeddict.Dict)
        else:
            assert type(obliqu_result_on_float) is dict
            assert type(obliqu_result_off_float) is dict

        for (p, m), result_off in obliqu_result_on_array.items():
            assert type(p) is int
            assert type(m) is int
            assert type(result_off) is np.ndarray
        for (p, m), result_on in obliqu_result_off_array.items():
            assert type(p) is int
            assert type(m) is int
            assert type(result_on) is np.ndarray