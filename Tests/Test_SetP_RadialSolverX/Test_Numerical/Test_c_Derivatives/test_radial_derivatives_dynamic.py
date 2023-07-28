""" Test the `TidalPy.radial_solver.radial_derivatives_dynamic` (cython extension) functionality. """

import numpy as np
import pytest

import TidalPy
TidalPy.test_mode()

from TidalPy.radial_solver.numerical.derivatives.radial_derivatives_dynamic_wrap import (
    radial_derivatives_solid_general_wrap, radial_derivatives_liquid_general_wrap
    )

radius = 100.
gravity = 1.5
density = 3000.
shear_modulus = 1.0e50 + 200.j
bulk_modulus = 1.0e50
frequency = 1.0e-5
G = 6.67430e-11


@pytest.mark.parametrize('degree_l', (2, 3))
def test_radial_derivatives_solid_general(degree_l):

    # This function takes 6 complex-valued y variables, so we need to give it 12 floats.
    y = np.asarray(
        (100., 2., 100., 2., 300., 3., 400., 4., 600., 9., -33., 2.), dtype=np.float64
        )
    dy = np.empty(12, dtype=np.float64)

    # Check dimensions make sense
    radial_derivatives_solid_general_wrap(
        radius, y, dy, shear_modulus, bulk_modulus, density, gravity, frequency, degree_l, G_to_use=G
        )

    assert dy.shape == (12,)
    assert dy.dtype == np.float64
    # Check that the results match expectations
    if degree_l == 2:
        expected_dy = np.asarray(
            [2.286e+00, 2.000e-02, -4.114e+49, -3.600e+47, 2.000e+00, 1.000e-02, 3.257e+49, 3.000e+47,
                -5.100e+01, 1.730e+00, -3.300e-01, 2.000e-02], dtype=np.float64
            )
    elif degree_l == 3:
        expected_dy = np.asarray(
            [4.857e+00, 4.571e-02, -8.743e+49, -8.229e+47, 2.000e+00, 1.000e-02,
                7.371e+49, 7.114e+47, -5.700e+01, 1.640e+00, -6.601e-01, 4.000e-02], dtype=np.float64
            )
    assert np.allclose(dy, expected_dy, rtol=0.001)

    # Check that results don't change if we use the default G
    dy2 = np.empty(12, dtype=np.float64)
    radial_derivatives_solid_general_wrap(
        radius, y, dy2, shear_modulus, bulk_modulus, density, gravity, frequency, degree_l
        )

    assert np.allclose(dy, dy2)

    # Check that they do change if we don't
    dy3 = np.empty(12, dtype=np.float64)
    radial_derivatives_solid_general_wrap(
        radius, y, dy3, shear_modulus, bulk_modulus, density, gravity, frequency, degree_l, G_to_use=1.
        )

    assert not np.allclose(dy, dy3)


@pytest.mark.parametrize('degree_l', (2, 3))
def test_radial_derivatives_liquid_general(degree_l):

    # This function takes 4 complex-valued y variables, so we need to give it 8 floats.
    y = np.asarray(
        (100., 2., 100., 2., 300., 3., 400., 4.), dtype=np.float64
        )
    dy = np.empty(8, dtype=np.float64)

    # Check dimensions make sense
    radial_derivatives_liquid_general_wrap(
        radius, y, dy, bulk_modulus, density, gravity, frequency, degree_l, G_to_use=G
        )

    assert dy.shape == (8,)
    assert dy.dtype == np.float64
    # Check that the results match expectations
    if degree_l == 2:
        expected_dy = np.asarray(
            [-9.002e+08, -4.000e+03, -4.051e+12, -1.801e+07, 3.910e+02,
                3.910e+00, 2.269e+03, 5.006e-02], dtype=np.float64
            )
    elif degree_l == 3:
        expected_dy = np.asarray(
            [-1.800e+09, -8.000e+03, -8.102e+12, -3.601e+07, 3.880e+02,
                3.880e+00, 4.538e+03, 1.001e-01], dtype=np.float64
            )
    assert np.allclose(dy, expected_dy, rtol=0.001)

    # Check that results don't change if we use the default G
    dy2 = np.empty(8, dtype=np.float64)
    radial_derivatives_liquid_general_wrap(
        radius, y, dy2, bulk_modulus, density, gravity, frequency, degree_l
        )

    assert np.allclose(dy, dy2)

    # Check that they do change if we don't
    dy3 = np.empty(8, dtype=np.float64)
    radial_derivatives_liquid_general_wrap(
        radius, y, dy3, bulk_modulus, density, gravity, frequency, degree_l, G_to_use=1.
        )

    assert not np.allclose(dy, dy3)