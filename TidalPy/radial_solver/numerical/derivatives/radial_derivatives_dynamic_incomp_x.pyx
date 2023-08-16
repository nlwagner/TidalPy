# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
""" Functions to calculate the Rayleigh wave propagation differential equations

References
----------
KMN15 : Kamata+ (2015; JGR-P; DOI: 10.1002/2015JE004821)
B15   : Beuthe+ (2015; Icarus; DOI: 10.1016/j.icarus.2015.06.008)
TMS05 : Tobie+ (2005; Icarus; DOI: 10.1016/j.icarus.2005.04.006)
S74   : Saito (1974; J. Phy. Earth; DOI: 10.4294/jpe1952.22.123)
TS72  : Takeuchi, H., and M. Saito (1972), Seismic surface waves, Methods Comput. Phys., 11, 217–295.
"""

import cython
from scipy.constants import G as G_

from libc.math cimport pi

cdef double G
G = G_


@cython.exceptval(check=False)
cdef void dy_solid_dynamic_incompressible_x(
    double radius, double[:] radial_functions, double[:] dy,
    double complex shear_modulus, double density, double gravity, double frequency,
    unsigned int degree_l=2, double G_to_use=G) nogil:
    """ Calculates the radial derivative of the radial functions - for incompressible solid layers.

    Allows for dynamic tides.
    Tidal degree l is allowed to be an integer >= 2.

    References
    ----------
    KMN15; B15; TS72

    Parameters
    ----------
    radius : float
        Radius where the radial functions are calculated. [m]
    radial_functions : np.ndarray
        Tuple of radial functions for a solid layer broken up into real and imaginary portions.
        (y1_real, y1_imag, y2_real, y2_imag, y3_real, y3_imag, y4_real, y4_imag, y5_real, y5_imag, y6_real, y6_imag)
    dy : np.ndarray
        Derivative of the radial functions with respect to radius.
    shear_modulus : float
        Shear modulus (can be complex for dissipation) at `radius` [Pa]
    density : float
        Density at `radius` [kg m-3]
    gravity : float
        Acceleration due to gravity at `radius` [m s-2]
    frequency : float
        Forcing frequency (for spin-synchronous tides this is the orbital motion) [rad s-1]
    degree_l : int = 2
        Tidal harmonic degree.
    G_to_use : float = G
        Newton's gravitational constant. This can be provided as in its MKS units or dimensionless to match the other
        inputs.

    """

    # Pull out y values
    cdef double y1_real, y2_real, y3_real, y4_real, y5_real, y6_real
    cdef double y1_imag, y2_imag, y3_imag, y4_imag, y5_imag, y6_imag

    y1_real = radial_functions[0]
    y1_imag = radial_functions[1]
    y2_real = radial_functions[2]
    y2_imag = radial_functions[3]
    y3_real = radial_functions[4]
    y3_imag = radial_functions[5]
    y4_real = radial_functions[6]
    y4_imag = radial_functions[7]
    y5_real = radial_functions[8]
    y5_imag = radial_functions[9]
    y6_real = radial_functions[10]
    y6_imag = radial_functions[11]

    # Convert floats to complex
    cdef double complex y1, y2, y3, y4, y5, y6

    y1 = y1_real + 1.0j * y1_imag
    y2 = y2_real + 1.0j * y2_imag
    y3 = y3_real + 1.0j * y3_imag
    y4 = y4_real + 1.0j * y4_imag
    y5 = y5_real + 1.0j * y5_imag
    y6 = y6_real + 1.0j * y6_imag

    # Optimizations
    cdef double degree_l_flt, lp1, lm1, llp1
    cdef double r_inverse, density_gravity, dynamic_term, grav_term
    cdef double complex two_shear_r_inv, y1_y3_term

    degree_l_flt     = <double>degree_l
    lp1              = degree_l_flt + 1.
    lm1              = degree_l_flt - 1.
    llp1             = degree_l_flt * lp1
    r_inverse        = 1. / radius
    two_shear_r_inv  = 2. * shear_modulus * r_inverse
    density_gravity  = density * gravity
    dynamic_term     = -frequency * frequency * density * radius
    grav_term        = 4. * pi * G_to_use * density
    y1_y3_term       = 2. * y1 - llp1 * y3


    cdef double complex dy1, dy2, dy3, dy4, dy5, dy6

    dy1 = y1_y3_term * -1. * r_inverse

    dy2 = r_inverse * (
            y1 * (dynamic_term + 12. * shear_modulus * r_inverse - 4. * density_gravity) +
            y3 * llp1 * (density_gravity - 6. * shear_modulus * r_inverse) +
            y4 * llp1 +
            y5 * density * lp1 +
            y6 * -density * radius
    )

    dy3 = \
        y1 * -r_inverse + \
        y3 * r_inverse + \
        y4 * (1. / shear_modulus)

    dy4 = r_inverse * (
            y1 * (density_gravity - 3. * two_shear_r_inv) +
            y2 * -1. +
            y3 * (dynamic_term + two_shear_r_inv * (2. * llp1 - 1.))+
            y4 * -3. +
            y5 * -density
    )

    dy5 = \
        y1 * grav_term + \
        y5 * -lp1 * r_inverse + \
        y6

    dy6 = r_inverse * (
            y1 * grav_term * lm1 +
            y6 * lm1 +
            y1_y3_term * grav_term
    )

    # Convert back to floats
    dy[0] = dy1.real
    dy[1] = dy1.imag
    dy[2] = dy2.real
    dy[3] = dy2.imag
    dy[4] = dy3.real
    dy[5] = dy3.imag
    dy[6] = dy4.real
    dy[7] = dy4.imag
    dy[8] = dy5.real
    dy[9] = dy5.imag
    dy[10] = dy6.real
    dy[11] = dy6.imag


@cython.exceptval(check=False)
cdef void dy_liquid_dynamic_incompressible_x(
    double radius, double[:] radial_functions, double[:] dy,
    double density, double gravity, double frequency,
    unsigned int degree_l=2, double G_to_use=G) nogil:
    """ Calculates the radial derivative of the radial functions - for incompressible solid layers.

    Allows for dynamic tides.
    Tidal degree l is allowed to be an integer >= 2.

    References
    ----------
    KMN15; B15; TS72

    Parameters
    ----------
    radius : float
        Radius where the radial functions are calculated. [m]
    radial_functions : np.ndarray
        Tuple of radial functions for a solid layer broken up into real and imaginary portions.
        (y1_real, y1_imag, y2_real, y2_imag, y3_real, y3_imag, y4_real, y4_imag, y5_real, y5_imag, y6_real, y6_imag)
    dy : np.ndarray
        Derivative of the radial functions with respect to radius.
    density : float
        Density at `radius` [kg m-3]
    gravity : float
        Acceleration due to gravity at `radius` [m s-2]
    frequency : float
        Forcing frequency (for spin-synchronous tides this is the orbital motion) [rad s-1]
    degree_l : int = 2
        Tidal harmonic degree.
    G_to_use : float = G
        Newton's gravitational constant. This can be provided as in its MKS units or dimensionless to match the other
        inputs.

    """

    # Pull out y values
    cdef double y1_real, y2_real, y5_real, y6_real
    cdef double y1_imag, y2_imag, y5_imag, y6_imag

    y1_real = radial_functions[0]
    y1_imag = radial_functions[1]
    y2_real = radial_functions[2]
    y2_imag = radial_functions[3]
    y5_real = radial_functions[4]
    y5_imag = radial_functions[5]
    y6_real = radial_functions[6]
    y6_imag = radial_functions[7]

    # Convert floats to complex
    cdef double complex y1, y2, y5, y6

    y1 = y1_real + 1.0j * y1_imag
    y2 = y2_real + 1.0j * y2_imag
    y5 = y5_real + 1.0j * y5_imag
    y6 = y6_real + 1.0j * y6_imag

    # Optimizations
    cdef double degree_l_flt, lp1, lm1, llp1
    cdef double r_inverse, density_gravity, dynamic_term, grav_term

    degree_l_flt     = <double>degree_l
    lp1              = degree_l_flt + 1.
    lm1              = degree_l_flt - 1.
    llp1             = degree_l_flt * lp1
    r_inverse        = 1. / radius
    density_gravity  = density * gravity
    dynamic_term     = -frequency * frequency * density * radius
    grav_term        = 4. * pi * G_to_use * density

    # y3 derivative is undetermined for a liquid layer, but we can calculate its value which is still used in the
    #   other derivatives.
    cdef double complex y3, y1_y3_term
    y3 = (1. / dynamic_term) * (y2 + density * y5 - density_gravity * y1)
    y1_y3_term = 2. * y1 - llp1 * y3


    cdef double complex dy1, dy2, dy5, dy6

    dy1 = y1_y3_term * -r_inverse

    dy2 = r_inverse * (
            y1 * (dynamic_term - 2. * density_gravity) +
            y5 * density * lp1 +
            y6 * -density * radius +
            # TODO: In the solid version there is a [2. * (lame + shear_modulus) * r_inverse] coefficient for y1_y3_term
            #   In TS72 the first term is gone. Shouldn't Lame + mu = Lame = Bulk for liquid layer?
            y1_y3_term * -density_gravity
    )

    dy5 = \
        y1 * grav_term + \
        y5 * -lp1 * r_inverse + \
        y6

    dy6 = r_inverse * (
            y1 * grav_term * lm1 +
            y6 * lm1 +
            y1_y3_term * grav_term
    )

    # Convert back to floats
    dy[0] = dy1.real
    dy[1] = dy1.imag
    dy[2] = dy2.real
    dy[3] = dy2.imag
    dy[4] = dy5.real
    dy[5] = dy5.imag
    dy[6] = dy6.real
    dy[7] = dy6.imag