""" Functions to calculate the initial guess for radial functions at the bottom of a solid or liquid layer

These functions do not allow for dynamic tides (w=0):
    - Compressibility
    - Liquid layer propagation

References
----------
KMN15 : Kamata+ (2015; JGR-P; DOI: 10.1002/2015JE004821)
B15   : Beuthe+ (2015; Icarus; DOI: 10.1016/j.icarus.2015.06.008)
TMS05 : Tobie+ (2005; Icarus; DOI: 10.1016/j.icarus.2005.04.006)
S74   : Saito (1974; J. Phy. Earth; DOI: 10.4294/jpe1952.22.123)
TS72  : Takeuchi, H., and M. Saito (1972), Seismic surface waves, Methods Comput. Phys., 11, 217–295.
"""

from typing import Tuple, Union

import numpy as np

from .initial_solution_dynamic import z_calc
from .functions import takeuchi_phi_psi
from ....constants import pi, G
from ....utilities.performance import njit
from ....utilities.types import FloatArray, ComplexArray

CmplxFltArray = Union[FloatArray, ComplexArray]
SolidGuess = Tuple[CmplxFltArray, CmplxFltArray]
LiquidGuess = CmplxFltArray


# @njit(cacheable=True)
# def solid_guess_kamata(radius: FloatArray, shear_modulus: CmplxFltArray, bulk_modulus: CmplxFltArray,
#                        density: FloatArray, order_l: int = 2) -> SolidGuess:
#     """ Calculate the initial guess at the bottom of a solid layer.
#
#     For these assumptions there will be three different initial guesses used to calculate three independent solutions.
#
#     Allows for general tidal harmonic l, for dynamic tides, and compressibility.
#
#     References
#     ----------
#     KMN15
#
#     Parameters
#     ----------
#     radius : FloatArray
#         Radius where the radial functions are calculated. [m]
#     shear_modulus : CmplxFltArray
#         Shear modulus (can be complex for dissipation) at `radius` [Pa]
#     bulk_modulus : CmplxFltArray
#         Bulk modulus (can be complex for dissipation) at `radius` [Pa]
#     density : FloatArray
#         Density at  at `radius` [kg m-3]
#     order_l : int = 2
#         Tidal harmonic
#
#     Returns
#     -------
#     solid_guesses : SolidGuess
#         The three solid guesses (sn1, sn2, sn3)
#
#     """
#
#     # Convert compressibility parameters
#     lame = bulk_modulus - (2. / 3.) * shear_modulus
#
#     # Constants (See Eqs. B13-B16 of KMN15)
#     alpha_2 = (lame + 2. * shear_modulus) / density
#     beta_2 = shear_modulus / density
#     gamma = 4. * pi * G * density / 3.
#
#     # Optimizations
#     r_inverse = 1. / radius
#     r2_inverse = r_inverse * r_inverse
#
#     # Helper functions
#     k2_quad_pos = (dynamic_term / beta_2) + ((dynamic_term + 4. * gamma) / alpha_2)
#     k2_quad_neg = (dynamic_term / beta_2) - ((dynamic_term + 4. * gamma) / alpha_2)
#     k2_quad = k2_quad_neg**2 + ((4. * order_l * (order_l + 1) * gamma**2) / (alpha_2 * beta_2))
#
#     # TODO: TS74 has these flipped compared to KMN15. Going with TS74 for now.
#     k2_pos = (1. / 2.) * (k2_quad_pos - np.sqrt(k2_quad))
#     k2_neg = (1. / 2.) * (k2_quad_pos + np.sqrt(k2_quad))
#
#     f = (beta_2 * k2) / gamma
#
#     h = f - (order_l + 1.)
#
#     z = z_calc(k2 * radius**2, order_l=order_l)
#
#     # See Eqs. B1-B12 of KMN15
#     # TODO: TS774 has the 3rd solutions' radius raised to l powers.
#     y1_s1 = -f_k2_pos * z_k2_pos * r_inverse
#     y1_s2 = -f_k2_neg * z_k2_neg * r_inverse
#     y1_s3 = order_l * r_inverse
#
#     y2_s1 = -density * f_k2_pos * alpha_2 * k2_pos + \
#             (2. * shear_modulus * r2_inverse) * (2. * f_k2_pos + order_l * (order_l + 1.)) * z_k2_pos
#     y2_s2 = -density * f_k2_neg * alpha_2 * k2_neg + \
#             (2. * shear_modulus * r2_inverse) * (2. * f_k2_neg + order_l * (order_l + 1.)) * z_k2_neg
#     y2_s3 = 2. * shear_modulus * order_l * (order_l - 1) * r2_inverse
#
#     y3_s1 = z_k2_pos * r_inverse
#     y3_s2 = z_k2_neg * r_inverse
#     y3_s3 = r_inverse
#
#     y4_s1 = shear_modulus * k2_pos - (2. * shear_modulus * r2_inverse) * (f_k2_pos + 1.) * z_k2_pos
#     y4_s2 = shear_modulus * k2_neg - (2. * shear_modulus * r2_inverse) * (f_k2_neg + 1.) * z_k2_neg
#     y4_s3 = 2. * shear_modulus * (order_l - 1.) * r2_inverse
#
#     y5_s1 = 3. * gamma * f_k2_pos - h_k2_pos * (order_l * gamma - dynamic_term)
#     y5_s2 = 3. * gamma * f_k2_neg - h_k2_neg * (order_l * gamma - dynamic_term)
#     y5_s3 = order_l * gamma - dynamic_term
#
#     y6_s1 = (2. * order_l + 1.) * y5_s1 * r_inverse
#     y6_s2 = (2. * order_l + 1.) * y5_s2 * r_inverse
#     y6_s3 = (2. * order_l + 1.) * y5_s3 * r_inverse - (3. * order_l * gamma * r_inverse)
#
#     # Combine the three solutions
#     tidaly_s1 = np.vstack((y1_s1, y2_s1, y3_s1, y4_s1, y5_s1, y6_s1))
#     tidaly_s2 = np.vstack((y1_s2, y2_s2, y3_s2, y4_s2, y5_s2, y6_s2))
#     tidaly_s3 = np.vstack((y1_s3, y2_s3, y3_s3, y4_s3, y5_s3, y6_s3))
#
#     return tidaly_s1, tidaly_s2, tidaly_s3
#
# # OPT: This can not be njited because the takeuchi_phi_psi function is not njited due to spherical bessel.
# # @njit(cacheable=True)
# def solid_guess_takeuchi(radius: FloatArray, shear_modulus: CmplxFltArray, bulk_modulus: CmplxFltArray,
#                          density: FloatArray, order_l: int = 2) -> SolidGuess:
#     """ Calculate the initial guess at the bottom of a solid layer.
#
#     For these assumptions there will be three different initial guesses used to calculate three independent solutions.
#
#     Allows for general tidal harmonic l, for dynamic tides, and compressibility.
#
#     References
#     ----------
#     TS72
#
#     Parameters
#     ----------
#     radius : FloatArray
#         Radius where the radial functions are calculated. [m]
#     shear_modulus : CmplxFltArray
#         Shear modulus (can be complex for dissipation) at `radius` [Pa]
#     bulk_modulus : CmplxFltArray
#         Bulk modulus (can be complex for dissipation) at `radius` [Pa]
#     density : FloatArray
#         Density at  at `radius` [kg m-3]
#     order_l : int = 2
#         Tidal harmonic
#
#     Returns
#     -------
#     solid_guesses : SolidGuess
#         The three solid guesses (sn1, sn2, sn3)
#
#     """
#
#     # Convert compressibility parameters
#     lame = bulk_modulus - (2. / 3.) * shear_modulus
#
#     # Constants (See Eqs. B13-B16 of KMN15)
#     dynamic_term = frequency * frequency
#     alpha_2 = (lame + 2. * shear_modulus) / density
#     beta_2 = shear_modulus / density
#     gamma = 4. * pi * G * density / 3.
#
#     # Optimizations
#     r_inverse = 1. / radius
#
#     # Helper functions
#     k2_quad_pos = (dynamic_term / beta_2) + ((dynamic_term + 4. * gamma) / alpha_2)
#     k2_quad_neg = (dynamic_term / beta_2) - ((dynamic_term + 4. * gamma) / alpha_2)
#     k2_quad = k2_quad_neg**2 + ((4. * order_l * (order_l + 1.) * gamma**2) / (alpha_2 * beta_2))
#
#     # TODO: TS74 has these flipped compared to KMN15. Going with TS74 for now.
#     k2_pos = (1. / 2.) * (k2_quad_pos - np.sqrt(k2_quad))
#     k2_neg = (1. / 2.) * (k2_quad_pos + np.sqrt(k2_quad))
#
#     f_k2_pos = (beta_2 * k2_pos - dynamic_term) / gamma
#     f_k2_neg = (beta_2 * k2_neg - dynamic_term) / gamma
#
#     h_k2_pos = f_k2_pos - (order_l + 1.)
#     h_k2_neg = f_k2_neg - (order_l + 1.)
#
#     # Calculate Takeuchi and Saito functions
#     # TODO: do we need to worry about the plus/minus on this sqrt?
#     # z_k2_pos = np.sqrt(k2_pos) * radius
#     # z_k2_neg = np.sqrt(k2_neg) * radius
#     z_k2_pos = k2_pos * radius**2
#     z_k2_neg = k2_neg * radius**2
#
#     phi_k2_pos, phi_lp1_k2_pos, psi_k2_pos = takeuchi_phi_psi(z_k2_pos, order_l)
#     phi_k2_neg, phi_lp1_k2_neg, psi_k2_neg = takeuchi_phi_psi(z_k2_neg, order_l)
#
#     # See Eq. 102 in TS72
#     # # y1 solutions
#     y1_s1 = ((-radius**(order_l + 1.)) / (2. * order_l + 3.)) * (0.5 * order_l * h_k2_pos * psi_k2_pos +
#                                                                  f_k2_pos * phi_lp1_k2_pos)
#     y1_s2 = ((-radius**(order_l + 1.)) / (2. * order_l + 3.)) * (0.5 * order_l * h_k2_neg * psi_k2_neg +
#                                                                  f_k2_neg * phi_lp1_k2_neg)
#     y1_s3 = order_l * radius**(order_l - 1.)
#
#     # # y2 solutions
#     y2_s1 = -(lame + 2. * shear_modulus) * radius**order_l * f_k2_pos * phi_k2_pos + \
#         (shear_modulus * radius**order_l / (2. * order_l + 3.)) * \
#             (-order_l * (order_l - 1.) * h_k2_pos * psi_k2_pos +
#              2. * (2. * f_k2_pos + order_l * (order_l + 1.)) * phi_lp1_k2_pos)
#     y2_s2 = -(lame + 2. * shear_modulus) * radius**order_l * f_k2_neg * phi_k2_neg + \
#             (shear_modulus * radius**order_l / (2. * order_l + 3.)) * \
#             (-order_l * (order_l - 1.) * h_k2_neg * psi_k2_neg +
#              2. * (2. * f_k2_neg + order_l * (order_l + 1.)) * phi_lp1_k2_neg)
#     y2_s3 = 2. * shear_modulus * order_l * (order_l - 1.) * radius**(order_l - 2.)
#
#     # # y3 solutions
#     y3_s1 = (-radius**(order_l + 1.) / (2. * order_l + 3.)) * (0.5 * h_k2_pos * psi_k2_pos - phi_lp1_k2_pos)
#     y3_s2 = (-radius**(order_l + 1.) / (2. * order_l + 3.)) * (0.5 * h_k2_neg * psi_k2_neg - phi_lp1_k2_neg)
#     y3_s3 = radius**(order_l - 1.)
#
#     # # y4 solutions
#     y4_s1 = shear_modulus * radius**order_l * \
#             (phi_k2_pos - (1. / (2. * order_l + 3.)) * ((order_l - 1.) * h_k2_pos * psi_k2_pos +
#                                                         2. * (f_k2_pos + 1.) * phi_lp1_k2_pos))
#     y4_s2 = shear_modulus * radius**order_l * \
#             (phi_k2_neg - (1. / (2. * order_l + 3.)) * ((order_l - 1.) * h_k2_neg * psi_k2_neg +
#                                                         2. * (f_k2_neg + 1.) * phi_lp1_k2_neg))
#     y4_s3 = 2. * shear_modulus * (order_l - 1.) * radius**(order_l - 2.)
#
#     # # y5 solutions
#     y5_s1 = radius**(order_l + 2.) * ((alpha_2 * f_k2_pos - (order_l + 1.) * beta_2) / radius**2 -
#                                       (3. * gamma * f_k2_pos / (2. * (2. * order_l + 3.))) * psi_k2_pos)
#     y5_s2 = radius**(order_l + 2.) * ((alpha_2 * f_k2_neg - (order_l + 1.) * beta_2) / radius**2 -
#                                       (3. * gamma * f_k2_neg / (2. * (2. * order_l + 3.))) * psi_k2_neg)
#     y5_s3 = (order_l * gamma - dynamic_term) * radius**order_l
#
#     # # y6 solutions
#     y6_s1 = (2. * order_l + 1.) * r_inverse * y5_s1 + \
#             (3. * order_l * gamma * h_k2_pos * radius**(order_l + 1.) / (2. * (2. * order_l + 3.))) * psi_k2_pos
#     y6_s2 = (2. * order_l + 1.) * r_inverse * y5_s2 + \
#             (3. * order_l * gamma * h_k2_neg * radius**(order_l + 1.) / (2. * (2. * order_l + 3.))) * psi_k2_neg
#     y6_s3 = (2. * order_l + 1.) * r_inverse * y5_s3 - \
#             3. * order_l * gamma * radius**(order_l - 1.)
#
#     # Combine the three solutions
#     tidaly_s1 = np.vstack((y1_s1, y2_s1, y3_s1, y4_s1, y5_s1, y6_s1))
#     tidaly_s2 = np.vstack((y1_s2, y2_s2, y3_s2, y4_s2, y5_s2, y6_s2))
#     tidaly_s3 = np.vstack((y1_s3, y2_s3, y3_s3, y4_s3, y5_s3, y6_s3))
#
#     return tidaly_s1, tidaly_s2, tidaly_s3


# OPT: This can not be njited because the takeuchi_phi_psi function is not njited due to spherical bessel.
# @njit(cacheable=True)
def liquid_guess_saito(radius: FloatArray, bulk_modulus: CmplxFltArray,
                          density: FloatArray, order_l: int = 2) -> LiquidGuess:
    """ Calculate the initial guess at the bottom of a solid layer.

    For these assumptions there will be three different initial guesses used to calculate three independent solutions.

    Allows for general tidal harmonic l, for dynamic tides, and compressibility.

    References
    ----------
    Saito 1974

    Parameters
    ----------
    radius : FloatArray
        Radius where the radial functions are calculated. [m]
    bulk_modulus : CmplxFltArray
        Bulk modulus (can be complex for dissipation) at `radius` [Pa]
    density : FloatArray
        Density at  at `radius` [kg m-3]
    order_l : int = 2
        Tidal harmonic

    Returns
    -------
    solid_guesses : SolidGuess
        The three solid guesses (sn1, sn2, sn3)

    """

    # See Eq. 19 in Saito 1974
    # # y5 solutions
    y5_s1 = radius**order_l

    # # y6 solutions
    y6_s1 = 2. * (order_l - 1.) * radius**(order_l - 1.)

    # Combine the three solutions
    tidaly_s1 = np.vstack((y5_s1, y6_s1))

    return tidaly_s1

# @njit(cacheable=True)
# def liquid_guess_kamata(radius: FloatArray, bulk_modulus: CmplxFltArray,
#                         density: FloatArray, order_l: int = 2) -> LiquidGuess:
#     """ Calculate the initial guess at the bottom of a liquid layer.
#
#     For these assumptions there will be two different initial guesses used to calculate two independent solutions.
#
#     Allows for general tidal harmonic l, for dynamic tides, and compressibility.
#
#     References
#     ----------
#     KMN15
#
#     Parameters
#     ----------
#     radius : FloatArray
#         Radius where the radial functions are calculated. [m]
#     bulk_modulus : CmplxFltArray
#         Bulk modulus (can be complex for dissipation) at `radius` [Pa]
#     density : FloatArray
#         Density at  at `radius` [kg m-3]
#     order_l : int = 2
#         Tidal harmonic
#
#     Returns
#     -------
#     solid_guesses : LiquidGuess
#         The three solid guesses (sn1, sn2, sn3)
#
#     """
#
#     # Convert compressibility parameters
#     # For the liquid layer the shear modulus is zero so the 1st Lame parameter = bulk modulus
#     lame = bulk_modulus
#
#     # Optimizations
#     r_inverse = 1. / radius
#
#     # Helper functions
#     gamma = (4. * pi * G * density / 3.)
#     alpha_2 = lame / density
#     k2 = (1. / alpha_2) * (dynamic_term + 4. * gamma - (order_l * (order_l + 1) * gamma**2 / dynamic_term))
#     f = -dynamic_term / gamma
#     h = f - (order_l + 1.)
#
#     # See Eqs. B33--B36 in KMN15
#     y1_s1 = -f * r_inverse * z_calc(k2 * radius**2, order_l=order_l)
#     y1_s2 = order_l * r_inverse
#
#     y2_s1 = -density * (f * (dynamic_term + 4 * gamma) + order_l * (order_l + 1) * gamma)
#     y2_s2 = 0. * radius
#
#     y5_s1 = 3. * gamma * f - h * (order_l * gamma - dynamic_term)
#     y5_s2 = order_l * gamma - dynamic_term
#
#     y6_s1 = (2. * order_l + 1.) * y5_s1 * r_inverse
#     y6_s2 = ((2. * order_l + 1) * y5_s2 * r_inverse) - ((3. * order_l * gamma) * r_inverse)
#
#     # Combine the two solutions
#     tidaly_s1 = np.vstack((y1_s1, y2_s1, y5_s1, y6_s1))
#     tidaly_s2 = np.vstack((y1_s2, y2_s2, y5_s2, y6_s2))
#
#     return tidaly_s1, tidaly_s2
