""" Tides Compliance Functions

How To Implement a New Tides:
    * Make a new function here with a function doc-string that follows the format of the pre-built rheologies.
        - Items under the 'Parameters' header will be used to help construct the rheology. You need to supply them.
        - All rheologies require three arguments: compliance (non-complex), viscosity, and frequency.
            These should be the first arguments and in this order.
        - Additional arguments can be supplied by adding them after these required arguments but they must be included in the function's
            doc-string after the 'other args' key (in the same order).
    * Any new arguments that are simple constants can then be added to the rheology section of a planet's layer configuration.
        - Non-constants will require additional coding. As of right now there is not a clean way to add such functionality. But, you can
            look at the zeta-frequency dependency implementation.
"""

import numpy as np

from TidalPy.performance import find_factorial, njit
from TidalPy.types import float_eps, float_lognat_max


@njit
def off(frequency: float, compliance: float, viscosity: float) -> complex:
    """ Calculates the complex compliance utilizing the model: Off

    !TPY_args live: self.compliance, self.viscosity

    Parameters
    ----------
    frequency : float
        Planet's tidal frequency [rads s-1]
        Note that a planet may experience multiple tidal frequencies for NSR tides or Fourier Degree l>2
    compliance : float
        Layer or Planet's compliance (inverse of shear modulus) [Pa-1]
    viscosity : float
        Layer or Planet's effective viscosity [Pa s]

    Returns
    -------
    complex_compliance : float
        Complex compliance (complex number) [Pa-1]
    """

    real_j = compliance
    imag_j = 0.

    complex_compliance = real_j + 1.0j * imag_j

    return complex_compliance


@njit
def fixed_q(frequency: float, compliance: float, viscosity: float,
            planet_beta: float = 3.443e11, quality_factor: float = 10.) -> complex:
    """ Calculates the complex compliance utilizing the model: Fixed-Q

    !TPY_args live: self.compliance, self.viscosity, self.beta, self.quality_factor

    Parameters
    ----------
    frequency : float
        Planet's tidal frequency [rads s-1]
        Note that a planet may experience multiple tidal frequencies for NSR tides or Fourier Degree l>2
    compliance : float
        Layer or Planet's compliance (inverse of shear modulus) [Pa-1]
    viscosity : float
        Layer or Planet's effective viscosity [Pa s]
    planet_beta : float
        Planet or Layer: Radius * Density * Gravity Accl. [kg m-1 s-2]
    quality_factor : float
        Planet or Layer's Quality factor (k_2 / Q)

    Returns
    -------
    complex_compliance : float
        Complex compliance (complex number) [Pa-1]
    """

    real_j = -19. / (2. * planet_beta)
    imag_j = -quality_factor * (19. / 4.) * (2. * planet_beta * compliance + 19.) / (compliance * planet_beta**2)

    if np.abs(frequency) <= float_eps:
        imag_j = 0.

    complex_compliance = real_j + 1.0j * imag_j

    return complex_compliance


@njit
def maxwell(frequency: float, compliance: float, viscosity: float) -> complex:
    """ Calculates the complex compliance utilizing the model: Maxwell

    !TPY_args live: self.compliance, self.viscosity

    Parameters
    ----------
    frequency : float
        Planet's tidal frequency [rads s-1]
        Note that a planet may experience multiple tidal frequencies for NSR tides or Fourier Degree l>2
    compliance : float
        Layer or Planet's compliance (inverse of shear modulus) [Pa-1]
    viscosity : float
        Layer or Planet's effective viscosity [Pa s]

    Returns
    -------
    complex_compliance : float
        Complex compliance (complex number) [Pa-1]
    """

    real_j = compliance
    denominator = (viscosity * frequency)
    imag_j = -1.0 / denominator

    if np.abs(denominator - 1.) <= float_eps:
        imag_j = 0.

    complex_compliance = real_j + 1.0j * imag_j

    return complex_compliance


@njit
def voigt(frequency: float, compliance: float, viscosity: float,
          voigt_compliance_offset: float = 0.2, voigt_viscosity_offset: float = 0.02) -> complex:
    """ Calculates the complex compliance utilizing the model: Voigt-Kelvin

    !TPY_args live: self.compliance, self.viscosity
    !TPY_args const: voigt_compliance_offset, voigt_viscosity_offset

    Parameters
    ----------
    frequency : float
        Planet's tidal frequency [rads s-1]
        Note that a planet may experience multiple tidal frequencies for NSR tides or Fourier Degree l>2
    compliance : float
        Layer or Planet's compliance (inverse of shear modulus) [Pa-1]
    viscosity : float
        Layer or Planet's effective viscosity [Pa s]
    voigt_compliance_offset : float
        Voigt component's compliance offset eta_voigt = voigt_compliance_offset * compliance
    voigt_viscosity_offset : float
        Voigt component's viscosity offset eta_voigt = voigt_viscosity_offset * viscosity

    Returns
    -------
    complex_compliance : float
        Complex compliance (complex number) [Pa-1]
    """

    voigt_comp = voigt_compliance_offset * compliance
    voigt_visc = voigt_viscosity_offset * viscosity

    denominator = (voigt_comp * voigt_visc * frequency)**2 + 1.
    real_j = voigt_comp / denominator
    imag_j = -voigt_comp**2 * voigt_visc * frequency / denominator

    if np.abs(denominator - 1.) <= float_eps:
        imag_j = 0.

    complex_compliance = real_j + 1.0j * imag_j

    return complex_compliance


@njit
def burgers(frequency: float, compliance: float, viscosity: float,
            voigt_compliance_offset: float = 0.2, voigt_viscosity_offset: float = 0.02) -> complex:
    """ Calculates the complex compliance utilizing the model: Burgers

    !TPY_args live: self.compliance, self.viscosity
    !TPY_args const: voigt_compliance_offset, voigt_viscosity_offset

    Parameters
    ----------
    frequency : float
        Planet's tidal frequency [rads s-1]
        Note that a planet may experience multiple tidal frequencies for NSR tides or Fourier Degree l>2
    compliance : float
        Layer or Planet's compliance (inverse of shear modulus) [Pa-1]
    viscosity : float
        Layer or Planet's effective viscosity [Pa s]
    voigt_compliance_offset : float
        Voigt component's compliance offset eta_voigt = voigt_compliance_offset * compliance
    voigt_viscosity_offset : float
        Voigt component's viscosity offset eta_voigt = voigt_viscosity_offset * viscosity

    Returns
    -------
    complex_compliance : float
        Complex compliance (complex number) [Pa-1]
    """

    voigt_complex_comp = voigt(compliance, viscosity, frequency, voigt_compliance_offset, voigt_viscosity_offset)
    maxwell_complex_comp = maxwell(compliance, viscosity, frequency)

    complex_compliance = voigt_complex_comp + maxwell_complex_comp

    return complex_compliance


@njit
def andrade(frequency: float, compliance: float, viscosity: float, alpha: float = 0.3, zeta: float = 1.) -> complex:
    """ Calculates the complex compliance utilizing the model: Andrade

    !TPY_args live: self.compliance, self.viscosity
    !TPY_args const: alpha, zeta

    Parameters
    ----------
    frequency : float
        Planet's tidal frequency [rads s-1]
        Note that a planet may experience multiple tidal frequencies for NSR tides or Fourier Degree l>2
    compliance : float
        Layer or Planet's compliance (inverse of shear modulus) [Pa-1]
    viscosity : float
        Layer or Planet's effective viscosity [Pa s]
    alpha : float
        Andrade exponent parameter
    zeta : float
        Andrade timescale parameter

    Returns
    -------
    complex_compliance : float
        Complex compliance (complex number) [Pa-1]
    """

    maxwell_complex_comp = maxwell(compliance, viscosity, frequency)

    andrade_term = compliance * viscosity * frequency * zeta
    const_term = compliance * andrade_term**(-alpha) * find_factorial(alpha)
    real_j = np.cos(alpha * np.pi / 2.) * const_term
    imag_j = -np.sin(alpha * np.pi / 2.) * const_term

    if np.abs(andrade_term) <= float_eps:
        imag_j = 0.

    andrade_complex_comp = real_j + 1.0j * imag_j

    complex_compliance = maxwell_complex_comp + andrade_complex_comp

    return complex_compliance

@njit
def andrade_freq(frequency: float, compliance: float, viscosity: float,
                 alpha: float = 0.3, zeta: float = 1., critical_freq: float = 2.e-5) -> complex:
    """ Calculates the complex compliance utilizing the model: Andrade with a frequency-dependent zeta

    !TPY_args live: self.compliance, self.viscosity
    !TPY_args const: alpha, zeta, critical_freq

    Parameters
    ----------
    frequency : float
        Planet's tidal frequency [rads s-1]
        Note that a planet may experience multiple tidal frequencies for NSR tides or Fourier Degree l>2
    compliance : float
        Layer or Planet's compliance (inverse of shear modulus) [Pa-1]
    viscosity : float
        Layer or Planet's effective viscosity [Pa s]
    alpha : float
        Andrade exponent parameter
    zeta : float
        Andrade timescale parameter
    critical_freq : float
        For forcing frequencies larger than the critical frequency, the Andrade component converges to
            the Maxwell component [rads s-1]

    Returns
    -------
    complex_compliance : float
        Complex compliance (complex number) [Pa-1]
    """

    maxwell_complex_comp = maxwell(compliance, viscosity, frequency)

    # Update Zeta based on an additional frequency dependence.
    freq_ratio = critical_freq / frequency
    if freq_ratio > float_lognat_max:
        freq_ratio = float_lognat_max

    zeta *= np.exp(freq_ratio)

    andrade_term = compliance * viscosity * frequency * zeta
    const_term = compliance * andrade_term**(-alpha) * find_factorial(alpha)
    real_j = np.cos(alpha * np.pi / 2.) * const_term
    imag_j = -np.sin(alpha * np.pi / 2.) * const_term

    if np.abs(andrade_term) <= float_eps:
        imag_j = 0.

    andrade_complex_comp = real_j + 1.0j * imag_j

    complex_compliance = maxwell_complex_comp + andrade_complex_comp

    return complex_compliance

@njit
def sundberg(frequency: float, compliance: float, viscosity: float,
             voigt_compliance_offset: float = 0.2, voigt_viscosity_offset: float = 0.02,
             alpha: float = 0.3, zeta: float = 1.) -> complex:
    """ Calculates the complex compliance utilizing the model: Sundberg-Cooper

    !TPY_args live: self.compliance, self.viscosity
    !TPY_args const: voigt_compliance_offset, voigt_viscosity_offset, alpha, zeta

    Parameters
    ----------
    frequency : float
        Planet's tidal frequency [rads s-1]
        Note that a planet may experience multiple tidal frequencies for NSR tides or Fourier Degree l>2
    compliance : float
        Layer or Planet's compliance (inverse of shear modulus) [Pa-1]
    viscosity : float
        Layer or Planet's effective viscosity [Pa s]
    voigt_compliance_offset : float
        Voigt component's compliance offset eta_voigt = voigt_compliance_offset * compliance
    voigt_viscosity_offset : float
        Voigt component's viscosity offset eta_voigt = voigt_viscosity_offset * viscosity
    alpha : float
        Andrade exponent parameter
    zeta : float
        Andrade timescale parameter

    Returns
    -------
    complex_compliance : float
        Complex compliance (complex number) [Pa-1]
    """

    voigt_complex_comp = voigt(compliance, viscosity, frequency, voigt_compliance_offset, voigt_viscosity_offset)
    andrade_complex_comp = andrade(compliance, viscosity, frequency, alpha, zeta)

    complex_compliance = voigt_complex_comp + andrade_complex_comp

    return complex_compliance

@njit
def sundberg_freq(frequency: float, compliance: float, viscosity: float,
                  voigt_compliance_offset: float = 0.2, voigt_viscosity_offset: float = 0.02,
                  alpha: float = 0.3, zeta: float = 1., critical_freq: float = 2.e-5) -> complex:
    """ Calculates the complex compliance utilizing the model: Sundberg-Cooper with a frequency-dependent zeta

    !TPY_args live: self.compliance, self.viscosity
    !TPY_args const: voigt_compliance_offset, voigt_viscosity_offset, alpha, zeta, critical_freq

    Parameters
    ----------
    frequency : float
        Planet's tidal frequency [rads s-1]
        Note that a planet may experience multiple tidal frequencies for NSR tides or Fourier Degree l>2
    compliance : float
        Layer or Planet's compliance (inverse of shear modulus) [Pa-1]
    viscosity : float
        Layer or Planet's effective viscosity [Pa s]
    voigt_compliance_offset : float
        Voigt component's compliance offset eta_voigt = voigt_compliance_offset * compliance
    voigt_viscosity_offset : float
        Voigt component's viscosity offset eta_voigt = voigt_viscosity_offset * viscosity
    alpha : float
        Andrade exponent parameter
    zeta : float
        Andrade timescale parameter
    critical_freq : float
        For forcing frequencies larger than the critical frequency, the Andrade component converges to
            the Maxwell component [rads s-1]

    Returns
    -------
    complex_compliance : float
        Complex compliance (complex number) [Pa-1]
    """

    voigt_complex_comp = voigt(compliance, viscosity, frequency, voigt_compliance_offset, voigt_viscosity_offset)
    andrade_complex_comp = andrade_freq(compliance, viscosity, frequency, alpha, zeta, critical_freq)

    complex_compliance = voigt_complex_comp + andrade_complex_comp

    return complex_compliance
