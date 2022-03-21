""" Tests for extracting useful information out of a multilayer tidal propagation
"""

import numpy as np

import TidalPy
from TidalPy.constants import G
from TidalPy.tides.potential import tidal_potential_nsr_modes, tidal_potential_obliquity_nsr_modes, tidal_potential_nsr, tidal_potential_obliquity_nsr
from TidalPy.toolbox.conversions import orbital_motion2semi_a

TidalPy.config['stream_level'] = 'ERROR'
TidalPy.use_disk = False
TidalPy.reinit()

# Model planet - 2layers
density_array = 5000. * np.ones(10)
radius_array = np.linspace(0., 1.e6, 11)
longitude_array = np.radians(np.linspace(0., 360., 12))
colat_array = np.radians(np.linspace(0.5, 179.5, 13))

volume_array = (4. / 3.) * np.pi * (radius_array[1:]**3 - radius_array[:-1]**3)
mass_array = volume_array * density_array
planet_mass = sum(mass_array)
mass_below = np.asarray([np.sum(mass_array[:i + 1]) for i in range(10)])
gravity_array = G * mass_below / (radius_array[1:]**2)
shear_array = 5.e10 * np.ones(10, dtype=np.complex)
host_mass = 50000. * planet_mass
orbital_freq = (2. * np.pi / (86400. * 6.))
semi_major_axis = orbital_motion2semi_a(orbital_freq, host_mass, planet_mass)
eccentricity = 0.3
obliquity = np.radians(15.)

time_array = np.linspace(0., 2. * np.pi / orbital_freq, 5)

long_mtx, colat_mtx, time_mtx = np.meshgrid(longitude_array, colat_array, time_array)

def test_tidal_potential_nsr_modes():
    """ Test the modal tidal potential assuming moderate eccentricity, no obliquity, and synchronous rotation """
    # Test arrays - Static=False
    spin_freq = 1.5 * orbital_freq
    freqs, modes, potential_dict, potential_dtheta_dict, potential_dphi_dict, potential_d2theta_dict, \
        potential_d2phi_dict, potential_dtheta_dphi_dict = \
        tidal_potential_nsr_modes(
            radius_array[-1], long_mtx, colat_mtx, time_mtx,
            orbital_freq, spin_freq,
            eccentricity,
            host_mass, semi_major_axis,
            use_static=False)

    # Check mode frequency types
    assert len(modes) == 9
    assert len(freqs) == 9
    for mode_label, mode_freq in modes.items():
        assert type(mode_label) == str
        assert type(mode_freq) in [float, np.float, np.float64]

        assert type(potential_dict[mode_label]) == np.ndarray
        assert type(potential_dtheta_dict[mode_label]) == np.ndarray
        assert type(potential_dphi_dict[mode_label]) == np.ndarray
        assert type(potential_d2theta_dict[mode_label]) == np.ndarray
        assert type(potential_d2phi_dict[mode_label]) == np.ndarray
        assert type(potential_dtheta_dphi_dict[mode_label]) == np.ndarray

        assert potential_dict[mode_label].shape == long_mtx.shape
        assert potential_dtheta_dict[mode_label].shape == long_mtx.shape
        assert potential_dphi_dict[mode_label].shape == long_mtx.shape
        assert potential_d2theta_dict[mode_label].shape == long_mtx.shape
        assert potential_d2phi_dict[mode_label].shape == long_mtx.shape
        assert potential_dtheta_dphi_dict[mode_label].shape == long_mtx.shape

        assert type(potential_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_dtheta_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_dphi_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_d2theta_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_d2phi_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_dtheta_dphi_dict[mode_label][0, 0, 0]) in [float, np.float64]

    # Test arrays - Static=True
    spin_freq = 1.5 * orbital_freq
    freqs, modes, potential_dict, potential_dtheta_dict, potential_dphi_dict, potential_d2theta_dict, \
    potential_d2phi_dict, potential_dtheta_dphi_dict = \
        tidal_potential_nsr_modes(
            radius_array[-1], long_mtx, colat_mtx, time_mtx,
            orbital_freq, spin_freq,
            eccentricity,
            host_mass, semi_major_axis,
            use_static=True
            )

    # Check mode frequency types
    assert len(modes) == 9
    assert len(freqs) == 9
    for mode_label, mode_freq in modes.items():
        assert type(mode_label) == str
        assert type(mode_freq) in [float, np.float, np.float64]

        assert type(potential_dict[mode_label]) == np.ndarray
        assert type(potential_dtheta_dict[mode_label]) == np.ndarray
        assert type(potential_dphi_dict[mode_label]) == np.ndarray
        assert type(potential_d2theta_dict[mode_label]) == np.ndarray
        assert type(potential_d2phi_dict[mode_label]) == np.ndarray
        assert type(potential_dtheta_dphi_dict[mode_label]) == np.ndarray

        assert potential_dict[mode_label].shape == long_mtx.shape
        assert potential_dtheta_dict[mode_label].shape == long_mtx.shape
        assert potential_dphi_dict[mode_label].shape == long_mtx.shape
        assert potential_d2theta_dict[mode_label].shape == long_mtx.shape
        assert potential_d2phi_dict[mode_label].shape == long_mtx.shape
        assert potential_dtheta_dphi_dict[mode_label].shape == long_mtx.shape

        assert type(potential_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_dtheta_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_dphi_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_d2theta_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_d2phi_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_dtheta_dphi_dict[mode_label][0, 0, 0]) in [float, np.float64]

def test_tidal_potential_obliquity_nsr_modes():
    """ Test the modal tidal potential assuming moderate eccentricity, moderate obliquity, and synchronous rotation """
    # Test arrays - Static=False
    spin_freq = 1.5 * orbital_freq
    freqs, modes, potential_dict, potential_dtheta_dict, potential_dphi_dict, potential_d2theta_dict, \
        potential_d2phi_dict, potential_dtheta_dphi_dict = \
        tidal_potential_obliquity_nsr_modes(
            radius_array[-1], long_mtx, colat_mtx, time_mtx,
            orbital_freq, spin_freq,
            eccentricity, obliquity,
            host_mass, semi_major_axis,
            use_static=False)

    # Check mode frequency types
    assert len(modes) == 17
    assert len(freqs) == 17
    for mode_label, mode_freq in modes.items():
        assert type(mode_label) == str
        assert type(mode_freq) in [float, np.float, np.float64]

        assert type(potential_dict[mode_label]) == np.ndarray
        assert type(potential_dtheta_dict[mode_label]) == np.ndarray
        assert type(potential_dphi_dict[mode_label]) == np.ndarray
        assert type(potential_d2theta_dict[mode_label]) == np.ndarray
        assert type(potential_d2phi_dict[mode_label]) == np.ndarray
        assert type(potential_dtheta_dphi_dict[mode_label]) == np.ndarray

        assert potential_dict[mode_label].shape == long_mtx.shape
        assert potential_dtheta_dict[mode_label].shape == long_mtx.shape
        assert potential_dphi_dict[mode_label].shape == long_mtx.shape
        assert potential_d2theta_dict[mode_label].shape == long_mtx.shape
        assert potential_d2phi_dict[mode_label].shape == long_mtx.shape
        assert potential_dtheta_dphi_dict[mode_label].shape == long_mtx.shape

        assert type(potential_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_dtheta_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_dphi_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_d2theta_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_d2phi_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_dtheta_dphi_dict[mode_label][0, 0, 0]) in [float, np.float64]

    # Test arrays - Static=True
    spin_freq = 1.5 * orbital_freq
    freqs, modes, potential_dict, potential_dtheta_dict, potential_dphi_dict, potential_d2theta_dict, \
    potential_d2phi_dict, potential_dtheta_dphi_dict = \
        tidal_potential_obliquity_nsr_modes(
            radius_array[-1], long_mtx, colat_mtx, time_mtx,
            orbital_freq, spin_freq,
            eccentricity, obliquity,
            host_mass, semi_major_axis,
            use_static=True
            )

    # Check mode frequency types
    assert len(modes) == 17
    assert len(freqs) == 17
    for mode_label, mode_freq in modes.items():
        assert type(mode_label) == str
        assert type(mode_freq) in [float, np.float, np.float64]

        assert type(potential_dict[mode_label]) == np.ndarray
        assert type(potential_dtheta_dict[mode_label]) == np.ndarray
        assert type(potential_dphi_dict[mode_label]) == np.ndarray
        assert type(potential_d2theta_dict[mode_label]) == np.ndarray
        assert type(potential_d2phi_dict[mode_label]) == np.ndarray
        assert type(potential_dtheta_dphi_dict[mode_label]) == np.ndarray

        assert potential_dict[mode_label].shape == long_mtx.shape
        assert potential_dtheta_dict[mode_label].shape == long_mtx.shape
        assert potential_dphi_dict[mode_label].shape == long_mtx.shape
        assert potential_d2theta_dict[mode_label].shape == long_mtx.shape
        assert potential_d2phi_dict[mode_label].shape == long_mtx.shape
        assert potential_dtheta_dphi_dict[mode_label].shape == long_mtx.shape

        assert type(potential_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_dtheta_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_dphi_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_d2theta_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_d2phi_dict[mode_label][0, 0, 0]) in [float, np.float64]
        assert type(potential_dtheta_dphi_dict[mode_label][0, 0, 0]) in [float, np.float64]

def test_tidal_potential_nsr_modes_vs_non_mode():
    """ Test the modal tidal potential assuming moderate eccentricity, no obliquity, and synchronous rotation vs.
        the non-modal version"""
    # First find the mode version
    # Test arrays - Static=False
    spin_freq = 1.5 * orbital_freq
    freqs, modes, potential_modes, potential_dtheta_modes, potential_dphi_modes, potential_d2theta_modes, \
        potential_d2phi_modes, potential_dtheta_dphi_modes = \
        tidal_potential_nsr_modes(
            radius_array[-1], long_mtx, colat_mtx, time_mtx,
            orbital_freq, spin_freq,
            eccentricity,
            host_mass, semi_major_axis,
            use_static=False)

    # Then calculate the non-mode version.
    potential, potential_partial_theta, potential_partial_phi, \
    potential_partial2_theta2, potential_partial2_phi2, potential_partial2_theta_phi = \
        tidal_potential_nsr(
            radius_array[-1], longitude=long_mtx, colatitude=colat_mtx, time=time_mtx,
            orbital_frequency=orbital_freq, rotation_frequency=spin_freq,
            eccentricity=eccentricity,
            host_mass=host_mass, semi_major_axis=semi_major_axis,
            use_static=False
            )

    for mode_pot, reg_pot in [(potential_modes, potential),
                               (potential_dtheta_modes, potential_partial_theta),
                               (potential_dphi_modes, potential_partial_phi),
                               (potential_d2theta_modes, potential_partial2_theta2),
                               (potential_d2phi_modes, potential_partial2_phi2),
                               (potential_dtheta_dphi_modes, potential_partial2_theta_phi)]:

        # We need to collapse the mode version so that it matches the non-mode version.
        # They will only match if we make the CPL assumption (linearly add the modes together).
        collapsed_mode_pot = sum(list(mode_pot.values()))

        assert collapsed_mode_pot.shape == reg_pot.shape
        assert np.allclose(collapsed_mode_pot, reg_pot)

def test_tidal_potential_obliquity_nsr_modes_vs_non_mode():
    """ Test the modal tidal potential assuming moderate eccentricity, moderate obliquity, and synchronous rotation vs.
        the non-modal version"""
    # First find the mode version
    # Test arrays - Static=False
    spin_freq = 1.5 * orbital_freq
    freqs, modes, potential_modes, potential_dtheta_modes, potential_dphi_modes, potential_d2theta_modes, \
        potential_d2phi_modes, potential_dtheta_dphi_modes = \
        tidal_potential_obliquity_nsr_modes(
            radius_array[-1], long_mtx, colat_mtx, time_mtx,
            orbital_freq, spin_freq,
            eccentricity, obliquity,
            host_mass, semi_major_axis,
            use_static=False)

    # Then calculate the non-mode version.
    potential, potential_partial_theta, potential_partial_phi, \
    potential_partial2_theta2, potential_partial2_phi2, potential_partial2_theta_phi = \
        tidal_potential_obliquity_nsr(
            radius_array[-1], longitude=long_mtx, colatitude=colat_mtx, time=time_mtx,
            orbital_frequency=orbital_freq, rotation_frequency=spin_freq,
            eccentricity=eccentricity, obliquity=obliquity,
            host_mass=host_mass, semi_major_axis=semi_major_axis,
            use_static=False
            )

    for mode_pot, reg_pot in [(potential_modes, potential),
                               (potential_dtheta_modes, potential_partial_theta),
                               (potential_dphi_modes, potential_partial_phi),
                               (potential_d2theta_modes, potential_partial2_theta2),
                               (potential_d2phi_modes, potential_partial2_phi2),
                               (potential_dtheta_dphi_modes, potential_partial2_theta_phi)]:

        # We need to collapse the mode version so that it matches the non-mode version.
        # They will only match if we make the CPL assumption (linearly add the modes together).
        collapsed_mode_pot = sum(list(mode_pot.values()))

        assert collapsed_mode_pot.shape == reg_pot.shape
        assert np.allclose(collapsed_mode_pot, reg_pot)