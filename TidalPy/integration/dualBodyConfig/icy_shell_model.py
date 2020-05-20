import os

import numpy as np
import matplotlib.pyplot as plt

from ...constants import G
from ...tools.conversions import orbital_motion2semi_a, sec2myr
from ...utilities.performance import njit
from ...tides.mode_manipulation import build_mode_manipulators
from ...tides.dissipation import calc_tidal_susceptibility
from ...cooling.cooling_models import conduction, convection
from ...rheology.viscosity import known_models as known_viscosity_models
from ...rheology.complexCompliance import known_models as known_complex_compliance_models
from ...rheology.partialMelt import known_models as known_partial_melt_models

plt.rcParams.update({'font.size': 14})

MIN_INTERVAL_SCALE = 0.005
MAX_DATA_SIZE = 2000

def build_2layer_icy_shell_diffeq(obj0_config: dict, obj1_config: dict, orbital_config: dict, integration_config: dict):

    # Load planetary data
    object_configs = (obj0_config, obj1_config)
    object_names = tuple([object_config['name'] for object_config in object_configs])
    object_masses = tuple([object_config['mass'] for object_config in object_configs])
    object_radii = tuple([object_config['radius'] for object_config in object_configs])
    object_obliquities = tuple([object_config['constant_obliquity'] for object_config in object_configs])
    surface_temperatures = tuple([object_config['surface_temperature'] for object_config in object_configs])
    num_layers = tuple([int(len(object_config['layers'])) for object_config in object_configs])

    # Layer data are stored as the following: ( (obj0_layer_0, obj0_layer_1), (obj1_layer_0, obj1_layer1) )
    layer_names = tuple(
        [tuple([layer_name for layer_name, layer_config in object_config['layers'].items()])
        for object_config in object_configs]
    )
    layer_masses = tuple(
        [tuple([layer_config['mass'] for _, layer_config in object_config['layers'].items()])
        for object_config in object_configs]
    )
    layer_radii_upper = tuple(
        [tuple([layer_config['radius_upper'] for _, layer_config in object_config['layers'].items()])
        for object_config in object_configs]
    )
    layer_radii_lower = tuple(
        [tuple([layer_config['radius_lower'] for _, layer_config in object_config['layers'].items()])
        for object_config in object_configs]
    )
    material_densities = tuple(
        [tuple([layer_config['material_density'] for _, layer_config in object_config['layers'].items()])
         for object_config in object_configs]
    )
    thermal_conductivities = tuple(
        [tuple([layer_config['thermal_conductivity'] for _, layer_config in object_config['layers'].items()])
        for object_config in object_configs]
    )
    thermal_expansions = tuple(
        [tuple([layer_config['thermal_expansion'] for _, layer_config in object_config['layers'].items()])
        for object_config in object_configs]
    )
    specific_heats = tuple(
        [tuple([layer_config['specific_heat'] for _, layer_config in object_config['layers'].items()])
        for object_config in object_configs]
    )
    latent_heats = tuple(
            [tuple([layer_config['latent_heat'] for _, layer_config in object_config['layers'].items()])
             for object_config in object_configs]
    )
    convection_betas = tuple(
        [tuple([layer_config['convection_beta'] for _, layer_config in object_config['layers'].items()])
        for object_config in object_configs]
    )
    convection_alphas = tuple(
        [tuple([layer_config['convection_alpha'] for _, layer_config in object_config['layers'].items()])
        for object_config in object_configs]
    )
    critical_rayleighs = tuple(
        [tuple([layer_config['critical_rayleigh'] for _, layer_config in object_config['layers'].items()])
         for object_config in object_configs]
    )
    static_shears = tuple(
        [tuple([layer_config['static_shear'] for _, layer_config in object_config['layers'].items()])
        for object_config in object_configs]
    )
    viscosity_model_names = tuple(
        [tuple([layer_config['viscosity_model'] for _, layer_config in object_config['layers'].items()])
        for object_config in object_configs]
    )
    viscosity_inputs = tuple(
        [tuple([layer_config['viscosity_input'] for _, layer_config in object_config['layers'].items()])
        for object_config in object_configs]
    )
    partial_melt_model_names = tuple(
        [tuple([layer_config['partial_melt_model'] for _, layer_config in object_config['layers'].items()])
        for object_config in object_configs]
    )
    partial_melt_inputs = tuple(
        [tuple([layer_config['partial_melt_input'] for _, layer_config in object_config['layers'].items()])
        for object_config in object_configs]
    )
    solidus_temperature = tuple(
        [tuple([layer_config['solidus_temperature'] for _, layer_config in object_config['layers'].items()])
        for object_config in object_configs]
    )
    liquidus_temperature = tuple(
            [tuple([layer_config['liquidus_temperature'] for _, layer_config in object_config['layers'].items()])
             for object_config in object_configs]
    )
    rheology_model_names = tuple(
        [tuple([layer_config['rheology_model'] for _, layer_config in object_config['layers'].items()])
        for object_config in object_configs]
    )
    rheology_inputs = tuple(
        [tuple([layer_config['rheology_input'] for _, layer_config in object_config['layers'].items()])
        for object_config in object_configs]
    )
    tidal_scales = tuple(
        [tuple([layer_config.get('tidal_scale', 1.) for _, layer_config in object_config['layers'].items()])
        for object_config in object_configs]
    )
    growth_layer_flags = tuple(
        [tuple([layer_config['growth_layer'] for _, layer_config in object_config['layers'].items()])
         for object_config in object_configs]
    )
    constant_ocean_temperatures = tuple(
        [tuple([layer_config.get('constant_ocean_temperature', 100.) for _, layer_config in object_config['layers'].items()])
         for object_config in object_configs]
    )
    constant_viscoelastic_temperatures = tuple(
        [tuple([layer_config.get('constant_viscoelastic_temperature', 100.) for _, layer_config in object_config['layers'].items()])
         for object_config in object_configs]
    )
    viscoelastic_top_temperatures = tuple(
        [tuple([layer_config.get('viscoelastic_top_temperature', 100.) for _, layer_config in object_config['layers'].items()])
         for object_config in object_configs]
    )

    # Load model functions
    viscosity_funcs = tuple(
            [tuple([known_viscosity_models[_model_name] for _model_name in _layer_visco_models])
             for _layer_visco_models in viscosity_model_names]
    )
    complex_compliance_funcs = tuple(
            [tuple([known_complex_compliance_models[_model_name] for _model_name in _layer_rheology_models])
             for _layer_rheology_models in rheology_model_names]
    )
    partial_melt_funcs = tuple(
            [tuple([known_partial_melt_models[_model_name] for _model_name in _layer_partial_melt_models])
             for _layer_partial_melt_models in partial_melt_model_names]
    )

    # Load orbital configurations
    eccentricity_truncation = orbital_config['eccentricity_truncation']
    max_tidal_order_l = orbital_config['max_tidal_order_l']
    use_obliquity = orbital_config['use_obliquity']
    #    Build tidal mode manipulators
    calculate_tidal_terms, collapse_modes = \
        build_mode_manipulators(max_tidal_order_l, eccentricity_truncation, use_obliquity=use_obliquity)

    # Load integration configurations
    use_planetary_params_for_tide_calc = integration_config['use_planetary_params_for_tides']
    use_tidal_scale = integration_config['use_tidal_scale']
    use_visco_volume_for_tidal_scale = integration_config['use_visco_volume_for_tidal_scale']
    use_julia = integration_config['use_julia']
    time_span = integration_config['time_span']
    time_interval = time_span[-1] - time_span[0]

    # Calculate derived properties
    object_volumes = tuple([(4. / 3.) * np.pi * _radius**3 for _radius in object_radii])
    object_densities_bulk = tuple([_mass / _volume for _mass, _volume in zip(object_masses, object_volumes)])
    object_gravities = tuple([G * _mass / _radius**2 for _mass, _radius in zip(object_masses, object_radii)])
    layer_volumes = tuple(
        [tuple([(4. / 3.) * np.pi * (_radius_upper**3 - _radius_lower**3) for _radius_upper, _radius_lower
                in zip(_obj_radius_upper, _obj_radius_lower)])
         for _obj_radius_upper, _obj_radius_lower in zip(layer_radii_upper, layer_radii_lower)]
    )
    layer_thicknesses = tuple(
        [tuple([_radius_upper - _radius_lower for _radius_upper, _radius_lower
                in zip(_obj_radii_upper, _obj_radii_lower)])
         for _obj_radii_upper, _obj_radii_lower in zip(layer_radii_upper, layer_radii_lower)]
    )
    layer_surf_areas = tuple(
        [tuple([4. * np.pi * _radius_upper**2 for _radius_upper in _obj_radius_upper])
         for _obj_radius_upper in layer_radii_upper]
    )
    layer_masses_below = list()
    for _object_i in range(2):
        _n_layer = num_layers[_object_i]
        _mass = 0.
        _layer_masses_below_for_object = list()
        for _layer_i in range(_n_layer):
            _mass += layer_masses[_object_i][_layer_i]
            _layer_masses_below_for_object.append(_mass)
        layer_masses_below.append(tuple(_layer_masses_below_for_object))
    layer_masses_below = tuple(layer_masses_below)
    layer_densities_bulk = tuple(
        [tuple([_mass / _volume for _mass, _volume in zip(_obj_masses, _obj_volumes)])
         for _obj_masses, _obj_volumes in zip(layer_masses, layer_volumes)]
    )
    layer_gravities = tuple(
        [tuple([G * _mass_below / _radius_upper**2 for _mass_below, _radius_upper
                in zip(_obj_masses_below, _obj_upper_radii)])
         for _obj_masses_below, _obj_upper_radii in zip(layer_masses_below, layer_radii_upper)]
    )
    layer_betas = tuple(
        [tuple([_radius * _gravity * _density_bulk for _radius, _gravity, _density_bulk
                in zip(_obj_radii, _obj_gravities, _obj_densities)])
         for _obj_radii, _obj_gravities, _obj_densities in zip(layer_radii_upper, layer_gravities, layer_densities_bulk)]
    )
    thermal_diffusivities = tuple(
        [tuple([_k / (_rho * _c_p) for _k, _rho, _c_p in zip(_ks, _rhos, _c_ps)])
         for _ks, _rhos, _c_ps in zip(thermal_conductivities, material_densities, specific_heats)]
    )

    # We are going to be building several functions in this function. Make sure the namespace is as clean as possible.
    del _layer_i, _layer_masses_below_for_object, _mass, _n_layer, _object_i,

    @njit
    def diffeq_scipy(time, variables):

        # Progress bar
        percent_done = round(1000. * (time / time_interval)) / 1000.
        print('Percent Done:', percent_done, '%')
        # print('\rPercent Done: {:0>5.2f}%'.format(100. * percent_done), flush=True, end='')

        # Pull out independent variables
        temperatures = list()
        elastic_dxs = list()
        viscoelastic_dxs = list()
        spin_rates = list()
        index = 0
        for object_i in range(2):
            n_layers = num_layers[object_i]
            temperatures_by_layer = list()
            elastic_dx_by_layer = list()
            viscoelastic_dx_by_layer = list()
            for layer_i in range(n_layers):
                temperatures_by_layer.append(variables[index])
                elastic_dx_by_layer.append(variables[index + 1])
                viscoelastic_dx_by_layer.append(variables[index + 2])
                index += 3
            temperatures.append(temperatures_by_layer)
            elastic_dxs.append(elastic_dx_by_layer)
            viscoelastic_dxs.append(viscoelastic_dx_by_layer)
            # Next item is the spin-rate
            spin_rates.append(variables[index])
            index += 1
        orbital_motion = variables[index]
        eccentricity = variables[index + 1]

        # Calculate parameters that only depend on orbital properties
        # FIXME
        # semi_major_axis = orbital_motion2semi_a(orbital_motion, object_masses[0], object_masses[1])

        # Derivative storage
        derivative_storage = list()

        # Perform thermal evolution on each object's layers
        for object_i in range(2):
            number_of_layers = num_layers[object_i]

            # Need opposite object index for tidal calculations
            opposite_object_i = 1
            if object_i == 1:
                opposite_object_i = 0

            # Pull out parameters referenced often
            object_radius = object_radii[object_i]
            object_mass = object_masses[object_i]
            tidal_host_mass = object_masses[opposite_object_i]

            # # Calculate tidal modes and susceptibility
            # unique_frequencies, tidal_results_by_frequency = \
            #     calculate_tidal_terms(orbital_motion, spin_rate[object_i], eccentricity, object_obliquity[object_i],
            #                           semi_major_axis, obj_radius)
            # tidal_susceptibility = \
            #     calc_tidal_susceptibility(tidal_host_mass, obj_radius, semi_major_axis)

            # First loop through layers to determine geometry and find bottom temperatures which are used for next loop
            bottom_temperatures = list()
            layer_geometries = list()
            for layer_i in range(number_of_layers):
                bottom_layer = layer_i == 0
                top_layer = layer_i == num_layers[object_i] - 1
                # Pull out often used parameters
                #    State Properties
                viscoelastic_temperature = temperatures[object_i][layer_i]
                layer_radius_upper = layer_radii_upper[object_i][layer_i]
                layer_radius_lower = layer_radii_lower[object_i][layer_i]
                #    Layer Properties
                layer_mass_below = layer_masses_below[object_i][layer_i]
                #    Material Properties
                material_density = material_densities[object_i][layer_i]

                # Determine layer geometry
                layer_stagnant = False
                layer_freeze_out = False
                layer_lost_lid = False
                is_growth_layer = growth_layer_flags[object_i][layer_i]

                if is_growth_layer:
                    elastic_radius_upper = layer_radius_upper
                    elastic_radius_lower = layer_radius_upper - elastic_dxs[object_i][layer_i]

                    # Check if the layer is totally elastic or if the elastic layer is totally gone
                    if (elastic_radius_lower < layer_radius_lower + 1.) or \
                            abs(viscoelastic_temperature - viscoelastic_top_temperatures[object_i][layer_i]) < 1.:
                        elastic_radius_lower = layer_radius_lower
                        # Layer is all elastic. No viscoelastic portion
                        layer_stagnant = True

                    elif elastic_radius_lower > layer_radius_upper:
                        # Elastic layer is basically non-existent. Make it small so we don't have divide by zero issues
                        elastic_radius_lower = layer_radius_upper
                        elastic_radius_upper = layer_radius_upper
                        layer_lost_lid = True

                    if layer_stagnant:
                        # Viscoelastic layer is non-existent
                        viscoelastic_radius_lower = layer_radius_lower
                        viscoelastic_radius_upper = layer_radius_lower
                        if top_layer:
                            # TODO: this is not right, but we don't really care about the thermal evolution after freezeout.
                            viscoelastic_temperature = 0.5 * (surface_temperatures[object_i] +
                                                              constant_viscoelastic_temperatures[object_i][layer_i])
                        else:
                            # TODO: And this should really be a based off the bottom temperature of the above layer.
                            viscoelastic_temperature = constant_viscoelastic_temperatures[object_i][layer_i]
                        bottom_temperature = viscoelastic_temperature
                    else:
                        # Viscoelastic layer is present
                        viscoelastic_radius_upper = elastic_radius_lower
                        viscoelastic_radius_lower = elastic_radius_lower - viscoelastic_dxs[object_i][layer_i]
                        if viscoelastic_radius_lower < layer_radius_lower:
                            viscoelastic_radius_lower = layer_radius_lower
                            # Layer is all either elastic or viscoelastic. No ocean
                            ocean_radius_lower = layer_radius_lower
                            ocean_radius_upper = layer_radius_lower
                            layer_freeze_out = True
                            # Bottom temperature is now equal to the viscoelastic temperature
                            bottom_temperature = viscoelastic_temperature
                        else:
                            # Ocean layer is still present
                            ocean_radius_lower = layer_radius_lower
                            ocean_radius_upper = viscoelastic_radius_lower
                            # If the ocean layer is present then the viscoelastic temperature is a constant
                            viscoelastic_temperature = constant_viscoelastic_temperatures[object_i][layer_i]
                            # And bottom temperature is the ocean temperature
                            bottom_temperature = constant_ocean_temperatures[object_i][layer_i]

                    # Calculate derivec properties
                    elastic_thickness = elastic_radius_upper - elastic_radius_lower
                    elastic_volume = (4. / 3.) * np.pi * \
                        (elastic_radius_upper * elastic_radius_upper * elastic_radius_upper -
                         elastic_radius_lower * elastic_radius_lower * elastic_radius_lower)
                    elastic_surf_area = 4. * np.pi * (elastic_radius_upper * elastic_radius_upper)
                    elastic_mass = elastic_volume * material_density
                    elastic_mass_below = layer_mass_below
                    elastic_gravity = G * elastic_mass_below / \
                                      (elastic_radius_upper * elastic_radius_upper)

                    viscoelastic_thickness = viscoelastic_radius_upper - viscoelastic_radius_lower
                    viscoelastic_volume = (4. / 3.) * np.pi * \
                        (viscoelastic_radius_upper * viscoelastic_radius_upper * viscoelastic_radius_upper -
                         viscoelastic_radius_lower * viscoelastic_radius_lower * viscoelastic_radius_lower)
                    viscoelastic_surf_area = 4. * np.pi * (viscoelastic_radius_upper * viscoelastic_radius_upper)
                    viscoelastic_mass = viscoelastic_volume * material_density
                    viscoelastic_mass_below = layer_mass_below - elastic_mass
                    viscoelastic_gravity = G * viscoelastic_mass_below / \
                                           (viscoelastic_radius_upper * viscoelastic_radius_upper)

                else:
                    # For the non-growth model, the entire layer is assumed to be viscoelastic (no stagnant lid)
                    viscoelastic_radius_lower = layer_radii_lower[object_i][layer_i]
                    viscoelastic_radius_upper = layer_radii_upper[object_i][layer_i]
                    viscoelastic_thickness = layer_thicknesses[object_i][layer_i]
                    viscoelastic_volume = layer_volumes[object_i][layer_i]
                    viscoelastic_surf_area = layer_surf_areas[object_i][layer_i]
                    viscoelastic_mass = layer_masses[object_i][layer_i]
                    viscoelastic_gravity = layer_gravities[object_i][layer_i]

                    # Elastic parameters will not be used.
                    elastic_radius_lower = 0.
                    elastic_radius_upper = 0.
                    elastic_thickness = 0.
                    elastic_volume = 0.
                    elastic_surf_area = 0.
                    elastic_mass = 0.
                    elastic_gravity = 0.

                    # Determine the temperature at the base of the layer (used for other layer's thermal evolution)
                    bottom_temperature = viscoelastic_temperature

                if use_tidal_scale:
                    if use_visco_volume_for_tidal_scale:
                        layer_tidal_scale = viscoelastic_volume / object_volumes[object_i]
                    else:
                        layer_tidal_scale = tidal_scales[object_i][layer_i]
                else:
                    layer_tidal_scale = 1.

                bottom_temperatures.append(bottom_temperature)
                layer_geometries.append(
                        (layer_tidal_scale, layer_freeze_out, layer_stagnant, layer_lost_lid,
                         viscoelastic_radius_lower, viscoelastic_radius_upper, viscoelastic_thickness,
                         viscoelastic_volume, viscoelastic_surf_area, viscoelastic_mass, viscoelastic_gravity,
                         viscoelastic_temperature,
                         elastic_radius_lower, elastic_radius_upper, elastic_thickness,
                         elastic_volume, elastic_surf_area, elastic_mass, elastic_gravity)
                )

            # Storage for various parameters that need to be accessed during the layer looping
            layer_coolings = list()

            # Now that layer's bottom temperatures are known we can loop through again and calculate everything else.
            for layer_i in range(number_of_layers):
                bottom_layer = layer_i == 0
                top_layer = layer_i == num_layers[object_i] - 1
                is_growth_layer = growth_layer_flags[object_i][layer_i]

                # Pull out often used parameters
                #    Material Properties
                material_density = material_densities[object_i][layer_i]
                thermal_conductivity = thermal_conductivities[object_i][layer_i]
                thermal_expansion = thermal_expansions[object_i][layer_i]
                thermal_diffusivity = thermal_diffusivities[object_i][layer_i]
                specific_heat = specific_heats[object_i][layer_i]
                latent_heat = latent_heats[object_i][layer_i]
                solidus = solidus_temperature[object_i][layer_i]
                liquidus = liquidus_temperature[object_i][layer_i]
                #    Convection Properties
                alpha_conv = convection_alphas[object_i][layer_i]
                beta_conv = convection_betas[object_i][layer_i]
                critical_rayleigh = critical_rayleighs[object_i][layer_i]

                # Pull out geometry information that was just calculated in the previous loop
                layer_tidal_scale, layer_freeze_out, layer_stagnant, layer_lost_lid, \
                viscoelastic_radius_lower, viscoelastic_radius_upper, viscoelastic_thickness, \
                viscoelastic_volume, viscoelastic_surf_area, viscoelastic_mass, viscoelastic_gravity, \
                viscoelastic_temperature, \
                elastic_radius_lower, elastic_radius_upper, elastic_thickness, \
                elastic_volume, elastic_surf_area, elastic_mass, elastic_gravity = \
                    layer_geometries[layer_i]

                elastic_passes_all_heat = False
                if layer_lost_lid:
                    elastic_passes_all_heat = True

                viscoelastic_passes_all_heat = False
                # Calculate viscoelastic portion's strength
                if viscoelastic_volume != 0.:
                #    TODO: Assume 0. pressure dependence for now (the zero in the input below)
                    pre_melt_viscosity = \
                        viscosity_funcs[object_i][layer_i](viscoelastic_temperature, 0., *viscosity_inputs[object_i][layer_i])
                    pre_melt_shear = static_shears[object_i][layer_i]
                    #    Apply partial melting
                    melt_fraction = (viscoelastic_temperature - solidus) / (liquidus - solidus)
                    if melt_fraction < 0.:
                        melt_fraction = 0.
                    elif melt_fraction > 1.:
                        melt_fraction = 1.
                    viscosity, shear_modulus = \
                        partial_melt_funcs[object_i][layer_i](melt_fraction, pre_melt_viscosity, pre_melt_shear,
                                                              *partial_melt_inputs[object_i][layer_i])
                    #    Check for over/undershoots
                    if viscosity < 0.1:
                        viscosity = 0.1
                    elif viscosity > 1.e100:
                        viscosity = 1.e100
                    if shear_modulus < 0.1:
                        shear_modulus = 0.1
                    elif shear_modulus > 1.e100:
                        shear_modulus = 1.e100
                    compliance = 1. / shear_modulus

                    # Calculate viscoelastic cooling
                    if is_growth_layer:
                        if layer_lost_lid:
                            if top_layer:
                                visco_top_temp = surface_temperatures[object_i]
                            else:
                                visco_top_temp = bottom_temperatures[layer_i + 1]
                        else:
                            visco_top_temp = viscoelastic_top_temperatures[object_i][layer_i]
                    else:
                        if top_layer:
                            visco_top_temp = surface_temperatures[object_i]
                        else:
                            visco_top_temp = bottom_temperatures[layer_i + 1]

                    #    Covective cooling for viscoelastic layer
                    viscoelastic_temperature_delta = viscoelastic_temperature - visco_top_temp
                    viscoelastic_cooling_flux, viscoelastic_boundary_layer_thickness, viscoelastic_rayleigh, \
                    viscoelastic_nusselt = \
                        convection(viscoelastic_temperature_delta, viscosity, thermal_conductivity,
                                   thermal_diffusivity, thermal_expansion,
                                   viscoelastic_thickness, viscoelastic_gravity, material_density, alpha_conv,
                                   beta_conv, critical_rayleigh)
                    viscoelastic_cooling = viscoelastic_surf_area * viscoelastic_cooling_flux

                    #    Calculate complex compliance based on layer's strength and the unique tidal forcing frequencies

                    # unique_complex_compliances = \
                    #     {freq_sig: complex_compliance_func(freq, compliance, viscosity, *rheology_input[object_i][layer_i])
                    #      for freq_sig, freq in unique_frequencies.items()}
                    #
                    # # Calculate tidal dissipation
                    # if use_planetary_params_for_tide_calc:
                    #     _radius = object_radius[object_i]
                    #     _gravity = object_gravity[object_i]
                    #     _density = object_density_bulk[object_i]
                    # else:
                    #     _radius = radius_upper[object_i][layer_i]
                    #     _gravity = layer_gravity[object_i][layer_i]
                    #     _density = layer_density_bulk[object_i][layer_i]
                    # tidal_heating, dUdM, dUdw, dUdO, love_number, negative_imk = \
                    #     collapse_modes(_gravity, _radius, _density, shear_modulus, unique_complex_compliances,
                    #                    tidal_results_by_frequency, tidal_susceptibility, tidal_host_mass, tidal_scale=,
                    #                    cpl_ctl_method=False)

                    # FIXME
                    tidal_heating = 0.

                else:
                    # No viscoelastic layer present. No tides or convection
                    viscosity = 0.
                    shear_modulus = 0.
                    melt_fraction = 0.
                    compliance = 0.
                    tidal_heating = 0.
                    visco_top_temp = bottom_temperatures[layer_i]

                    viscoelastic_temperature_delta = 0.
                    viscoelastic_cooling = 0.
                    viscoelastic_cooling_flux = 0.
                    viscoelastic_passes_all_heat = True

                # Calculate radiogenic heating
                # FIXME
                radiogenic_heating = 0.

                # Determine Heating
                total_incoming_heating = tidal_heating + radiogenic_heating
                if not bottom_layer:
                    total_incoming_heating += layer_coolings[layer_i - 1]

                # Determine thermal evolution model: growing layer or static
                if is_growth_layer:

                    # Determine heating fluxes
                    viscoelastic_heating_flux = total_incoming_heating / viscoelastic_surf_area

                    if viscoelastic_passes_all_heat:
                        viscoelastic_cooling_flux = viscoelastic_heating_flux

                    elastic_heating_flux = viscoelastic_cooling_flux

                    if elastic_passes_all_heat:
                        elastic_cooling_flux = viscoelastic_cooling_flux
                        elastic_cooling = viscoelastic_cooling_flux * elastic_surf_area
                        elastic_boundary_layer_thickness = 0.
                        elastic_rayleigh = 0.
                        elastic_nusselt = 1.
                    else:
                        # Determine Cooling for the elastic layer
                        if top_layer:
                            elastic_delta_temperature = visco_top_temp - surface_temperatures[object_i]
                        else:
                            elastic_delta_temperature = visco_top_temp - temperatures[object_i][layer_i + 1]
                        #     conduction function
                        elastic_cooling_flux, elastic_boundary_layer_thickness, elastic_rayleigh, elastic_nusselt = \
                            conduction(elastic_delta_temperature, thermal_conductivity, elastic_thickness)
                        elastic_cooling = elastic_surf_area * elastic_cooling_flux

                    if layer_stagnant:
                        # If the layer is stagnant (all elastic) then there is no change in temperature.
                        elastic_layer_change = 0.
                        viscoelastic_layer_change = 0.
                        viscoelastic_temperature_change = 0.
                    else:
                        energy_to_freeze = specific_heat * viscoelastic_temperature_delta
                        elastic_layer_change = \
                            (elastic_cooling_flux - elastic_heating_flux) / (material_density * energy_to_freeze)
                        if layer_freeze_out:
                            # The viscoelastic layer's temperature is allowed to decrease if the layer is frozen out.
                            viscoelastic_layer_change = -elastic_layer_change
                            viscoelastic_temperature_change = \
                                (total_incoming_heating - viscoelastic_cooling) / \
                                (viscoelastic_volume * material_density * specific_heat)
                        else:
                            viscoelastic_temperature_change = 0.
                            # If there is an ocean then the viscoelastic layer will change depending on the
                            #     energy balance
                            viscoelastic_layer_change = \
                                (viscoelastic_cooling_flux - viscoelastic_heating_flux) / \
                                (material_density * latent_heat)

                    # Heat coming out of the layer is equal to the elastic cooling
                    layer_cooling = elastic_cooling

                else:
                    # Static layer has no viscoelastic/elastic layer growth. Only the average temperature is tracked.
                    viscoelastic_temperature_change = \
                        (total_incoming_heating - viscoelastic_cooling) / \
                        (viscoelastic_volume * material_density * specific_heat)

                    elastic_layer_change = 0.
                    viscoelastic_layer_change = 0.
                    layer_cooling = viscoelastic_cooling

                # Store results and derivatives
                layer_coolings.append(layer_cooling)
                derivative_storage.append(viscoelastic_temperature_change)
                derivative_storage.append(elastic_layer_change)
                derivative_storage.append(viscoelastic_layer_change)


            # Determine change in this object's spin-rate
            # FIXME:
            spin_rate_change = 0.
            derivative_storage.append(spin_rate_change)

        # Determine orbital changes
        # FIXME:
        eccentricity_change = 0.
        semi_major_axis_change = 0.
        orbital_motion_change = 0.
        derivative_storage.append(orbital_motion_change)
        derivative_storage.append(eccentricity_change)

        return derivative_storage


    def diffeq_julia(variables, parameters, time):

        output = diffeq_scipy(time, variables)

        return output

    if use_julia:
        diffeq = diffeq_julia
    else:
        diffeq = diffeq_scipy


    def plotter(variables, time_domain, logtime: bool = False, save_locale: str = None):

        # Plot Styles
        object_colors = ('r', 'b')

        # Saving information
        run_save_name = object_names[0].lower() + '_' + object_names[1].lower()
        if save_locale is None:
            save_locale = os.getcwd()
        else:
            if not os.path.exists(save_locale):
                os.makedirs(save_locale)

        # Setup x-axis
        if logtime:
            x_label = 'Time [yr]'
            x = time_domain / 3.154e7
            x_scale = 'log'
        else:
            x_label = 'Time [Myr]'
            x = time_domain / 3.154e13
            x_scale = 'linear'

        # Figure 1: Orbital Motion & Spin Plot
        orbspin_fig, orbspin_axes = plt.subplots(ncols=2, figsize=(6.4*1.75, 4.8), tight_layout=True)
        ax_semia = orbspin_axes[0]
        ax_spin = orbspin_axes[1]
        for ax in [ax_semia, ax_spin]:
            ax.set(xscale=x_scale, xlabel=x_label)
        ax_eccen = ax_semia.twinx()
        ax_semia.set(ylabel='Semi-major Axis [km]')
        ax_eccen.set(ylabel='Eccentricity (dotted)')
        ax_spin.set(ylabel='Spin Period [days]')

        # Plot non-object dependent parameters
        semi_major_axis = variables['semi_major_axis'] / 1000.
        eccentricity = variables['eccentricity']
        ax_semia.plot(x, semi_major_axis, ls='-')
        ax_eccen.plot(x, eccentricity, ls=':')

        # Plot object dependent parameters
        planet_figures = list()
        for object_i, object_name in enumerate(object_names):
            object_variables = variables[object_name]
            color = object_colors[object_i]

            # Plot spin period
            spin_period = (2. * np.pi / (object_variables['spin_rate'])) / 86400.
            ax_spin.plot(x, spin_period, label=object_name, ls='-', c=color)

            # Plot layer properties
            n_layers = num_layers[object_i]
            width_factor = (1.75 / 2.) * n_layers
            layer_fig, layer_axes = plt.subplots(ncols=n_layers, figsize=(6.4*width_factor, 4.8), tight_layout=True)
            layer_fig.suptitle(object_name)
            growth_model_encountered = False
            for layer_i, layer_name in enumerate(layer_names[object_i]):
                ax = layer_axes[layer_i]
                ax.set(title=layer_name)
                layer_variables = object_variables[layer_name]

                # Determine if this is a growth or regular layer
                use_growth_model = growth_layer_flags[object_i][layer_i]
                if use_growth_model:
                    ax.set(ylabel='Thickness [layer %]')
                    layer_thickness = layer_thicknesses[object_i][layer_i]
                    elastic_thickness = layer_variables['elastic_thickness']
                    viscoelastic_thickness = layer_variables['viscoelastic_thickness']
                    ocean_thickness = layer_thicknesses[object_i][layer_i] - \
                                      (elastic_thickness + viscoelastic_thickness)
                    ax.plot(x, 100 * elastic_thickness / layer_thickness, c=color, ls='-', label='Elastic')
                    ax.plot(x, 100 * viscoelastic_thickness / layer_thickness, c=color, ls='--', label='Viscoelastic')
                    ax.plot(x, 100 * ocean_thickness / layer_thickness, c=color, ls=':', label='Ocean')

                    if not growth_model_encountered:
                        ax.legend(loc='best')
                    growth_model_encountered = True
                else:
                    ax.set(ylabel='Temperature [K]')
                    temperature = layer_variables['temperature']
                    ax.plot(x, temperature, c=color, ls='-')

            # Finish up this object's layer plot
            layer_fig.savefig(os.path.join(save_locale, f'{run_save_name}_{object_name}_LayerPlot.pdf'))
            planet_figures.append(layer_fig)

        # Finish up Orbital & Spin Figure
        ax_spin.legend(loc='best')
        orbspin_fig.savefig(os.path.join(save_locale, f'{run_save_name}_OrbitalSpin.pdf'))

        plt.show()

        return orbspin_fig, planet_figures


    def integrator(initial_conditions, diffeq_to_use=diffeq,
                   integration_rtol: float = 1.e-8,
                   auto_plot: bool = True, save_data: bool = False,
                   save_locale: str = None, **plotter_kwargs):

        # Determine save location
        run_save_name = object_names[0].lower() + '_' + object_names[1].lower()
        if save_locale is None:
            save_locale = os.getcwd()
        else:
            if not os.path.exists(save_locale):
                os.makedirs(save_locale)

        result_dict = dict()
        print('Integrating Dual Body System:')
        print(f'\t{object_names[0]} and {object_names[1]}')
        if use_julia:
            print('\tUsing Julia Diffeq...')

            # Import Julia's Diffeqpy and setup the problem
            from diffeqpy import de
            min_interval = MIN_INTERVAL_SCALE * time_interval
            problem = de.ODEProblem(diffeq_to_use, initial_conditions, time_span)
            print(f'Solving...')
            solution = de.solve(problem, de.BS3(), saveat=min_interval, abstol=1e-8, reltol=integration_rtol)
            print('\nIntegration Done!')

            y = np.transpose(solution.u)
            t = solution.t

            del solution
        else:
            from scipy.integrate import solve_ivp

            solution = solve_ivp(diffeq_to_use, time_span, initial_conditions,
                                 method='LSODA', vectorized=False)

            # Pull out dependent variables
            if len(solution.t) > MAX_DATA_SIZE:
                print('Solution data size is very large. Reducing to avoid memory errors in auxiliary grab.')
                t = np.linspace(solution.t[0], solution.t[-1], MAX_DATA_SIZE)
                y = np.asarray([np.interp(t, solution.t, solution.y[i, :])
                                         for i in range(solution.y.shape[0])])
            else:
                t = solution.t
                y = solution.y
            del solution

        # Pull out independent variables
        starting_index = 0
        for object_i, object_name in enumerate(object_names):
            result_dict[object_name] = dict()
            for layer_i, layer_name in enumerate(layer_names[object_i]):
                result_dict[object_name][layer_name] = dict()
                result_dict[object_name][layer_name]['temperature'] = \
                    y[starting_index + 0, :]
                result_dict[object_name][layer_name]['elastic_thickness'] = \
                    y[starting_index + 1, :]
                result_dict[object_name][layer_name]['viscoelastic_thickness'] = \
                    y[starting_index + 2, :]
                starting_index += 3
            result_dict[object_name]['spin_rate'] = y[starting_index, :]
            starting_index += 1
        result_dict['orbital_motion'] = y[starting_index, :]
        result_dict['semi_major_axis'] = \
            orbital_motion2semi_a(result_dict['orbital_motion'], object_masses[0], object_masses[1])
        result_dict['eccentricity'] = y[starting_index + 1, :]
        time_domain = t

        del y

        if save_data:
            data_save_dir = save_locale
            print(f'Saving Data to:\n\t{data_save_dir}')
            np.save(os.path.join(data_save_dir, f'{run_save_name}_TimeDomainMyr.npy'), time_domain)
            for object_i, object_name in enumerate(object_names):
                object_results = result_dict[object_name]
                np.save(os.path.join(data_save_dir, f'{run_save_name}_{object_name}_TimeDomainMyr.npy'),
                        object_results['spin_rate'])
                for layer_name in layer_names[object_i]:
                    layer_results = result_dict[object_name][layer_name]
                    for result_name, result in layer_results.items():
                        np.save(os.path.join(data_save_dir,
                                             f'{run_save_name}_{object_name}_{layer_name}_{result_name}.npy'),
                                result)
            print('Data Saved.')

        if auto_plot:
            print('Calling Plotter...')
            plotter(result_dict, time_domain, save_locale=save_locale, **plotter_kwargs)
            print('Plotter Finished.')


    return diffeq, integrator, plotter