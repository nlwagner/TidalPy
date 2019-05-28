from TidalPy.exceptions import MissingArgumentError, UnknownModelError
from TidalPy.io import unique_path
from TidalPy.structures.worlds import world_types, BurnManWorld
from TidalPy.utilities.pathing import get_all_files_of_type
from .dilled_planets import dilled_planets_loc
from .planet_configs import planet_config_loc
import warnings
import dill
from .. import log, other_data_locs
import json5
from ..bm.build import build_planet as build_bm_planet
import os

def check_for_duplicates(dict_to_check: dict):

    potential_dups = list()
    for planet_name, planet_filepath in dict_to_check.items():
        possible_dup_index = planet_name.split('_')
        try:
            int(possible_dup_index)
        except ValueError:
            continue
        else:
            if planet_name not in potential_dups:
                potential_dups.append(planet_name)

    for planet_name in potential_dups:
        warnings.warn(f'Possible duplicate saved planet found: {planet_name}. Ensure you use the correct subscript.')


# Locate all dilled planets
known_planets_dill = dict()
for potential_data_location in other_data_locs + [dilled_planets_loc]:
    known_planets_dill = {**known_planets_dill, **get_all_files_of_type(potential_data_location, ['dill', 'pickle'])}
check_for_duplicates(known_planets_dill)

# Locate all planet configurations
known_planets_cfg = get_all_files_of_type(planet_config_loc, ['cfg', 'json', 'json5'])
check_for_duplicates(known_planets_cfg)
_configs_are_dict = False

def _cfgpath_to_json():
    global known_planets_cfg
    global _configs_are_dict
    for planet_name, config_path in known_planets_cfg.items():
        known_planets_cfg[planet_name] = json5.load(config_path)

    _configs_are_dict = True

# Check for conflict nam

def build_planet(planet_name: str, planet_config: dict = None, force_build: bool = False, auto_save: bool = True,
                 planet_dill_path: str = None):

    log(f'Preparing to find and/or build world: {planet_name}')

    # See if dilled planet exists
    need_to_build = force_build
    if not force_build:

        if planet_dill_path is None and planet_name in known_planets_dill:
            planet_dill_path = known_planets_dill[planet_name]

        if planet_dill_path is not None:
            log(f'Dilled planet was found! This will save a lot of time...')
            # Planet was found! This will save a lot of time.
            with open(planet_dill_path, 'r') as planet_file:
                planet = dill.load(planet_file)

            # If no new planet_config is provided then we are done
            if planet_config is None:
                log(f'No new configurations were provided. Returning dilled planet')
                return planet
            else:
                log(f'New configurations were provided. Attempting to load them into dilled planet.')
                planet.user_config = planet_config
                planet.reinit()
                log(f'New configurations were successful loaded with no obvious issues.')
        else:
            log(f'Dilled version of {planet_name} was not found. Attempting to build.')
            need_to_build = True

    planet = None
    if need_to_build:
        if planet_config is None:
            log(f'No configuration file for {planet_name} was provided, attempting to locate saved version')
            if not _configs_are_dict:
                _cfgpath_to_json()
            if planet_name in known_planets_cfg:
                log(f'Configuration found')
                planet_config = known_planets_cfg[planet_name]
            else:
                raise MissingArgumentError('No Planet configuration found or provided. Not enough information to build planet.')

        # Determine world type
        planet_type = planet_config['type']
        if planet_type not in world_types:
            raise UnknownModelError('Unknown world type encountered.')
        planet_class = world_types[planet_type]

        if planet_class == BurnManWorld:
            log('Burnman planet type detected. Attempting to build BurnMan planet. This may take a while.')
            # Build BurnMan Planet first
            burnman_layers, burnman_planet = build_bm_planet(planet_config)
            log('Burnman planet build complete!')
            planet = planet_class(planet_config, burnman_planet, burnman_layers)
        else:
            log(f'{planet_class.pyname} planet type detected.')
            planet = planet_class(planet_config)

    if planet is None:
        raise Exception

    log('World Load Successful!')
    if auto_save:
        log('Attempting to save world to speed up future runs.')
        save_path = unique_path(attempt_path=os.path.join(dilled_planets_loc, f'{planet_name}.dill'), is_dir=False)
        try:
            with open(save_path, 'w') as new_dill_file:
                dill.dump(planet, new_dill_file)
        except Exception as e:
            log(f'Failed to save world due to following exception:\n\t{e}')

    return planet




