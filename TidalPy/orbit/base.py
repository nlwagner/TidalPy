from typing import Union, Dict, List

from ..helpers.orbit_help import pull_out_orbit_from_config
from .. import log
from ..exceptions import ImproperPropertyHandling, BadWorldSignature, BadWorldSignatureType, TidalPyOrbitError
from ..structures.worlds import AllWorldType, StarWorld, all_world_types
from ..tools.conversions import rads2days, days2rads, semi_a2orbital_motion, orbital_motion2semi_a
from ..utilities.classes.base import TidalPyClass
from ..utilities.types import FloatArray

WorldSignatureType = Union[str, int, AllWorldType]

class OrbitBase(TidalPyClass):

    """ OrbitBase class
    Contains attributes and methods to track the orbit of multiple TidalPy worlds.

    Orbits allow TidalPy worlds to communicate with one another and for tides to be calculated.

    Assumptions
    -----------
    .. All TidalPy orbits currently assume no interaction between tidal bodies or the orbit's star. The only interaction
        that is permitted is between a single tidal body and the orbit's tidal host (which could be the star).

    See Also
    --------
    TidalPy.orbit.PhysicsOrbit
    """

    class_name = 'base'

    def __init__(self, star: StarWorld = None, tidal_host: AllWorldType = None,
                 tidal_bodies: Union[AllWorldType, List[AllWorldType]] = None, star_host: bool = False,
                 host_tide_raiser: AllWorldType = None, initialize: bool = True):
        """ OrbitBase constructor

        Orbits allow TidalPy worlds to communicate with one another and for tides to be calculated.

        Notes
        -----
        .. Orbit instances can be made without any TidalPy worlds initially. Worlds can be later added with the
            `add_tidal_world` and `add_star` methods.

        Parameters
        ----------
        star : StarWorld
            Initialized TidalPy StarWorld that will become this orbit's star.
        tidal_host : AllWorldType
            Initialized TidalPy world that will become this orbit's tidal host.
        tidal_bodies : Union[AllWorldType, List[AllWorldType]]
            Initialized TidalPy world(s) that will become connected via this orbit to the tidal host and star.
            Multiple tidal bodies can be entered at once if provided to the constructor as a list.
        star_host : bool = False
            If `True`, then the star will act as the tidal host in addition to being the orbit's star.
        host_tide_raiser : AllWorldType = None
            The tidal host experiences tides from all worlds. This pointer is used to set which tidal body is currently
                being used as the host body's tide raiser.
        initialize : bool = True
            If `True`, then the constructor will make the first call to the orbit's `reinit()` method.
        """

        # Initialized properties
        self._all_objects = list()
        self._all_tidal_world_orbit_index_by_name = dict()
        self._all_tidal_world_orbit_index_by_instance = dict()
        self._tidal_objects = list()
        self._tidal_host = None
        self._star = None
        self._star_host = None

        # State properties
        self._host_tide_raiser = None
        self._eccentricities = list()
        self._semi_major_axes = list()
        self._orbital_frequencies = list()
        self._orbital_periods = list()
        self._universal_time = None

        # Construct anything we can
        run_update = False
        if star is not None and tidal_host is not None:
            # Check is the star is the tidal host or not, regardless of what the user provided for star_host
            if star is tidal_host:
                log.debug(f"Orbit's, {self}, star appears to be the tidal host.")
                star_host = True
            else:
                if star_host:
                    # User said star host, but this does not appear to be the case
                    log.error(f"User set star_host to True for orbit: {self}, but star and tidal host were both "
                              f"provided and not the same instance.")
                    raise TidalPyOrbitError('Tidal and Star not same instance for star_host')

        if star is not None:
            self.add_star(star, is_tidal_host=star_host, run_update=False)

        if tidal_host is not None:
            self.add_tidal_host(tidal_host, run_update=False)
            run_update = True

        if tidal_bodies is not None:
            run_update = True
            # Check if it is a list of bodies or a single body
            if type(tidal_bodies) in [list, tuple]:
                for tidal_body in tidal_bodies:
                    self.add_tidal_world(tidal_body, is_tidal_host=False, run_update=False)
            else:
                self.add_tidal_world(tidal_bodies, is_tidal_host=False, run_update=False)

        # Set the host body's tide raiser
        if host_tide_raiser is not None:
            self.set_host_tide_raiser(host_tide_raiser)

        if initialize:
            # Make initial call to orbit's reinit method
            self.reinit(initial_init=True, reinit_worlds=False, run_update=run_update)

    def reinit(self, initial_init: bool = False, reinit_worlds: bool = False, run_update: bool = True):
        """ Reinitialize various orbit properties.

        Parameters
        ----------
        initial_init : bool = False
            Is set to `True` for the first time the `reinit` method is called.
        reinit_worlds : bool = False
            If `True`, calls to each world's reinit will be made.
        run_update : bool = True
            If `True`, then the orbit's `update_orbit` method will be called.
        """

        if initial_init:
            log.debug(f'Initializing orbit: {self}.')
        else:
            log.debug(f'Reinit called for orbit: {self}.')

        if reinit_worlds:
            for world in self.all_objects:
                world.reinit()

        if not initial_init:
            self.clear_state()

            # Go through tidal worlds and make sure the orbit has the default orbital parameters
            for tidal_world in self.tidal_objects:
                orbital_freq, semi_major_axis, eccentricity = pull_out_orbit_from_config(tidal_world.config)
                self.set_state(tidal_world, new_eccentricity=eccentricity,
                               new_orbital_frequency=orbital_freq, new_semi_major_axis=semi_major_axis,
                               called_from_orbit=True)

        if run_update:
            self.update_orbit()

    def add_star(self, star_world: StarWorld, is_tidal_host: bool = False, run_update: bool = True):
        """ Add a star to the orbit. This star may or may not be the tidal host.

        Stars that are not tidal hosts are only used only for insolation calculations.

        Parameters
        ----------
        star_world : StarWorld
            TidalPy star instance to be added to the orbit.
        is_tidal_host : bool = False
            If `True`, then the star will be added as a tidal world as well.
        run_update : bool = True
            If `True`, the orbit's `update_orbit` method will be called after the world has been added.
        """

        # Check that the star is an instance of the expected class
        if not isinstance(star_world, StarWorld):
            log.warning(f'Star world is being added to orbit: {self}, but is of type {type(star_world)}. '
                        'Unexpected interactions may occur.')

        if not isinstance(star_world, all_world_types):
            log.error(f'Star world is being added to orbit: {self}, but is of type {type(star_world)}. '
                        'This is not supported.')
            raise TidalPyOrbitError('Unsupported world type encountered.')

        # Add star to orbit
        if self._star is not None:
            log.warning(f'Star world is being added to orbit: {self}, but star already present. Replacing...')
        self._star = star_world

        # Add it as a tidal world if it is the tidal host
        if is_tidal_host:
            log.debug(f'Star, {star_world}, is the tidal host, adding it to orbit as such.')
            self._star_host = True
            self.add_tidal_world(star_world, is_tidal_host=True, run_update=False)
        else:
            # Perform some initialization that will have been skipped by not calling `add_tidal_world`
            if star_world in self.all_objects:
                log.warning(f"Trying to add star: {star_world} to orbit's all objects, but instance already "
                            f"present. Skipping...")
            else:
                self.all_objects.append(star_world)

            # Now add the orbit to the world
            if star_world.orbit is not None:
                log.warning(f'Trying to add orbit: {self} to {star_world} but orbit is already present. Replacing...')
            star_world.orbit = self

        # Update orbit, if needed.
        if run_update:
            self.update_orbit()

    def add_tidal_world(self, tidal_world: AllWorldType, is_tidal_host: bool = False, run_update: bool = True):
        """ Add a new tidal world to the orbit, in order from closest to host to farthest away.

        Parameters
        ----------
        tidal_world : AllWorldType
            TidalPy world instance to be added to orbit.
        is_tidal_host : bool = False
            If true, then additional checks will be done to ensure proper functionality.
        run_update : bool = True
            If `True`, the orbit's `update_orbit` method will be called after the world has been added.
        """

        log.info(f'Adding {tidal_world} to orbit: {self}.')

        if not isinstance(tidal_world, all_world_types):
            log.error(f'Tidal world is being added to orbit: {self}, but is of type {type(tidal_world)}. '
                        'This is not supported.')
            raise TidalPyOrbitError('Unsupported world type encountered.')

        # Add to tidal objects list
        if is_tidal_host:
            orbit_index = 0
            # Check if the tidal host location is already full or not.
            if self.tidal_objects != list() or self.tidal_host is not None:
                if tidal_world is self.tidal_host:
                    log.warning(f'Adding tidal host to orbit: {self}, when one already present. '
                                f'Tidal host is equal to replacement: doing nothing.')
                else:
                    log.warning(f'Adding tidal host to orbit: {self}, when one already present. Replacing tidal host.')
                    self._tidal_host = tidal_world
                    self._tidal_objects[0] = tidal_world
            else:
                log.debug(f'Adding {tidal_world} as the tidal host for orbit: {self}.')
                self._tidal_host = tidal_world
                self._tidal_objects.append(tidal_world)
        else:
            if self.tidal_objects == list():
                if self.tidal_host is not None:
                    raise TidalPyOrbitError('How did this happen?')

                log.warning(f'Trying to add tidal worlds to orbit: {self} when no tidal host has been set. '
                            f'It is better to set the tidal host first. Proceeding...')
                # Have to append a None so that the first spot is reserved for the tidal host.
                self._tidal_objects.append(None)
                # Now the tidal world can be added as usual.

            # Add world to the tidal objects list
            orbit_index = len(self.tidal_objects)
            self._tidal_objects.append(tidal_world)

        # Add world to other storage locations
        #    All objects list
        if tidal_world in self.all_objects:
            log.warning(f"Trying to add tidal world: {tidal_world} to orbit's all objects, but instance already "
                        f"present. Skipping...")
        else:
            self.all_objects.append(tidal_world)
        #    Instance dict
        if tidal_world in self.all_tidal_world_orbit_index_by_instance:
            log.warning(f"Trying to add tidal world: {tidal_world} to orbit's instance dict, but instance already "
                        f"present. Skipping...")
        else:
            self._all_tidal_world_orbit_index_by_instance[tidal_world] = orbit_index
        #    Name dict
        tidal_world_name = tidal_world.name
        names_to_save = list()
        for name in [tidal_world_name, tidal_world_name.lower(), tidal_world_name.title()]:
            # Create a list of unique possible names for the world.
            if name not in names_to_save:
                names_to_save.append(name)
        for name in names_to_save:
            # Save unique names to the name look-up dict.
            if name in self.all_tidal_world_orbit_index_by_name:
                log.warning(f"Trying to add tidal world: {tidal_world} to orbit's name dict, but name key already "
                            f"present. Replacing...")
            self._all_tidal_world_orbit_index_by_name[name] = orbit_index

            # Save unique names to the orbits __dict__
            if name in self.__dict__:
                log.warning(f"Trying to add tidal world: {tidal_world} to orbit's `__dict__`, but name key already "
                            f"present. Replacing...")
            setattr(self, name, tidal_world)

        # Now add the orbit to the world
        if tidal_world.orbit is not None:
            log.warning(f'Trying to add orbit: {self} to {tidal_world} but orbit is already present. Replacing...')
        tidal_world.orbit = self

        # Add reference to the tidal host
        if is_tidal_host:
            if self.host_tide_raiser is not None:
                tidal_world.tidal_host = self.host_tide_raiser
        else:
            tidal_world.tidal_host = self.tidal_host

        # Make storage locations for this world's orbital parameters
        storage_warn = False
        for storage_list in [self._eccentricities, self._semi_major_axes, self._orbital_frequencies,
                             self._orbital_periods]:
            if len(storage_list) >= orbit_index + 1 and not storage_warn:
                # Already something at this location.
                log.warning(f'Trying to add orbital parameter placeholders for {tidal_world} in {self} but parameters '
                            f'already present (orbit index = {orbit_index}). Replacing...')
                # Set this flag to True so that a warning does not appear for each list.
                storage_warn = True

                # Replace with None.
                storage_list[orbit_index] = None
            else:
                # Append with None.
                storage_list.append(None)

        if not is_tidal_host:
            # Check the world's config and pull out any user-provided orbit information (not done for tidal host)
            orbital_freq, semi_major_axis, eccentricity = pull_out_orbit_from_config(tidal_world.config)
            self.set_state(tidal_world, new_eccentricity=eccentricity,
                           new_orbital_frequency=orbital_freq, new_semi_major_axis=semi_major_axis,
                           called_from_orbit=True)

            # If the tidal host tide raiser pointer is None, we can set it to the first tidal world added.
            if self.host_tide_raiser is None:
                log.debug(f'Host tide raiser pointer is not set in {self}. Setting to first tidal body.')
                self.set_host_tide_raiser(tidal_world)

        # Update orbit if needed
        if run_update:
            self.update_orbit()

    def add_tidal_host(self, tidal_host: AllWorldType, run_update: bool = True):
        """ Add a new tidal host to the orbit, in order from closest to host to farthest away.

        This is a convenience wrapper to OrbitBase.add_tidal_world

        Parameters
        ----------
        tidal_host : AllWorldType
            TidalPy world instance to be added to orbit.
        run_update : bool = True
            If `True`, the orbit's `update_orbit` method will be called after the world has been added.
        """

        self.add_tidal_world(tidal_host, is_tidal_host=True, run_update=run_update)

    def update_orbit(self):
        """ Method that updates anything that depends upon the state orbital properties of the tidal worlds.

        Should be called whenever there is a change to one or more orbital properties.
        """

        log.debug(f'Update orbit called for: {self}')

        # This method's functionality is largely set by child classes.

    def clear_state(self, clear_all: bool = True, clear_specific: WorldSignatureType = None,
                    clear_world_state: bool = False):
        """ Clears the orbital information for a world or all worlds without destroying the orbit instance.

        Parameters
        ----------
        clear_all : bool = True
            If `True`, all world's orbital state properties will be cleared.
        clear_specific : BadWorldSignatureType = None
            If not `None`, then only a specific world's state properties will be cleared.
        clear_world_state : bool = False
            If `True`, then a call will be made to a specific (or all if `clear_all == True`) world's `clear_state`
            method.
        """
        log.debug(f'Clear state called for orbit: {self}')

        if clear_all:
            log.debug(f'{self} clearing orbital data for all objects')
            if clear_world_state:
                log.debug(f'{self} clearing world data for all objects')

            for object_ in self.all_objects:
                if object_ is self.star:
                    pass
                else:
                    object_index = self.world_signature_to_index(object_)
                    # Clear orbital information
                    self._orbital_periods[object_index] = None
                    self._orbital_frequencies[object_index] = None
                    self._semi_major_axes[object_index] = None
                    self._eccentricities[object_index] = None

                if clear_world_state:
                    # Clear world data
                    object_.clear_state(preserve_orbit=True)
        else:
            if clear_specific is not None:
                specific_world_index = self.world_signature_to_index(clear_specific)
                specific_world = self.tidal_objects[specific_world_index]
                log.debug(f'{self} clearing orbital data for {specific_world}.')

                # Clear orbital information
                self._orbital_periods[specific_world_index] = None
                self._orbital_frequencies[specific_world_index] = None
                self._semi_major_axes[specific_world_index] = None
                self._eccentricities[specific_world_index] = None

                if clear_world_state:
                    log.debug(f'{self} clearing world data for {specific_world}.')
                    # Clear world's state
                    specific_world.clear_state(preserve_orbit=True)
            else:
                # Nothing to do...
                log.warning(f'Clear state called for {self}, but nothing to clear.')

    def set_host_tide_raiser(self, tide_raiser: WorldSignatureType):
        """ Set the pointer used by the orbit class to find which body is currently raising tides on the host world.

        Parameters
        ----------
        tide_raiser : WorldSignatureType
            Signature (instance, orbital location, or name) of the tidal body that is currently raising tides on the
                host.
            The tidal body must be added to the orbit before it can be set as the host's tide raiser.

        """

        # Get the tide raiser's orbital location
        tide_raiser_index = self.world_signature_to_index(tide_raiser)

        # Get the instance of the tide raiser and store it
        tide_raiser_instance = self.tidal_objects[tide_raiser_index]
        self._host_tide_raiser = tide_raiser_instance

        # Now switch the "tidal host" on the tidal_host (tides are weird!)
        self.tidal_host.tidal_host = self.host_tide_raiser

    def set_state(self, world_signature: WorldSignatureType,
                  new_eccentricity: FloatArray = None,
                  new_semi_major_axis: FloatArray = None,
                  new_orbital_frequency: FloatArray = None,
                  new_orbital_period: FloatArray = None,
                  called_from_orbit: bool = False):
        """ Set the orbital state for a world with the provided signature.

        This largely wraps the other orbit setter methods.

        Parameters
        ----------
        world_signature : WorldSignatureType
            A signature used to distinguish one tidal world from another. This could be its name,
                orbital location index, or the instance of an initialized TidalPy world.
        new_eccentricity : FloatArray
            New orbital eccentricity for this world.
        new_semi_major_axis : FloatArray
            New orbital semi-major axis for this world [m].
        new_orbital_frequency : FloatArray
            New orbital frequency for this world [rad s-1].
        new_orbital_period : FloatArray
            New mean orbital period for this world [days].
        called_from_orbit : bool = False
            Flag for if this method was called from an OrbitBase method. This avoids repeated calls to the
                OrbitBase.update_orbit() method.
        """

        # Figure out what orbital separation information was provided.
        orbital_freq_provided = new_orbital_frequency is not None
        orbital_period_provided = new_orbital_period is not None
        orbital_semi_a_provided = new_semi_major_axis is not None

        # To avoid issues duplicate array pointers - just raise an error if too much information was provided.
        if orbital_freq_provided and orbital_period_provided and orbital_semi_a_provided:
            log.error(f'All of: orbital motion, orbital period, and semi-major axis were provided to {self}. '
                      f'Please provide only one.')
            raise TidalPyOrbitError(f'All of: orbital motion, orbital period, and semi-major axis were provided to {self}. '
                                    f'Please provide only one.')
        if (orbital_freq_provided and orbital_period_provided) or \
           (orbital_period_provided and orbital_semi_a_provided) or \
           (orbital_freq_provided and orbital_semi_a_provided):
            log.error(f'Two of: orbital motion, orbital period, and semi-major axis were provided to {self}. '
                      f'Please provide only one.')
            raise TidalPyOrbitError(
                f'Two of: orbital motion, orbital period, and semi-major axis were provided to {self}. '
                f'Please provide only one.')

        # Calculate the orbital conversions
        if orbital_freq_provided:
            new_orbital_period = rads2days(new_orbital_frequency)
            new_semi_major_axis = self.orbital_motion2semi_a(world_signature, new_orbital_frequency)
        elif orbital_period_provided:
            new_orbital_frequency = days2rads(new_orbital_period)
            new_semi_major_axis = self.orbital_motion2semi_a(world_signature, new_orbital_frequency)
        elif new_semi_major_axis:
            new_orbital_frequency = self.semi_a2orbital_motion(world_signature, new_semi_major_axis)
            new_orbital_period = rads2days(new_orbital_frequency)

        # Set state properties
        if orbital_freq_provided or orbital_period_provided or orbital_semi_a_provided:
            self.set_semi_major_axis(world_signature, new_semi_major_axis, called_from_orbit=True)
            self.set_orbital_frequency(world_signature, new_orbital_frequency, called_from_orbit=True)
            self.set_orbital_period(world_signature, new_orbital_period, called_from_orbit=True)

        # Now deal with eccentricity
        eccentricity_provided = new_eccentricity is not None
        if eccentricity_provided:
            self.set_eccentricity(world_signature, new_eccentricity, called_from_orbit=True)

        # Finally, make a single call to the update_orbit method
        if not called_from_orbit:
            self.update_orbit()

    def set_eccentricity(self, world_signature: WorldSignatureType, new_eccentricity: FloatArray,
                        called_from_orbit: bool = False):
        """ Set the eccentricity for a world with the provided signature.

        Parameters
        ----------
        world_signature : WorldSignatureType
            A signature used to distinguish one tidal world from another. This could be its name,
                orbital location index, or the instance of an initialized TidalPy world.
        new_eccentricity : FloatArray
            New orbital eccentricity for this world.
        called_from_orbit : bool = False
            Flag for if this method was called from an OrbitBase method. This avoids repeated calls to the
                OrbitBase.update_orbit() method.
        """

        # Get world's index
        world_index = self.world_signature_to_index(world_signature)

        # Update its eccentricity
        self._eccentricities[world_index] = new_eccentricity

        # Make sure changes are propagated everywhere they need to be
        if not called_from_orbit:
            self.update_orbit()

    def set_semi_major_axis(self, world_signature: WorldSignatureType, new_semi_major_axis: FloatArray,
                            called_from_orbit: bool = False):
        """ Set the semi-major axis for a world with the provided signature.

        Parameters
        ----------
        world_signature : WorldSignatureType
            A signature used to distinguish one tidal world from another. This could be its name,
                orbital location index, or the instance of an initialized TidalPy world.
        new_semi_major_axis : FloatArray
            New orbital semi-major axis for this world [m].
        called_from_orbit : bool = False
            Flag for if this method was called from an OrbitBase method. This avoids repeated calls to the
                OrbitBase.update_orbit() method.
        """

        # Get world's index
        world_index = self.world_signature_to_index(world_signature)

        # Update its semi-major axis
        self._semi_major_axes[world_index] = new_semi_major_axis

        # Make sure changes are propagated everywhere they need to be
        if not called_from_orbit:
            # Changing the semi-major axis will change the orbital motion (and orbital period)
            new_orbital_frequency = self.semi_a2orbital_motion(world_index, new_semi_major_axis)
            new_orbital_period = rads2days(new_orbital_frequency)
            self.set_orbital_frequency(world_index, new_orbital_frequency, called_from_orbit=True)
            self.set_orbital_period(world_index, new_orbital_period, called_from_orbit=True)

            # Now we can update the orbit
            self.update_orbit()

    def set_orbital_frequency(self, world_signature: WorldSignatureType, new_orbital_frequency: FloatArray,
                              called_from_orbit: bool = False):
        """ Set the mean orbital motion for a world with the provided signature.

        Parameters
        ----------
        world_signature : WorldSignatureType
            A signature used to distinguish one tidal world from another. This could be its name,
                orbital location index, or the instance of an initialized TidalPy world.
        new_orbital_frequency : FloatArray
            New orbital frequency for this world [rad s-1].
        called_from_orbit : bool = False
            Flag for if this method was called from an OrbitBase method. This avoids repeated calls to the
                OrbitBase.update_orbit() method.
        """

        # Get world's index
        world_index = self.world_signature_to_index(world_signature)

        # Update its orbital frequency
        self._orbital_frequencies[world_index] = new_orbital_frequency

        # Worlds that are forced to be in synchronous rotation will need to have their spin-rates changed to match
        #    the new orbital frequency.
        if self.tidal_objects[world_index].force_spin_sync:
            self.tidal_objects[world_index].set_spin_frequency(new_orbital_frequency, call_updates=False)

        # Make sure changes are propagated everywhere they need to be
        if not called_from_orbit:
            # Changing the orbital motion will change the semi-major axis and orbital period
            new_semi_major_axis = self.orbital_motion2semi_a(world_index, new_orbital_frequency)
            new_orbital_period = rads2days(new_orbital_frequency)
            self.set_semi_major_axis(world_index, new_semi_major_axis, called_from_orbit=True)
            self.set_orbital_period(world_index, new_orbital_period, called_from_orbit=True)

            # Now we can update the orbit
            self.update_orbit()

    def set_orbital_period(self, world_signature: WorldSignatureType, new_orbital_period: FloatArray,
                           called_from_orbit: bool = False):
        """ Set the mean orbital period for a world with the provided signature.

        Parameters
        ----------
        world_signature : WorldSignatureType
            A signature used to distinguish one tidal world from another. This could be its name,
                orbital location index, or the instance of an initialized TidalPy world.
        new_orbital_period : FloatArray
            New mean orbital period for this world [days].
        called_from_orbit : bool = False
            Flag for if this method was called from an OrbitBase method. This avoids repeated calls to the
                OrbitBase.update_orbit() method.
        """

        # Get world's index
        world_index = self.world_signature_to_index(world_signature)

        # Update its orbital period
        self._orbital_periods[world_index] = new_orbital_period

        # Make sure changes are propagated everywhere they need to be
        if not called_from_orbit:
            # Changing the orbital period will change the orbital motion and the semi-major axis
            new_orbital_frequency = days2rads(new_orbital_period)
            new_semi_major_axis = self.orbital_motion2semi_a(world_index, new_orbital_frequency)
            self.set_semi_major_axis(world_index, new_semi_major_axis, called_from_orbit=True)
            self.set_orbital_frequency(world_index, new_orbital_frequency, called_from_orbit=True)

            # Now we can update the orbit
            self.update_orbit()

    def world_signature_to_index(self, world_signature: WorldSignatureType):
        """ Convert's a world's signature to the orbital index which is used for tracking various parameters.

        Parameters
        ----------
        world_signature : WorldSignatureType
            A signature used to distinguish one tidal world from another. This could be its name,
                orbital location index, or the instance of an initialized TidalPy world.

        Returns
        -------
        world_orbit_index : int
            The index of the world within the orbit's tidal_objects list.
        """

        # The tidal host's orbital information is stored in whichever tidal body that is currently raising tides on it.
        #    This flag is used to check if the method needs to return a different body's index.
        return_host_tide_raiser = False

        # Check if it is an integer and if it makes sense given the number of objects stored in this orbit
        if type(world_signature) == int:
            if world_signature >= len(self.tidal_objects):
                log.error(f'World signature provided as int, {world_signature}, but does not make sense with number of '
                          f'tidal objects within orbit: {len(self.tidal_objects)}.')
                raise BadWorldSignature
            elif world_signature == 0:
                # This is the index of the tidal host
                return_host_tide_raiser = True
            else:
                world_orbit_index = world_signature
                return world_orbit_index
        elif type(world_signature) == str:
            # Check if the world signature is a string if it is, assume it is the world's name
            if world_signature in self.all_tidal_world_orbit_index_by_name:
                # Name is fine.
                pass
            elif world_signature.lower() in self.all_tidal_world_orbit_index_by_name:
                # Name needs to be lowercase
                world_signature = world_signature.lower()
            elif world_signature.title() in self.all_tidal_world_orbit_index_by_name:
                # Name needs to be titlecase
                world_signature = world_signature.title()
            else:
                log.error(f'World signature provided as str, {world_signature}, but does not match any known tidal '
                          f'worlds stored in orbit.')
                raise BadWorldSignature
            if world_signature in [self.tidal_host.name, self.tidal_host.name.lower(), self.tidal_host.name.title()]:
                # This is the name of the tidal host
                return_host_tide_raiser = True
            else:
                world_orbit_index = self.all_tidal_world_orbit_index_by_name[world_signature]
                return world_orbit_index
        elif isinstance(world_signature, all_world_types):
            # Finally check if the world signature is an instance of a TidalPy World and if that instance is stored in orbit
            if world_signature is self.tidal_host:
                return_host_tide_raiser = True
            elif world_signature in self.all_tidal_world_orbit_index_by_instance:
                world_index = self.all_tidal_world_orbit_index_by_instance[world_signature]
                return world_index
            else:
                log.error(f'World signature provided as instance, {world_signature}, but does not match any known tidal '
                          f'worlds stored in orbit.')
                raise BadWorldSignature

        if return_host_tide_raiser:
            # Recall this method, but to get the tide raiser's index.
            if self.host_tide_raiser is None:
                log.error("Trying to access tidal host parameters but the host's tide raiser has not been set.")
                raise TidalPyOrbitError("Trying to access tidal host parameters but the host's tide raiser has not "
                                        "been set.")
            return self.world_signature_to_index(self.host_tide_raiser)

        # No idea what type the world signature is...
        log.error(f'World signature provided as an unexpected type: {type(world_signature)}.')
        raise BadWorldSignatureType


    # # Tidal World Getters
    def get_eccentricity(self, world_signature: WorldSignatureType) -> FloatArray:
        """ Provided a world's signature, this method will retrieve its orbital eccentricity.

        Parameters
        ----------
        world_signature : WorldSignatureType
            A signature used to distinguish one tidal world from another. This could be its name,
                orbital location index, or the instance of an initialized TidalPy world.

        Returns
        -------
        eccentricity : FloatArray
            Orbital eccentricity relative to the tidal host.
        """

        world_index = self.world_signature_to_index(world_signature)

        return self.eccentricities[world_index]

    def get_semi_major_axis(self, world_signature: WorldSignatureType) -> FloatArray:
        """ Provided a world's signature, this method will retrieve its orbital semi_major_axis [m].

        Parameters
        ----------
        world_signature : WorldSignatureType
            A signature used to distinguish one tidal world from another. This could be its name,
                orbital location index, or the instance of an initialized TidalPy world.

        Returns
        -------
        semi_major_axis : FloatArray
            Orbital semi-major axis relative to the tidal host [m].
        """

        world_index = self.world_signature_to_index(world_signature)

        return self.semi_major_axes[world_index]

    def get_orbital_frequency(self, world_signature: WorldSignatureType) -> FloatArray:
        """ Provided a world's signature, this method will retrieve its orbital frequency [rad s-1].

        Parameters
        ----------
        world_signature : WorldSignatureType
            A signature used to distinguish one tidal world from another. This could be its name,
                orbital location index, or the instance of an initialized TidalPy world.

        Returns
        -------
        orbital_motion : FloatArray
            Orbital mean motion relative to the tidal host [rad s-1].
        """

        world_index = self.world_signature_to_index(world_signature)

        return self.orbital_frequencies[world_index]

    def get_orbital_period(self, world_signature: WorldSignatureType) -> FloatArray:
        """ Provided a world's signature, this method will retrieve its orbital eccentricity [days].

        Parameters
        ----------
        world_signature : WorldSignatureType
            A signature used to distinguish one tidal world from another. This could be its name,
                orbital location index, or the instance of an initialized TidalPy world.

        Returns
        -------
        orbital_period : FloatArray
            Orbital period relative to the tidal host [days].
        """

        world_index = self.world_signature_to_index(world_signature)

        return self.orbital_periods[world_index]

    # # Conversions
    def semi_a2orbital_motion(self, world_signature: WorldSignatureType, semi_major_axis: FloatArray) -> FloatArray:
        """ Providing a world's signature and a semi-major axis, this method will calculate the world's orbital motion.

        This is largely a convenience wrapper around TidalPy.tools.conversions.semi_a2orbital_motion.

        Parameters
        ----------
        world_signature : WorldSignatureType
            A signature used to distinguish one tidal world from another. This could be its name,
                orbital location index, or the instance of an initialized TidalPy world.
        semi_major_axis : FloatArray
            The world's orbital semi-major axis relative to the tidal host [m].

        Returns
        -------
        orbital_motion : FloatArray
            The world's orbital mean motion relative to the tidal host [rad s-1].
        """

        world_index = self.world_signature_to_index(world_signature)
        if world_index == 0:
            log.warning('Trying to calculate orbital motion for the tidal host relative to itself.')

        world_mass = self.tidal_objects[world_index].mass
        orbital_motion = semi_a2orbital_motion(semi_major_axis, self.tidal_host.mass, world_mass)

        return orbital_motion

    def orbital_motion2semi_a(self, world_signature: WorldSignatureType, orbital_motion: FloatArray) -> FloatArray:
        """ Providing a world's signature and a orbital motion, this method will calculate the world's semi-major axis.

        This is largely a convenience wrapper around TidalPy.tools.conversions.orbital_motion2semi_a.

        Parameters
        ----------
        world_signature : WorldSignatureType
            A signature used to distinguish one tidal world from another. This could be its name,
                orbital location index, or the instance of an initialized TidalPy world.
        orbital_motion : FloatArray
            The world's orbital mean motion relative to the tidal host [rad s-1].

        Returns
        -------
        semi_major_axis : FloatArray
            The world's orbital semi-major axis relative to the tidal host [m].
        """

        world_index = self.world_signature_to_index(world_signature)
        if world_index == 0:
            log.warning('Trying to calculate semi-major axis for the tidal host relative to itself.')

        world_mass = self.tidal_objects[world_index].mass
        semi_major_axis = orbital_motion2semi_a(orbital_motion, self.tidal_host.mass, world_mass)

        return semi_major_axis


    # # Initialized properties
    @property
    def all_objects(self) -> List[AllWorldType]:
        """ An iterable list of all world-like instances reinit in this Orbit class. """
        return self._all_objects

    @all_objects.setter
    def all_objects(self, value):
        raise ImproperPropertyHandling

    @property
    def all_tidal_world_orbit_index_by_name(self) -> Dict[str, int]:
        """ Dictionary of tidal world orbit locations stored by their name. """
        return self._all_tidal_world_orbit_index_by_name

    @all_tidal_world_orbit_index_by_name.setter
    def all_tidal_world_orbit_index_by_name(self, value):
        raise ImproperPropertyHandling

    @property
    def all_tidal_world_orbit_index_by_instance(self) -> Dict[AllWorldType, int]:
        """ Dictionary of tidal world orbit locations stored by their TidalPy class instance. """
        return self._all_tidal_world_orbit_index_by_instance

    @all_tidal_world_orbit_index_by_instance.setter
    def all_tidal_world_orbit_index_by_instance(self, value):
        raise ImproperPropertyHandling

    @property
    def tidal_objects(self) -> List[AllWorldType]:
        """ An iterable list of all tidal world instances reinit in this Orbit class.

        This differs from Orbit.all_objects in that it excludes the star (unless the star is the tidal host)
        """

        return self._tidal_objects

    @tidal_objects.setter
    def tidal_objects(self, value):
        raise ImproperPropertyHandling

    @property
    def tidal_host(self) -> AllWorldType:
        """ A reference to the orbit's tidal host world. """
        return self._tidal_host

    @tidal_host.setter
    def tidal_host(self, value):
        raise ImproperPropertyHandling

    @property
    def star(self) -> StarWorld:
        """ A reference to the orbit's star. """
        return self._star

    @star.setter
    def star(self, value):
        raise ImproperPropertyHandling

    @property
    def star_host(self) -> bool:
        """ A flag for if the star is acting as the tidal host. """
        return self._star_host

    @star_host.setter
    def star_host(self, value):
        raise ImproperPropertyHandling


    # # State properties
    @property
    def eccentricities(self):
        """ A list of all the tidal object instances' orbital eccentricity relative to the tidal host. """
        return self._eccentricities

    @eccentricities.setter
    def eccentricities(self, value):
        raise ImproperPropertyHandling('Set the eccentricity using the `set_state` method on the Orbit class.')

    @property
    def semi_major_axes(self):
        """ A list of all the tidal object instances' orbital semi-major axis relative to the tidal host. """
        return self._semi_major_axes

    @semi_major_axes.setter
    def semi_major_axes(self, value):
        raise ImproperPropertyHandling('Set the semi-major axis using the `set_state` method on the Orbit class.')

    @property
    def orbital_frequencies(self):
        """ A list of all the tidal object instances' orbital motion relative to the tidal host. """
        return self._orbital_frequencies

    @orbital_frequencies.setter
    def orbital_frequencies(self, value):
        raise ImproperPropertyHandling('Set the orbital frequency using the `set_state` method on the Orbit class.')

    @property
    def orbital_periods(self):
        """ A list of all the tidal object instances' orbital period relative to the tidal host. """
        return self._orbital_periods

    @orbital_periods.setter
    def orbital_periods(self, value):
        raise ImproperPropertyHandling('Set the orbital period using the `set_state` method on the Orbit class.')

    @property
    def host_tide_raiser(self) -> AllWorldType:
        """ This pointer is used to set which tidal body is currently being used as the host body's tide raiser. """
        return self._host_tide_raiser

    @host_tide_raiser.setter
    def host_tide_raiser(self, value):
        self.set_host_tide_raiser(value)

    @property
    def universal_time(self) -> FloatArray:
        """ Time used in integration studies as well as for calculating radiogenic heating in all tidal worlds. """
        return self._universal_time

    @universal_time.setter
    def universal_time(self, value: FloatArray):

        self._universal_time = value

        # Need to tell all worlds that the time has been updated.
        for world in self.tidal_objects:
            world.update_time()

    # # Aliased properties
    @property
    def time(self):
        """ Wrapper for OrbitBase.universal_time """
        return self.universal_time

    @time.setter
    def time(self, value):
        self.universal_time = value


    # # Dunder properties
    def __iter__(self):
        return iter(self.tidal_objects)

    def __str__(self):
        orbit_name = f'Orbit ({self.class_name}'

        if self.tidal_host is not None:
            orbit_name += f', H={self.tidal_host.name}'

        if self.star is not None:
            orbit_name += f', S={self.star.name}'

        num_tidal_worlds = len(self.tidal_objects)
        if num_tidal_worlds > 1:
            orbit_name += f', TW={num_tidal_worlds}'

        orbit_name += ')'
        return orbit_name


    # # Static methods
    @staticmethod
    def rads2days(frequency: FloatArray) -> FloatArray:
        """ Convert radians/sec (frequency) to days (period)

        Wrapper for TidalPy.tools.conversions.rads2days

        Parameters
        ----------
        frequency : FloatArray
            Orbital or Spin frequency in [rad s-1]

        Returns
        -------
        days : FloatArray
            Orbital or Spin Period in [days]
        """

        return rads2days(frequency)

    @staticmethod
    def days2rads(days: FloatArray) -> FloatArray:
        """ Convert days (period) to radians/sec (frequency)

        Wrapper for TidalPy.tools.conversions.days2rads

        Parameters
        ----------
        days : FloatArray
            Orbital or Spin Period in [days]

        Returns
        -------
        frequency : FloatArray
            Orbital or Spin frequency in [rad s-1]
        """

        return days2rads(days)