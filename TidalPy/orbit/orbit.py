from __future__ import annotations

from TidalPy import debug_mode
import numpy as np
from typing import List, Union
from scipy.constants import G

from TidalPy.exceptions import ImproperAttributeHandling, ParameterError, ParameterMissingError, IncorrectArgumentType
from TidalPy.utilities.classes import TidalPyClass
from TidalPy.utilities.conversions import Au2m
from ..types import FloatArray
from .. import log

from typing import TYPE_CHECKING, Union
from ..structures.worlds import WorldBase, TidalWorld
PlanetRefType = Union[str, int, WorldBase]

class OrbitBase(TidalPyClass):


    def __init__(self, star, host = None, target_bodies: list = None):
        """

        :param star:
        :param host:
        :param target_bodies:
        """

        super().__init__()

        self.star = star
        if host is None:
            host = star
        self.host = host
        if target_bodies is None:
            self.target_bodies = list()
        else:
            self.target_bodies = target_bodies  # type: List[TidalWorld]
        self.all_objects = [self.star, self.host] + self.target_bodies
        self.all_objects_byname = {self.star.name: self.star}
        if self.host.name not in self.all_objects_byname:
            self.all_objects_byname[self.host.name] = self.host

        # State orbital variables (must be at least 2: for the star and the host)
        self._eccentricities = [np.asarray(0.), np.asarray(0.)]  # type: List[Union[None, np.ndarray]]
        self._inclinations = [np.asarray(0.), np.asarray(0.)]    # type: List[Union[None, np.ndarray]]
        self._orbital_freqs = [None, None]   # type: List[Union[None, np.ndarray]]
        self._semi_major_axis = [None, None] # type: List[Union[None, np.ndarray]]

        # Are target bodies orbiting the star or the host
        self.star_host = False
        if self.star is self.host:
            self.star_host = True
        self.equilib_distance_funcs = [None, lambda: self.host.semi_major_axis]
        self.equilib_eccentricity_funcs = [None, lambda: self.host.eccentricity]

        # Determine various constants that depend upon other objects properties
        for target_body in self.target_bodies:

            # Check for potential issues
            if target_body is self.star:
                raise ParameterError("The orbit's star can not be a target body")
            if target_body is self.host:
                raise ParameterError("The orbit's host can not be a target body. "
                                     "Nevertheless, tides can still be calculated in the host. See documentation.")

            # Store reference in a more human readable manner
            self.all_objects_byname[target_body.name] = target_body

            # Equilibrium temperature
            if self.star_host:
                equib_dist_func = lambda: target_body.semi_major_axis
                equib_ecc_func = lambda: target_body.eccentricity
            else:
                equib_dist_func = lambda: self.host.semi_major_axis
                equib_ecc_func = lambda: self.host.eccentricity
            self.equilib_distance_funcs.append(equib_dist_func)
            self.equilib_eccentricity_funcs.append(equib_ecc_func)

            # Inflated Tidal Susceptibility
            target_body.tidal_susceptibility_inflated = (3. / 2.) * G * self.host.mass**2 * target_body.radius**5

            # Store dummy values for the state variables
            self._eccentricities.append(np.asarray(0.))
            self._inclinations.append(np.asarray(0.))
            self._orbital_freqs.append(None)
            self._semi_major_axis.append(None)

        for t_i, world in enumerate(self.all_objects):
            # Add a reference to the orbit to the planet(s)
            world._orbit = self
            world.orbit_location = t_i

            # Star does not need (or have) any of the following parameters - so skip it
            if world is self.star:
                continue

            # Update those dummy variables if information was provided in the configurations
            orbital_freq = world.config.get('orbital_freq', None)
            orbital_period = world.config.get('orbital_period', None)
            if orbital_freq is not None and orbital_period is not None:
                log(f'Both orbital frequency and period were provided for {world.name}. '
                    f'Using frequency instead.', level='info')
            if orbital_freq is None and orbital_period is not None:
                # Assume orbital period is in days
                orbital_freq = 2. * np.pi / (orbital_period * 24. * 60. * 60.)
            semi_major_axis = world.config.get('semi_major_axis', None)
            semi_major_axis_inau = world.config.get('semi_major_axis_in_au', False)
            if semi_major_axis is not None:
                if semi_major_axis_inau:
                    semi_major_axis = Au2m(semi_major_axis)
                if orbital_freq is not None:
                    log(f'Both orbital frequency (or period) and semi-major axis were provided for {world.name}. '
                        f'Using frequency instead.', level='info')
                    semi_major_axis = None
            eccentricity = world.config.get('eccentricity', None)
            inclination = world.config.get('inclination', None)
            self.set_orbit(world, orbital_freq, semi_major_axis, eccentricity, inclination, set_by_planet=True)

        # Update parameters on the planets
        self.host.is_host = True

        # Attempt to initialize insolation heating
        for target_body in self.target_bodies:
            self.calculate_insolation(target_body)
        if isinstance(self.host, TidalWorld):
            self.calculate_insolation(self.host)

        # The host body may be tidally dissipating due to one of the other target planets.
        self._host_tide_raiser_loc = None

    def find_planet_pointer(self, planet_reference: PlanetRefType) -> WorldBase:
        """ Find the object pointer to a planet or object stored in this orbit

        Possible references:
            str:
                Name of the planet provided in the configuration
                    This is case insensitive unless there the config provided a capital letter any where other than
                    the first letter.
            int:
                Planet's location in the orbit based on the following scheme:
                    0 == Star
                    1 == Host
                    2+ == Various target planets (these are probably the ones you want to set!)

        Parameters
        ----------
        planet_reference : PlanetRefType
            User-friendly reference to a planet or object stored in this orbit

        Returns
        -------
        planet_reference : TidalWorld
            Pointer to the planet object
        """

        if isinstance(planet_reference, WorldBase):
            # Already is the pointer!
            return planet_reference
        if type(planet_reference) == int:
            return self.all_objects[planet_reference]
        if type(planet_reference) == str:
            try:
                return self.all_objects_byname[planet_reference]
            except KeyError:
                try:
                    return self.all_objects_byname[planet_reference.lower()]
                except KeyError:
                    return self.all_objects_byname[planet_reference.title()]
        raise IncorrectArgumentType

    # Functionality to set the orbit state variables
    def set_orbit(self, planet_reference: PlanetRefType, orbital_freq: FloatArray = None,
                  semi_major_axis: FloatArray = None, eccentricity: FloatArray = None, inclination: FloatArray = None,
                  set_by_planet: bool = False):
        """ Set the orbital state (orbital frequency, eccentricity, etc.) of a planet in this orbit.

        Can set orbital frequency, semi-major axis, eccentricity, and/or inclination of a planet at the
        orbit location: planet_loc. Use this method instead of the individual set_eccentricity, set_inclination, etc.
        methods when you are changing 2 or more parameters simultaneously (it is more efficient).

        Parameters
        ----------
        planet_reference : PlanetRefType
            Reference used to find the planet
        orbital_freq : FloatArray
            Orbital frequency in [rads s-1]
            Optional, Mutually exclusive with semi-major axis
        semi_major_axis : FloatArray
            Semi-major axis in [m]
            Optional, Mutually exclusive with orbital_freq
        eccentricity : FloatArray
            Eccentricity (only elliptical and circular orbits are supported in TidalPy)
            Optional
        inclination : FloatArray
            Orbital inclination relative to the orbital plane in [rads]
            Optional
        set_by_planet : bool = False
            A flag used by TidalPy planets to make calls to this method. Leave it False for user use.
            If set to true then other models and parameters will not update correctly.

        See Also
        --------
        OrbitBase.set_orbital_freq
        OrbitBase.set_semi_major_axis
        OrbitBase.set_eccentricity
        OrbitBase.set_inclination
        """

        planet_pointer = self.find_planet_pointer(planet_reference)

        # The "set_by_planet" are set to True below even if the "set_by_planet" is False above because the function
        #    Will do a single call to orbit_update at the end.
        if orbital_freq is not None:
            if semi_major_axis is not None:
                raise ImproperAttributeHandling('Only set orbital frequency or semi-major axis, not both.')
            self.set_orbital_freq(planet_pointer, orbital_freq, set_by_planet=True)

        if semi_major_axis is not None:
            self.set_semi_major_axis(planet_pointer, semi_major_axis, set_by_planet=True)

        if eccentricity is not None:
            self.set_eccentricity(planet_pointer, eccentricity, set_by_planet=True)

        if inclination is not None:
            self.set_inclination(planet_pointer, inclination, set_by_planet=True)

        if not set_by_planet:
            # Need to tell the planet to update any orbital-dependent state
            # This is the benefit of using this method over the other setters. Guarantees only one call to orbit_update
            planet_pointer.update_orbit()

    def set_orbital_freq(self, planet_reference: PlanetRefType, new_orbital_freq: FloatArray,
                         set_by_planet: bool = False):
        """ Set the orbital frequency of a planet at planet_loc

        Use Orbit.set_orbit if setting more than one state parameter.

        Parameters
        ----------
        planet_reference : PlanetRefType
            Reference used to find the planet
        new_orbital_freq : FloatArray
            Orbital frequency in [rads s-1]
            Optional, Mutually exclusive with semi-major axis
        set_by_planet : bool = False
            A flag used by TidalPy planets to make calls to this method. Leave it False for user use.
            If set to true then other models and parameters will not update correctly.

        See Also
        --------
        OrbitBase.set_orbit
        """

        planet_pointer = self.find_planet_pointer(planet_reference)
        planet_loc = planet_pointer.orbit_location

        if planet_pointer is self.star:
            raise ParameterError("Can not change a star's orbit")
        elif planet_pointer is self.host and not self.star_host and not set_by_planet:
            if debug_mode:
                # This would change the orbit between the host and the star, not between the target body and the star.
                # This may not be what the user wants to do.
                log("Attempting to change the host planet's orbit", level='debug')

        if type(new_orbital_freq) != np.ndarray:
            new_orbital_freq = np.asarray(new_orbital_freq)

        self._orbital_freqs[planet_loc] = new_orbital_freq

        # Changing the orbital frequency also changes the semi-major axis. Update via Kepler Laws
        if planet_pointer is self.host:
            self._semi_major_axis[planet_loc] = \
                np.cbrt(G * (planet_pointer.mass + self.star.mass) / new_orbital_freq**2)
        else:
            # Star and Target bodies all orbit the host (albeit the star doesn't do much...).
            self._semi_major_axis[planet_loc] = \
                np.cbrt(G * (planet_pointer.mass + self.host.mass) / new_orbital_freq**2)

        if not set_by_planet:
            # Need to tell the planet to update any orbital-dependent state
            planet_pointer.update_orbit()

    def set_semi_major_axis(self, planet_reference: PlanetRefType, new_semi_major_axis: FloatArray,
                            set_by_planet: bool = False):
        """ Set the semi-major axis of a planet at planet_loc

        Use Orbit.set_orbit if setting more than one state parameter.

        Parameters
        ----------
        planet_reference : PlanetRefType
            Reference used to find the planet
        new_semi_major_axis : FloatArray
            Semi-major axis in [m]
            Optional, Mutually exclusive with orbital_freq
        set_by_planet : bool = False
            A flag used by TidalPy planets to make calls to this method. Leave it False for user use.
            If set to true then other models and parameters will not update correctly.

        See Also
        --------
        OrbitBase.set_orbit
        """

        planet_pointer = self.find_planet_pointer(planet_reference)
        planet_loc = planet_pointer.orbit_location

        if planet_pointer is self.star:
            raise ParameterError("Can not change a star's orbit")
        elif planet_pointer is self.host and not self.star_host and not set_by_planet:
            if debug_mode:
                # This would change the orbit between the host and the star, not between the target body and the star.
                # This may not be what the user wants to do.
                log("Attempting to change the host planet's orbit", level='debug')

        if type(new_semi_major_axis) != np.ndarray:
            new_semi_major_axis = np.asarray(new_semi_major_axis)

        self._semi_major_axis[planet_loc] = new_semi_major_axis
        # Changing the orbital semi-major axis also changes the orbital frequency. Update via Kepler Laws
        if planet_pointer is self.host:
            self._orbital_freqs[planet_loc] = \
                np.sqrt(G * (planet_pointer.mass + self.star.mass) / new_semi_major_axis**3)
        else:
            # Star and Target bodies all orbit the host (albeit the star doesn't do much...).
            self._orbital_freqs[planet_loc] = \
                np.sqrt(G * (planet_pointer.mass + self.host.mass) / new_semi_major_axis**3)

        if not set_by_planet:
            # Need to tell the planet to update any orbital-dependent state
            planet_pointer.update_orbit()

    def set_eccentricity(self, planet_reference: PlanetRefType, new_eccentricity: FloatArray, set_by_planet: bool = False):
        """ Set the eccentricity of a planet at planet_loc

        Use Orbit.set_orbit if setting more than one state parameter.

        Parameters
        ----------
        planet_reference : PlanetRefType
            Reference used to find the planet
        new_eccentricity : FloatArray
            Eccentricity (only elliptical and circular orbits are supported in TidalPy)
            Optional
        set_by_planet : bool = False
            A flag used by TidalPy planets to make calls to this method. Leave it False for user use.
            If set to true then other models and parameters will not update correctly.

        See Also
        --------
        OrbitBase.set_orbit
        """

        planet_pointer = self.find_planet_pointer(planet_reference)
        planet_loc = planet_pointer.orbit_location

        if planet_pointer is self.star:
            raise ParameterError("Can not change a star's orbit")
        elif planet_pointer is self.host and not self.star_host and not set_by_planet:
            if debug_mode:
                # This would change the orbit between the host and the star, not between the target body and the star.
                # This may not be what the user wants to do.
                log("Attempting to change the host planet's orbit", level='debug')

        if type(new_eccentricity) != np.ndarray:
            new_eccentricity = np.asarray(new_eccentricity)

        self._eccentricities[planet_loc] = new_eccentricity

        if not set_by_planet:
            # Need to tell the planet to update any orbital-dependent state
            planet_pointer.update_orbit()

    def set_inclination(self, planet_reference: PlanetRefType, new_inclination: FloatArray, set_by_planet: bool = False):
        """ Set the inclination of a planet at planet_loc

        Use Orbit.set_orbit if setting more than one state parameter.

        Parameters
        ----------
        planet_reference : PlanetRefType
            Reference used to find the planet
        new_inclination : FloatArray
            Orbital inclination relative to the orbital plane in [rads]
            Optional
        set_by_planet : bool = False
            A flag used by TidalPy planets to make calls to this method. Leave it False for user use.
            If set to true then other models and parameters will not update correctly.

        See Also
        --------
        OrbitBase.set_orbit
        """

        planet_pointer = self.find_planet_pointer(planet_reference)
        planet_loc = planet_pointer.orbit_location

        if planet_pointer is self.star:
            raise ParameterError("Can not change a star's orbit")
        elif planet_pointer is self.host and not self.star_host and not set_by_planet:
            if debug_mode:
                # This would change the orbit between the host and the star, not between the target body and the star.
                # This may not be what the user wants to do.
                log("Attempting to change the host planet's orbit", level='debug')

        if type(new_inclination) != np.ndarray:
            new_inclination = np.asarray(new_inclination)

        self._inclinations[planet_loc] = new_inclination

        if not set_by_planet:
            # Need to tell the planet to update any orbital-dependent state
            planet_pointer.update_orbit()

    # Functionality to get the orbit state variables
    def get_orbital_freq(self, planet_reference: PlanetRefType) -> np.ndarray:

        planet_loc = self.find_planet_pointer(planet_reference).orbit_location
        return self._orbital_freqs[planet_loc]

    def get_semi_major_axis(self, planet_reference: PlanetRefType) -> np.ndarray:

        planet_loc = self.find_planet_pointer(planet_reference).orbit_location
        return self._semi_major_axis[planet_loc]

    def get_eccentricity(self, planet_reference: PlanetRefType) -> np.ndarray:

        planet_loc = self.find_planet_pointer(planet_reference).orbit_location
        return self._eccentricities[planet_loc]

    def get_inclination(self, planet_reference: PlanetRefType) -> np.ndarray:

        planet_loc = self.find_planet_pointer(planet_reference).orbit_location
        return self._inclinations[planet_loc]

    def calculate_insolation(self, planet_reference: PlanetRefType, set_planet_param: bool = True):
        """ Calculate the insolation heating received by a planet located at planet_loc

        The star-planet separation (semi-major axis) and eccentricity are used to estimate the orbit-averaged
        insolation heating received at the surface of the target planet. If the star is the host of the system then
        the target body's semi-a and eccentricity will be used. Otherwise the orbit.host's parameters will be used.

        The actual method used to make the calculation is stored in the target body's equilibrium_insolation_func.
        It is set by the planet's configuration---the difference between the models is how they handle an eccentric
        orbit.

        Parameters
        ----------
        planet_reference : PlanetRefType
            Reference used to find the planet
        set_planet_param : bool = True
            If the method should set the planet's insolation_heating state variable or not

        Returns
        -------
        insolation_heating : FloatArray
            The orbit averaged heating received at the surface of the target planet in [Watts]
        """

        planet_pointer = self.find_planet_pointer(planet_reference)
        planet_loc = planet_pointer.orbit_location

        if planet_pointer is self.star:
            raise ParameterError('Can not calculation insolation heating for the star.')

        # These separations and eccentricities should be relative to the star!
        star_separation = self.equilib_distance_funcs[planet_loc]()
        if star_separation is None:
            raise ParameterMissingError

        star_eccentricity = self.equilib_eccentricity_funcs[planet_loc]()
        if star_eccentricity is None:
            if debug_mode:
                log('Attempting to calculate insolation heating with no eccentricity set. Using e=0.')
            star_eccentricity = np.asarray(0.)

        insolation_heating = planet_pointer.equilibrium_insolation_func(
                self.star.luminosity, star_separation, planet_pointer.albedo, planet_pointer.radius,
                star_eccentricity)

        if set_planet_param:
            planet_pointer.insolation_heating = insolation_heating

        return insolation_heating

    @property
    def host_tide_raiser_loc(self) -> int:
        return self._host_tide_raiser_loc

    @host_tide_raiser_loc.setter
    def host_tide_raiser_loc(self, value: int):

        if value > len(self.all_objects):
            raise ParameterError('Host tide raiser location must be the orbit location of one of the target bodies.')
        elif value in [0, 1]:
            raise ParameterError('Host tide raiser location can not be the host or stars locations')
        elif value < 0:
            raise ParameterError('Host tide raiser location must be positive')

        self._host_tide_raiser_loc = value

        # Now update the host planet with relevant information
        pseudo_host_reference = self.all_objects[self.host_tide_raiser_loc]
        self.host.tidal_susceptibility_inflated = (3. / 2.) * G * pseudo_host_reference.mass**2 * \
                                                  self.host.radius**5
        self.host.tide_raiser_ref = pseudo_host_reference

    def __iter__(self):
        return iter(self.target_bodies)