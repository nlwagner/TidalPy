import operator
from typing import Tuple, Callable

from ..config.config import ConfigHolder
from ..config.dictionaryUtils import nested_get, nested_place
from .... import debug_mode, log
from ....exceptions import (ImplementedBySubclassError, MissingArgumentError, ParameterMissingError,
                            UnknownModelError, AttributeNotSetError, ImproperPropertyHandling,
                            OuterscopePropertySetError)


class ModelHolder(ConfigHolder):

    """ Parent class for physics models

    Provides basic functionality to load in default model inputs and run calculations using those inputs.
    """

    known_models = None
    known_model_const_args = None
    known_model_live_args = None

    def __init__(self, model_name: str = None, replacement_config: dict = None, auto_build_inputs: bool = True):

        # Setup parent class
        super().__init__(replacement_config=replacement_config)

        # Pull out model information, check if it is a valid model name, then store it
        if model_name is None and 'model' in self.config:
            model_name = self.config['model']

        # Store model name information
        self._model = model_name

        # Attempt to find the model's function information
        self._func = None
        # Some functions can support arrays and floats interchangeably. Others require separate functions.
        #    These functions should be stored separately.
        self._func_array = None
        # If a separately defined function was defined then set this flag to True.
        self._func_array_defined = False

        # Functions may have separately inputs. These are stored as inputs or live inputs:
        #    inputs: tuple of constants passed to the self.func
        #    live_inputs: tuple of inputs that may change after TidalPy is loaded (e.g., the viscosity of a layer)
        self.get_live_args = None
        self.live_inputs = None
        self.inputs = None
        self._constant_arg_names = None
        self._live_arg_names = None

        # Build constant and live inputs, and setup self.func
        if auto_build_inputs:
            self.build_inputs()

        # Switch between calculate and calculate debug. Generally, _calculate_debug is a much slower function that
        #    includes additional checks
        self._calc = self._calculate
        if '_calculate_debug' in self.__dict__:
            # If a debug function is available use it as the primary calculator in debug mode.
            # Give this function the same documentation as the regular calculate but with prepended text stating
            #     that it is a debug version
            self._calculate_debug.__doc__ = 'DEBUG MODE ENABLED\n'
            if self._calculate.__doc__ not in [None, '']:
                self._calculate_debug.__doc__ += self._calculate.__doc__

            if debug_mode:
                self._calc = self._calculate_debug
        else:
            if debug_mode:
                log.debug(f'Debug mode is on, but it appears that no debug calculation method has been implemented '
                          f'for {self.__class__.__name__}. Using regular calculate method.')

        # TODO: The below breaks when calculate is a method. Not sure if there is a solution:
        #    Error: AttributeError: attribute '__doc__' of 'method' objects is not writable
        # # Give calculate the same doc string as whatever is store in _calc (debug or regular)
        # if self._calc.__doc__ not in [None, '']:
        #     self.calculate.__doc__ = self._calc.__doc__

    def build_inputs(self):
        """ Builds the live and constant input tuples for the model's calculate function.
        """

        # Try to find the provided model and load in its main function
        try:
            self._func = self.known_models[self.model]
        except KeyError:
            try:
                self._func = self.known_models[self.model.lower()]
            except KeyError:
                log.error(f'Unknown model: "{self.model}" for {self.__class__.__name__}')
                raise UnknownModelError(f'Unknown model: {self.model} for {self.__class__.__name__}')
            else:
                # Model has capitalization that does not match TidalPy
                log.warning(f'Model "{self.model}" for {self.__class__.__name__} does not match TidalPy capitalization. '
                            f'Changing to {self.model.lower()}.')
                self._model = self.model.lower()

        # Load in other related functionality
        if self.model + '_array' in self.known_models:
            self._func_array = self.known_models[self.model + '_array']
            self._func_array_defined = True
        else:
            # No separate array function found. Try to use the regular float version
            self._func_array = self.func
            self._func_array_defined = False
        self._constant_arg_names = self.known_model_const_args[self.model]
        self._live_arg_names = self.known_model_live_args[self.model]

        # Pull out constant arguments
        self.inputs = self.build_args(self._constant_arg_names, parameter_dict=self.config, is_live_args=False)

        # Build live argument functions and calls
        self.live_inputs = None
        if len(self._live_arg_names) == 0:
            self.get_live_args = None
        else:
            live_funcs = self.build_args(self._live_arg_names, is_live_args=True)
            self.get_live_args = lambda: tuple([live_func(self) for live_func in live_funcs])

    def calculate(self, *args, **kwargs):

        # Some models have inputs that need to be updated at each call
        if self.get_live_args is not None:
            try:
                self.live_inputs = self.get_live_args()
            except AttributeError:
                raise AttributeNotSetError('One or more live arguments are not set or their references were not found.')

        return self._calc(*args, **kwargs)

    def _calculate(self, *args, **kwargs):
        raise ImplementedBySubclassError

    # def _calculate_debug(self, *args, other_inputs: tuple = tuple(), **kwargs):
    #     raise ImplementedBySubclassError

    @staticmethod
    def build_args(arg_names: Tuple[str, ...], parameter_dict: dict = None, is_live_args: bool = False):
        """ Build an input tuple based on the required, constant, arguments and a parameter dictionary

        Parameters
        ----------
        arg_names : Tuple[str, ...]
            List of required constant argument names needed for this model's function.
        parameter_dict : dict
            Dictionary of parameters.
        is_live_args : bool
            Flag if this call is looking for live args, rather than constant args.

        Returns
        -------
        args : Tuple[Any, ...]
            List of default argument parameters
        """

        args = list()

        if not is_live_args:
            # Constant Arguments
            if parameter_dict is None:
                log.error(f'Parameter configuration dictionary is required to build constant model arguments.')
                raise MissingArgumentError(f'Parameter configuration dictionary is required to build constant '
                                           f'model arguments.')

            for arg_name in arg_names:
                if arg_name not in parameter_dict:
                    log.error(f'Parameter: {arg_name} is missing from configuration dictionary.')
                    raise ParameterMissingError(f'Parameter: {arg_name} is missing from configuration dictionary.')
                args.append(parameter_dict[arg_name])
        else:
            # Live Arguments
            for live_arg_signature in arg_names:

                # Create a attrgetter for the live argument.
                if 'self.' in live_arg_signature:
                    live_arg_signature = live_arg_signature.split('self.')[1]

                getter_func = operator.attrgetter(live_arg_signature)
                args.append(getter_func)

        return tuple(args)

    # State properties
    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value):
        raise ImproperPropertyHandling

    @property
    def func(self) -> Callable:
        return self._func

    @func.setter
    def func(self, value):
        raise ImproperPropertyHandling

    @property
    def func_array(self) -> Callable:
        return self._func_array

    @func_array.setter
    def func_array(self, value):
        raise ImproperPropertyHandling

    @property
    def func_array_defined(self) -> bool:
        return self._func_array_defined

    @func_array_defined.setter
    def func_array_defined(self, value):
        raise ImproperPropertyHandling


class LayerModelHolder(ModelHolder):

    """ Parent class for physics models that are stored within a planet's layer

    Provides basic functionality to load in default model inputs and run calculations using those inputs and the layer's
        current state properties (e.g., temperature).

    See Also
    --------
    cooling/
    radiogenics/
    rheology/
    """

    model_config_key = None

    def __init__(self, layer, model_name: str = None,
                 store_config_in_layer: bool = True, auto_build_inputs: bool = True):

        # Store layer and world information
        self._layer = layer
        self._world = None
        if 'world' in layer.__dict__:
            self.world = layer.world

        # The layer's type is used to pull out default parameter information
        self.default_config_key = self.layer_type

        # Record if model config should be stored back into layer's config
        self.store_config_in_layer = store_config_in_layer

        config = None
        try:
            config = nested_get(self.layer.config, self.model_config_key, raiseon_nolocate=True)
        except KeyError:
            log.debug(f"User provided no model information for {self} ({self.layer}); using defaults instead.")

        if config is None and self.default_config is None:
            log.error(f"Config not provided for {self} ({self.layer}); and no defaults are set.")
            raise ParameterMissingError(f"Config not provided for {self} ({self.layer}); and no defaults are set.")

        # Setup ModelHolder and ConfigHolder classes. Using the layer's config file as the replacement config.
        super().__init__(model_name=model_name, replacement_config=config, auto_build_inputs=auto_build_inputs)

        if self.store_config_in_layer:
            # Once the configuration file is constructed (with defaults and any user-provided replacements) then
            #    store the new config in the layer's config, overwriting any previous parameters.
            nested_place(self.config, self.layer.config, self.model_config_key, make_copy=False, retain_old_value=True)

    # State properties
    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, value):
        raise ImproperPropertyHandling

    @property
    def world(self):
        return self._world

    @world.setter
    def world(self, value):
        raise ImproperPropertyHandling

    # Outer-scope Properties
    @property
    def layer_type(self):
        return self.layer.type

    @layer_type.setter
    def layer_type(self, value):
        raise OuterscopePropertySetError

    def __str__(self):
        return f'{self.__class__.__name__} ({self.layer})'