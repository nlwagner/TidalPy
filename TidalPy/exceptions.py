class TidalPyException(Exception):
    """ Default exception for all TidalPy-specific errors
    """

    default_message = 'A Default TidalPy Error Occurred.'

    def __init__(self, *args, **kwargs):

        # If no input is provided then the base exception will look at the class attribute 'default_message'
        #   and send that to sys.stderr
        if args or kwargs:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(self.default_message)


# Specific Exceptions
class BadArrayShape(TidalPyException):
    default_message = 'TidalPy requires certain arrays maintain the same shape for all layers and planets. ' \
                      'It has found an array with an unexpected shape.'


class BadAttributeValueError(TidalPyException):
    default_message = 'Bad value found in attribute setter.'


class ArgumentError(TidalPyException):
    default_message = 'There was an issue with one or more arguments provided to a TidalPy function or method.'


class ImplementationError(TidalPyException):
    default_message = 'Tried to use functionality that is not yet implemented.'


class ImplementedBySubclassError(ImplementationError):
    default_message = 'Trying to access sub-class functionality from a base class.'


class ReinitError(TidalPyException):
    default_message = 'One or more critical parameters have changed since planet was made. Construct new planet instead.'


class ReinitNotAllowedError(ReinitError):
    default_message = 'This class should be fully re-initialized upon load. Partial reinit (via self.reinit()) is not supported.'


class UnknownModelError(TidalPyException):
    default_message = 'A selected model, parameter, or switch is not currently supported.'


class UnknownTidalPyConfigValue(UnknownModelError):
    default_message = 'A configuration set in TidalPy.configurations is not know or has not yet been implemented.'


class ParameterMissingError(TidalPyException):
    default_message = 'One or more parameter(s) or configuration(s) are missing and have no defaults. ' \
                      'Check that keys have correct spelling and capitalization.'


class MissingArgumentError(ArgumentError):
    default_message = 'One or more required argument(s) and/or key-word argument(s) were not provided.'


class ParameterError(TidalPyException):
    default_message = 'One or more parameters are not supported as specified.'


class IncompatibleModelError(ParameterError):
    default_message = 'One or more model parameters are not compatible with each other'


class ImproperAttributeHandling(TidalPyException):
    default_message = 'The attribute you are attempting to set must be set by a different class or method.'


class MissingAttributeError(TidalPyException):
    default_message = 'The attribute you are attempting to access has not been set.'


class IncorrectAttributeType(TidalPyException):
    default_message = 'An attribute was set with incorrect type.'


class IncorrectArgumentType(ArgumentError):
    default_message = 'A method or function argument was provided the incorrect type.'


class AttributeNotSetError(IncorrectAttributeType):
    default_message = 'An attribute has not been changed from its default value.'


class BadValueError(TidalPyException):
    default_message = 'An unrealistic value was encountered.'


class UnusualRealValueError(BadValueError):
    default_message = 'An usually large or small value was encountered for a parameter.' \
                      'Confirm proper dimensional units.'


class IncorrectModelInitialized(TidalPyException):
    default_message = 'The currently set model does not support the functionality that you are attempting to use.'
