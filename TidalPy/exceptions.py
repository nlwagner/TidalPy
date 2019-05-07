import sys


class TidalPyException(RuntimeError):
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


class UnknownModelError(TidalPyException):

    default_message = 'A selected model, parameter, or switch is not currently supported.'


class ParameterMissingError(TidalPyException):

    default_message = 'One or more parameters are missing and have no fallbacks.' \
                      'Check that keys have correct spelling and capitalization.'

class ParameterError(TidalPyException):

    default_message = 'One or more parameters are not supported as specified.'


class ImproperAttributeHandling(TidalPyException):

    default_message = 'The attribute you are attempting to set must be set by a different class or method.'


class MissingAttributeError(TidalPyException):

    default_message = 'The attribute you are attempting to access has not been set.'


class IncorrectAttributeType(TidalPyException):

    default_message = 'An attribute was set with incorrect type.'


class AttributeNotSet(IncorrectAttributeType):

    default_message = 'An attribute has not been changed from its default value.'


class BadAttributeValueError(TidalPyException):

    default_message = 'Bad value found in attribute setter.'


class UnusualRealValueError(BadAttributeValueError):

    default_message = 'An usually large or small value was encountered for a parameter.' \
                      'Confirm proper dimensional units.'
