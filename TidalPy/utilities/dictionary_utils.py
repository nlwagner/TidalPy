import copy
from typing import List, Tuple, Union, Any


PossibleKeyType = Union[str, int, float]


def nested_get(input_dict: dict,
               nested_keys: Union[PossibleKeyType, List[PossibleKeyType, ...], Tuple[PossibleKeyType, ...]],
               default: Any = None,
               raiseon_nolocate: bool = False) -> Any:
    """ Returns a value from a series of nested dictionaries given a list of keys

    Parameters
    ----------
    input_dict : dict
        Dictionary containing nested dictionary(ies)
    nested_keys : Union[PossibleKeyType, List[PossibleKeyType, ...], Tuple[PossibleKeyType, ...]]
        List of keys to search the dictionary with.
    default : Any = None
        Default value if key can't be found
    raiseon_nolocate : bool = False
        If key can't be found and this is True then a KeyError exception will be raised.

    Returns
    -------
    value : Any
        Value stored in the nested dictionaries
    """
    # TODO: When numba .44 comes out test this with @njit to speed it up. That version of numba should support
    #     dict types
    #     --Update: The new version of numba is out, but you need to specifiy the types of the keys and items.
    #         So I don't think njiting is going to work except for very specfic cases.

    if type(nested_keys) not in [str, int, float, list, tuple]:
        raise TypeError

    internal_dict_value = input_dict
    if type(nested_keys) not in [tuple, list]:
        nested_keys = [nested_keys]

    for key in nested_keys:
        if type(key) not in [str, int, float]:
            # Items in nested_key can not be lists or Tuples
            raise TypeError

        if type(internal_dict_value) != dict:
            raise KeyError('There are more nested keys than there are nested dicts.')

        if key not in internal_dict_value:
            if raiseon_nolocate:
                raise KeyError
            else:
                return default
        else:
            internal_dict_value = internal_dict_value[key]

    return internal_dict_value


def nested_place(replacement_value: Any, dict_to_overwrite: dict,
                 nested_keys: Union[PossibleKeyType, List[PossibleKeyType, ...], Tuple[PossibleKeyType, ...]],
                 make_copy: bool = False, retain_old_value: bool = False) -> dict:
    """ Replaces a nested-dictionary value based on a list of keys.

    Parameters
    ----------
    replacement_value : Any
        New value that will replace any old values.
    dict_to_overwrite : dict
        Dictionary whose value will be replaced.
    nested_keys : Union[PossibleKeyType, List[PossibleKeyType, ...], Tuple[PossibleKeyType, ...]]
        List-like container of keys required to get to the location of the replacement value.
        The last key's value will be replaced by replacement_value.
    make_copy : bool = False
        If true then a new dictionary will be made (copy of dict_to_overwrite).
        In-place (mutation) replacement will not take place.
    retain_old_value : bool = False
        If true then any old value found will be saved with the key "<name of last key in list>_OLD"

    Returns
    -------
    dict_to_overwrite : dict
        Dictionary after the replacement has been made.
        Note: if make_copy is True then this will not be the same dictionary reference that was originally passed in.

    """

    if type(nested_keys) not in [str, int, float, list, tuple]:
        raise TypeError

    if make_copy:
        # Make a copy of the input dict so its original values remain unchanged. The copy will be returned to the user.
        dict_to_overwrite = copy.deepcopy(dict_to_overwrite)

    def _retain(_key: PossibleKeyType, _old_value: Any, _dict_ref: dict):
        """ Helper function to make a copy of previous value
        """
        if retain_old_value:
            i = 0
            while True:
                if i > 50:
                    raise StopIteration('A large number of retained key values found during nested placement.')
                if i == 0:
                    retain_key_attempt = f'{_key}_OLD'
                else:
                    retain_key_attempt = f'{_key}_OLD{i}'

                if retain_key_attempt not in _dict_ref:
                    _dict_ref[retain_key_attempt] = _old_value
                    break
                i += 1

        else:
            # Do Nothing
            pass

    if type(nested_keys) not in [tuple, list]:
        # No nesting: simply override the value stored at the key
        _retain(nested_keys, dict_to_overwrite[nested_keys], dict_to_overwrite)
        dict_to_overwrite[nested_keys] = replacement_value
    else:
        # Work forwards to find the location of the last nested dictionary, then replace its value at the last key.
        last_i = len(nested_keys) - 1
        dict_ref = dict_to_overwrite
        for k_i, key in enumerate(nested_keys):
            if k_i == last_i:
                # Make the replacement
                _retain(key, dict_ref[key], dict_ref)
                dict_ref[key] = replacement_value
            else:
                # Update reference
                dict_ref = dict_ref[key]

    return dict_to_overwrite