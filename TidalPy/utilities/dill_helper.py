import os
from typing import Tuple

from send2trash import send2trash

from .. import __version__
from .pathing import get_all_files_of_type
from ..io import tidalpy_loc

dilled_planets_loc = os.path.join(tidalpy_loc, 'planets', 'dilled_planets')


def delete_planets(ask_prompt: bool = True):
    """ Will delete all planets in the dilled_planets folder """

    if ask_prompt:
        response = input("You are about to delete all planets from TidalPy's dilled planet folder!\n"
                         "    These files make loading planets much faster for TidalPy. Only delete as a last resort "
                         "troubleshooting process or if planets have become outdated.\n"
                         "Proceed with deletion? [y/n]: ")
        if response.lower().strip() in ['y', 'yes']:
            pass
        else:
            return False

    for filename, filepath in get_all_files_of_type(dilled_planets_loc, ['dill', 'pickle']).items():
        send2trash(filepath)
    return True


def dill_file_path(object_name: str) -> Tuple[str, str]:
    name = f'{object_name}.TYPv{__version__}.dill'
    return name, os.path.join(dilled_planets_loc, name)