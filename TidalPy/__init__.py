from ..setup import version

import time
__version__ = version
_vers_major, _vers_minor, _vers_hotfix = version.split('.')
compatibility_signature = _vers_major + _vers_minor

def is_compatible(test_version: str):
    """ Tests rather or not a feature made in test_version is likely to be compatible in current __version__

    """
    test_vers_major, test_vers_minor, test_vers_hotfix = test_version.split('.')
    if int(test_vers_major) > int(_vers_major):
        return False
    if int(test_vers_major) < int(_vers_major):
        # Major releases may not be backwards compatible
        return False
    if int(test_vers_minor) > int(_vers_minor):
        return False
    if int(test_vers_minor) <= int(_vers_minor):
        # Minor releases should be backwards compatible
        pass

    return True

# Initial Runtime
init_time = time.time()

# debug_mode is an optional runtime mode which will call upon many more checks. The checks will help minimize bugs,
#   but can slow down TidalPy. It is reccomended that you always run a case (at least partially) with debug_mode=True
#   first, and then once you have confirmed that no obvious bugs are present, you can re-run/finish the run with it off.
debug_mode = True

# auto_write determines if TidalPy will automatically create output directories and save logs, data, plots, etc. to it.
auto_write = True

# verbose_level determines what logging information will be displayed to standard output.
# Set to 0 or False for no printing including warnings
# Set to 5 for all printing
# verbose_level = 1 only permits warnings
# Does not affect print()
verbose_level = 3

# Make logger that will be used by other
from .utilities.logging import TidalLogger
log = TidalLogger()

# Some data files are stored in the TidalPy directory chain. Need to know where TidalPy is located to find those files.
