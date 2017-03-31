#!/usr/bin/env python
"""Cell Tracking
Cell Tracking is a package for doing TITAN style storm cell tracking using
pyart grid objects.
"""

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

DOCLINES = __doc__.split('\n')
NAME = 'cell_tracking'
MAINTAINER = 'Mark Picel'
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = '\n'.join(DOCLINES[1:])
LICENSE = 'BSD'
PLATFORMS = 'Linux'
MAJOR = 0
MINOR = 1
MICRO = 0
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


def configuration(parent_package='', top_path=None):
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage('tracking')
    return config


def setup_package():
    setup(
          name=NAME,
          maintainer=MAINTAINER,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          version=VERSION,
          license=LICENSE,
          platforms=PLATFORMS,
          configuration=configuration,
          include_package_data=True
          )

if __name__ == '__main__':
    setup_package()
