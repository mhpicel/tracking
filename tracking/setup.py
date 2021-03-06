#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup for cell tracking package
"""
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration


def configuration(parent_package='', top_path=None):
    config = Configuration('tracking', parent_package, top_path)
    config.add_subpackage('core')
    return config

if __name__ == '__main__':
    setup(**configuration(top_path='').todict())
