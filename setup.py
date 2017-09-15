#!/usr/bin/env python

from setuptools import setup

version = '1.0.0'

required = open('requirements.txt').read().split('\n')

setup(
    name='spagen',
    version=version,
    description='spatial genetic models',
    author='Joseph Marcus / Hussein Al-Asadi',
    author_email='jhmarcus@uchicago.edu / halasadi@uchicago.edu',
    url='https://github.com/jhmarcus/spagen',
    packages=['spagen'],
    install_requires=required,
    long_description='See ' + 'https://github.com/jhmarcus/spagen',
    license='MIT'
)
