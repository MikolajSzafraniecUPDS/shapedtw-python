#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
import numpy


# --------------------
with open('README.md') as readme_file:
    readme_md = readme_file.read()


setup(
    author="Mikołaj Szafraniec",
    author_email='mikolaj.szafraniec.upds@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12'
    ],
    description="Extension of the dtw-python package, implementing shape dtw variant of the algorithm.",
    install_requires=['numpy>=1.22', 'scipy>=1.8', 'dtw>=1.1', 'PyWavelets>=1.3', 'pandas>=1.5', 'matplotlib>=3.5'],
    tests_require=["pytest"],
    python_requires='>=3.9',
    license="GNU General Public License v3",
    long_description=readme_md,
    long_description_content_type="text/markdown",
    include_package_data=False,
    keywords=['dtw', 'shapedtw', 'timeseries'],
    name='shapedtw',
    packages=['shapedtw'],
    include_dirs=numpy.get_include(),
    url='https://mikolajszafraniecupds.github.io/shapedtw-python/',
    version='0.0.12',
    zip_safe=False,
)