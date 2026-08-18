# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import sys
import setuptools

if sys.version_info < (3,):
    raise Exception("Python 2 is not supported by hcu-megatron.")

__description__ = 'hcu-megatron of Hygon'
__version__ = '0.18.2+das.opt1.dtk2604'
__author__ = 'Hygon'
__keywords__ = 'hcu-megatron, language, deep learning, NLP'
__package_name__ = 'hcu-megatron'
__contact_names__ = 'Hygon'

try:
    with open("README.md", "r") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ''


setuptools.setup(
    name=__package_name__,
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=__contact_names__,
    maintainer=__contact_names__,
    classifiers=[
        # Supported python versions
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        # Additional Setting
        'Environment :: Console',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    packages=setuptools.find_namespace_packages(include=["hcu_megatron", "hcu_megatron.*"]),
    # Add in any packaged data.
    include_package_data=True,
    install_package_data=True,
    zip_safe=False,
    # PyPI package information.
    keywords=__keywords__,
    cmdclass={},
    ext_modules=[]
)
