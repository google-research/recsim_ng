# Copyright 2021 The RecSim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
# Copyright 2020 The RecSim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup script for RecSim NG.

This script will install RecSim NG as a Python module.

See: https://github.com/google-research/recsim_ng

"""

from os import path
from setuptools import find_packages
from setuptools import setup

here = path.abspath(path.dirname(__file__))

install_requires = [
    'absl-py',
    'edward2',
    'gin-config',
    'gym',
    'matplotlib',
    'numpy',
    'tensorflow >= 2.3.0',
    'tensorflow-probability',
]

recsim_ng_description = (
    'RecSim NG: Toward Principled Uncertainty Modeling for Recommender Ecosystems'
)

with open('README.md', 'r') as fh:
  long_description = fh.read()

setup(
    name='recsim_ng',
    version='0.0.1',
    author='The RecSim Team',
    author_email='no-reply@google.com',
    description=recsim_ng_description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/google-research/recsim_ng',
    packages=find_packages(exclude=['docs']),
    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=install_requires,
    python_requires='>=3.6',
    project_urls={  # Optional
        'Documentation': 'https://github.com/google-research/recsim_ng',
        'Bug Reports': 'https://github.com/google-research/recsim_ng/issues',
        'Source': 'https://github.com/google-research/recsim_ng',
    },
    license='Apache 2.0',
    keywords='recsim reinforcement-learning recommender-system simulation probabilistic-programming'
)
