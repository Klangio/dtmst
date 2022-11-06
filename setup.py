# Copyright 2022 Klangio GmbH.
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
"""A setuptools based setup module for note-seq."""

import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

with open("requirements.txt") as f:
  requirements = f.read().splitlines()

package_data = [
    "checkpoints/*",
]

setuptools.setup(
    name="dtmst",
    version="0.0.1",
    author="Klangio GmbH",
    author_email="info@klangio.com",
    description="DTMST Singing Note Tracking Module",
    url="https://klangio.com",
    packages=setuptools.find_packages(exclude=["tests", "scripts", "docs"]),
    package_data={"dtmst": package_data},
    install_requires=requirements,
    python_requires=">=3.8",
)
