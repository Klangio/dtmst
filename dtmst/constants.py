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
"""Constants for model."""

# system imports
from typing import Dict


def HParams() -> Dict[str, any]:
  return {
      'sample_rate': 16000,

      # for dataset creation
      'min_length': 5,
      'max_length': 30,

      # for feature extraction
      'features': 'spectrogram',
      'spec_type': 'mel',
      'spec_mel_htk': False,
      'spec_diff': False,
      'spec_log_scaling': True,
      'spec_hop_length': 512,
      'spec_n_bins': 229,
      'spec_n_fft': 2048,
      'spec_fmin': 55.0,

      # for labels
      'labels': 'onsets_1d',
      'onset_mode': 'window',
      'onset_length': 100,
      'offset_length': 100,
      'onset_delay': 0,
      'onset_window': 1,
      'onset_smooth': False,

      # for midi based input features
      'min_midi': 40,
      'max_midi': 88,
      'const_length': 938 * 2,  # 938,

      # training parameters
      'epochs': 100,
      'lr': 1e-4,
      'batch_size': 32
  }
