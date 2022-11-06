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
"""DTMST Model classes."""

# additional packages
import torch
import torch.nn.functional as F
import torch.nn as nn


class OnsetNet(nn.Module):
  """OnsetNet model class."""

  def __init__(self, hparams):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 16, 5, padding='same')
    self.conv2 = nn.Conv2d(16, 32, 5, padding='same')
    self.conv3 = nn.Conv2d(32, 64, 5, padding='same')
    self.dropout = nn.Dropout(0.5)
    self.fc1 = nn.Linear(64 * hparams['spec_n_bins'], 1)

  def forward(self, x):
    x = x.permute(0, 3, 2, 1)
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    x = torch.relu(self.conv3(x))
    x = self.dropout(x)
    x = x.permute(0, 3, 2, 1)
    x = torch.flatten(x, 2, 3)
    x = torch.sigmoid(self.fc1(x))
    return x


class PitchNet(nn.Module):
  """PitchNet model class."""

  def __init__(self, hparams):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 16, (9, 3), padding='same')
    self.conv2 = nn.Conv2d(16, 32, (5, 3), padding='same')
    self.dropout = nn.Dropout(0.5)
    self.fc1 = nn.Linear(32 * hparams['spec_n_bins'], hparams['max_midi'] - hparams['min_midi'] + 1)

  def forward(self, x):
    x = x.permute(0, 3, 2, 1)
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    x = self.dropout(x)
    x = x.permute(0, 3, 2, 1)
    x = torch.flatten(x, 2, 3)
    x = self.fc1(x)
    return x


class DTMSTModel(nn.Module):
  """DTMST model class."""

  def __init__(self, hparams):
    super().__init__()
    self.onsetnet = OnsetNet(hparams)
    self.pitchnet = PitchNet(hparams)

  def forward(self, x):
    x_onset = self.onsetnet(x)
    x_pitch = self.pitchnet(x)
    return x_onset, x_pitch

  def run_on_batch(self, features, labels_onset, labels_pitch):
    outputs_onset, outputs_pitch = self(features)
    loss_onsets = F.binary_cross_entropy(outputs_onset, labels_onset)
    loss_pitches = F.cross_entropy(
        outputs_pitch.permute([0, 2, 1]),
        labels_pitch.argmax(axis=2),
    )
    loss = loss_onsets + loss_pitches

    outputs = {'onsets': outputs_onset, 'pitches': outputs_pitch}
    losses = {'total': loss, 'onsets': loss_onsets, 'pitches': loss_pitches}
    return outputs, losses
