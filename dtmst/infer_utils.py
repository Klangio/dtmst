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
"""Utilities for inference."""

# additional packages
import numpy as np
from scipy.signal import argrelextrema
import torch
import librosa
import note_seq

# internal packages
from .data import hparams_frames_per_second


def transcribe(example, model, hparams):
  x = example['spectrogram']
  model.eval()
  with torch.no_grad():
    x = torch.unsqueeze(x.cpu(), 0)
    onset_logits, pitch_logits = model(x)
    onset_logits = onset_logits[0, :, :].cpu().detach().numpy()
    pitch_logits = pitch_logits[0, :, :].cpu().detach().numpy()

  onset_predictions = onset_logits > 0.5

  y = example['audio']
  duration = len(y) / hparams['sample_rate']
  # Compute a spectral flux onset strength envelope: max(0, S[f, t] - ref[f, t - lag])
  # onset_envelope = librosa.onset.onset_strength(y=y, sr=hparams['sample_rate'])
  onset_envelope = librosa.onset.onset_strength(y=y, sr=hparams['sample_rate'], max_size=2, lag=1)
  local_maxima = np.array(argrelextrema(onset_envelope,
                                        np.greater)) * duration / onset_envelope.shape[0]
  return logits_to_notes(onset_predictions, pitch_logits, hparams, local_maxima)


def logits_to_notes(onset_predictions, pitch_logits, hparams, local_maxima=None):
  frames_per_second = hparams_frames_per_second(hparams)
  frame_length_seconds = 1 / frames_per_second
  min_duration = 60 / (160 * 4)  # 16th note
  onsets = []
  for i, value in enumerate(onset_predictions):
    if value > 0.5 and i > 0 and onset_predictions[i - 1] < 0.5:
      onset_time = (i + 2) * frame_length_seconds
      if len(onsets) > 0:
        if onset_time - onsets[-1] >= min_duration:
          onsets.append(onset_time)
      else:
        onsets.append(onset_time)

  sequence = note_seq.NoteSequence()
  for onset in onsets:
    if local_maxima is not None:
      onset = local_maxima[0, np.argmin(np.abs(onset - local_maxima))]
    if int(frames_per_second * onset + 3) < len(pitch_logits):
      pitch = np.argmax(pitch_logits[int(frames_per_second * onset + 3), :]) + hparams['min_midi']
      sequence.notes.add(pitch=pitch, start_time=onset, end_time=onset + 0.1, velocity=80)

  # set note offset to next note onset
  for i in range(1, len(sequence.notes)):
    sequence.notes[i - 1].end_time = sequence.notes[i].start_time
  if len(sequence.notes) > 0:
    sequence.notes[-1].end_time = sequence.notes[-1].start_time + 1

  return sequence
