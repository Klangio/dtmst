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
"""Evaluate DTMST Model."""

# system imports
import os
import sys
import glob

# additional imports
import torch
import librosa

# internal imports
sys.path.insert(0, os.path.abspath('../'))
# pylint: disable=C0413
from dtmst import DTMSTModel, data, infer_utils, eval_utils

fn_checkpoint = '../checkpoints/checkpoint_sinsy_logmel.pth'

checkpoint = torch.load(fn_checkpoint, map_location=torch.device('cpu'))
hparams = checkpoint['hparams']
model = DTMSTModel(hparams)
model.load_state_dict(checkpoint['model_state'])
model.eval()

wav_files = glob.glob('../datasets/singreal/*.wav')


def transcribe_dtmst(fn_audio: str):
  x, _ = librosa.load(fn_audio, sr=hparams['sample_rate'])
  x = librosa.util.normalize(x)

  mel = data.wav_to_mel(x, hparams)

  example = {'spectrogram': torch.from_numpy(mel).float().unsqueeze(2), 'audio': x}
  return infer_utils.transcribe(example, model, hparams)


if __name__ == '__main__':
  # pylint: disable=E1120
  metrics_avg = eval_utils.evaluate_dataset(wav_files, transcribe_dtmst)

  print('### Average Results ###')
  print('[Onsets] Precision: {:.2%}, Recall: {:.2%}, F-Measure: {:.2%}'.format(
      metrics_avg['p_onset'], metrics_avg['r_onset'], metrics_avg['f1_onset']))
  print(
      '[Note] Precision: {:.2%}, Recall: {:.2%}, F-Measure: {:.2%}, Average Overlap Ratio: {:.2%}'.
      format(
          metrics_avg['p_note'],
          metrics_avg['r_note'],
          metrics_avg['f1_note'],
          metrics_avg['overlap_note'],
      ))
  print(
      '[Note w. Offset] Precision: {:.2%}, Recall: {:.2%}, F-Measure: {:.2%}, Average Overlap Ratio: {:.2%}'
      .format(
          metrics_avg['p_note_w_offset'],
          metrics_avg['r_note_w_offset'],
          metrics_avg['f1_note_w_offset'],
          metrics_avg['overlap_note_w_offset'],
      ))
