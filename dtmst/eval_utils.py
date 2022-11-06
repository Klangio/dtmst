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
"""Utilities for evaluation."""

# system packages
import os
from typing import Dict, List

# additional packages
import numpy as np
from tqdm.auto import tqdm
import mir_eval
import note_seq


def midi_notes_to_seq(notes):
  sequence = note_seq.NoteSequence()
  sequence.tempos.add().qpm = 120
  sequence.ticks_per_quarter = 220

  for n in notes:
    note = sequence.notes.add()
    note.start_time = n.start
    note.end_time = n.end
    note.pitch = int(n.pitch)

  return sequence


def note_seq_to_intervals_and_pitches(sequence):
  notes = list(filter(lambda note: note.end_time > note.start_time, sequence.notes))
  intervals = np.array([(note.start_time, note.end_time) for note in notes])
  pitches = np.array([note.pitch for note in notes])
  onsets = np.array(sorted([note.start_time for note in notes]))
  return intervals, pitches, onsets


def evaluate_example(seq_est, seq_gt):
  ref_intervals, ref_pitches, ref_onsets = note_seq_to_intervals_and_pitches(seq_gt)
  est_intervals, est_pitches, est_onsets = note_seq_to_intervals_and_pitches(seq_est)

  if len(seq_gt.notes) < 1:
    print('No notes in ground truth!')
    precision, recall, f_measure, avg_overlap_ratio = 0, 0, 0, 0
    offset_p, offset_r, offset_f1, offset_avg_overlap_ratio = 0, 0, 0, 0
    onsets_f1, onsets_p, onsets_r = 0, 0, 0
  elif len(seq_est.notes) < 1:
    print('No notes detected!')
    precision, recall, f_measure, avg_overlap_ratio = 0, 0, 0, 0
    offset_p, offset_r, offset_f1, offset_avg_overlap_ratio = 0, 0, 0, 0
    onsets_f1, onsets_p, onsets_r = 0, 0, 0
  else:
    # pitch within 50 cents, onset within 50 ms, offset within 50 ms, duration within 20%
    precision, recall, f_measure, avg_overlap_ratio = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches, offset_ratio=None)

    # pitch within 50 cents, onset within 50 ms, offset within 50 ms, duration within 20%
    offset_p, offset_r, offset_f1, offset_avg_overlap_ratio = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches)

    onsets_f1, onsets_p, onsets_r = mir_eval.onset.f_measure(ref_onsets, est_onsets)

  return {
      'p_onset': onsets_p,
      'r_onset': onsets_r,
      'f1_onset': onsets_f1,
      'p_note': precision,
      'r_note': recall,
      'f1_note': f_measure,
      'overlap_note': avg_overlap_ratio,
      'p_note_w_offset': offset_p,
      'r_note_w_offset': offset_r,
      'f1_note_w_offset': offset_f1,
      'overlap_note_w_offset': offset_avg_overlap_ratio
  }


def evaluate_dataset(wav_files: List[str], transcribe_function) -> Dict[str, float]:
  example_counter = 0
  metrics_avg = {
      'p_onset': 0,
      'r_onset': 0,
      'f1_onset': 0,
      'p_note': 0,
      'r_note': 0,
      'f1_note': 0,
      'overlap_note': 0,
      'p_note_w_offset': 0,
      'r_note_w_offset': 0,
      'f1_note_w_offset': 0,
      'overlap_note_w_offset': 0
  }
  pbar = tqdm(wav_files, total=len(wav_files))
  for fn_wav in pbar:
    if not os.path.exists(fn_wav):
      continue
    fn_mid = fn_wav.replace('.wav', '.mid')

    seq_gt = note_seq.midi_file_to_note_sequence(fn_mid)
    seq_est = transcribe_function(fn_wav)
    metrics = evaluate_example(seq_est, seq_gt)

    metrics_avg['p_onset'] += metrics['p_onset']
    metrics_avg['r_onset'] += metrics['r_onset']
    metrics_avg['f1_onset'] += metrics['f1_onset']
    metrics_avg['p_note'] += metrics['p_note']
    metrics_avg['r_note'] += metrics['r_note']
    metrics_avg['f1_note'] += metrics['f1_note']
    metrics_avg['overlap_note'] += metrics['overlap_note']
    metrics_avg['p_note_w_offset'] += metrics['p_note_w_offset']
    metrics_avg['r_note_w_offset'] += metrics['r_note_w_offset']
    metrics_avg['f1_note_w_offset'] += metrics['f1_note_w_offset']
    metrics_avg['overlap_note_w_offset'] += metrics['overlap_note_w_offset']
    example_counter += 1

    # print informations
    pbar.set_description('[Evaluation]')
    pbar.set_postfix({
        'f1_onset': metrics_avg['f1_onset'] / example_counter,
        'f1_note': metrics_avg['f1_note'] / example_counter,
        'f1_note_w_offset': metrics_avg['f1_note_w_offset'] / example_counter,
    })

  metrics_avg = {key: metrics_avg.get(key, 0) / example_counter for key in set(metrics_avg)}

  return metrics_avg
