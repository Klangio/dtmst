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
"""Methods used for handling data."""

# system imports
import bisect

# additional imports
import numpy as np
import h5py
import librosa
import pretty_midi as pm
from note_seq import midi_io, music_pb2, audio_io, sequences_lib
import torch
from torch.utils.data import Dataset


def hparams_frames_per_second(hparams):
  """Compute frames per second as a function of HParams."""
  return hparams['sample_rate'] / hparams['spec_hop_length']


def wav_to_mel(y, hparams):
  """Transforms the contents of a wav file into a series of mel spec frames."""
  mel = librosa.feature.melspectrogram(
      y=y,
      sr=hparams['sample_rate'],
      n_fft=hparams['spec_n_fft'],
      hop_length=hparams['spec_hop_length'],
      fmin=hparams['spec_fmin'],
      n_mels=hparams['spec_n_bins'],
      htk=hparams['spec_mel_htk'],
  ).astype(np.float32)

  # Transpose so that the data is in [frame, bins] format.
  mel = mel.T

  if hparams['spec_log_scaling']:
    # log scaling
    mel = 10 * np.log10(1e-10 + mel)

  return mel


class TranscriptionHDF5Dataset(Dataset):
  """Transcription HDF5 Dataset class."""

  def __init__(self, h5_file, train, hparams, mir=False):
    super().__init__()
    self.h5_file = h5_file
    self.hparams = hparams
    self.train = train
    self.mir = mir
    with h5py.File(h5_file, 'r') as hdf:
      self.length = len(hdf.keys())

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    with h5py.File(self.h5_file, 'r') as hdf:
      example = hdf.get(str(idx))
      audio = np.array(example['audio'])
      sequence = np.array(example['sequence'])
      velocity_range = np.array(example['velocity_range'])

    ns = music_pb2.NoteSequence.FromString(sequence.astype(str))
    velocity_range = music_pb2.VelocityRange.FromString(velocity_range.astype(str))

    # Convert sequence to pianorolls
    roll = sequences_lib.sequence_to_pianoroll(
        ns,
        frames_per_second=hparams_frames_per_second(self.hparams),
        min_pitch=self.hparams['min_midi'],
        max_pitch=self.hparams['max_midi'],
        min_frame_occupancy_for_label=0,
        onset_mode=self.hparams['onset_mode'],
        onset_window=self.hparams['onset_window'],
        onset_length_ms=self.hparams['onset_length'],
        offset_length_ms=self.hparams['offset_length'],
        onset_delay_ms=self.hparams['onset_delay'],
        min_velocity=velocity_range.min,
        max_velocity=velocity_range.max,
    )
    (onsets, active) = (roll.onsets, roll.active)

    mel = wav_to_mel(audio, self.hparams)

    # fix lengths
    length = mel.shape[0]
    if self.train:
      length = self.hparams['const_length']

    length_delta = mel.shape[0] - length
    if length_delta < 0:
      mel = np.pad(mel, [(0, -length_delta), (0, 0)])
    elif length_delta > 0:
      mel = mel[:-length_delta]

    length_delta = roll.onsets.shape[0] - length
    if length_delta < 0:
      onsets = np.pad(onsets, [(0, -length_delta), (0, 0)])
      active = np.pad(active, [(0, -length_delta), (0, 0)])
    elif length_delta > 0:
      onsets = onsets[:-length_delta]
      active = active[:-length_delta]

    onsets_1d = np.amax(onsets, axis=1)

    return_dict = {
        'spectrogram': torch.from_numpy(mel).float().view(1, mel.shape[0], mel.shape[1]),
        'onsets_1d': torch.from_numpy(onsets_1d).float().view(onsets_1d.shape[0], 1),
        'labels': torch.from_numpy(active).float()
    }
    if self.mir:
      return_dict['sequence'] = ns
      return_dict['audio'] = audio

    return return_dict


def create_folds(fn_h5, num_splits=5):
  file_indices = []
  with h5py.File(fn_h5, 'r') as hdf:
    length = len(hdf.keys())
    fn_mid_last = ''
    for idx in range(length):
      example = hdf.get(str(idx))
      fn_mid = str(np.array(example['id']).astype(str))
      if fn_mid != fn_mid_last:
        file_indices.append(idx)
      fn_mid_last = fn_mid

  split_indices = []
  for i in range(num_splits):
    split_indices.append(file_indices[int(i / num_splits * len(file_indices))])

  split_ranges = []
  for i in range(num_splits):
    if i < num_splits - 1:
      split_ranges.append(list(np.arange(split_indices[i], split_indices[i + 1])))
    else:
      split_ranges.append(list(np.arange(split_indices[i], length)))

  return split_ranges


def create_examples_hdf_rwc(fn_audio, fn_midi, example_index, hdf, split, hparams):
  samples, sample_rate = librosa.load(fn_audio, sr=hparams['sample_rate'])
  samples = librosa.util.normalize(samples, norm=np.inf)
  wav_data = audio_io.samples_to_wav_data(samples, sample_rate)

  # filter singing notes
  midi = pm.PrettyMIDI(fn_midi)
  midi.instruments = list(
      filter(
          lambda c: 'vocal' in c.name.lower() or
          ('melo' in c.name.lower() and not 'gt melo' in c.name.lower()), midi.instruments))
  for note in midi.instruments[0].notes:
    note.start -= 1
    note.end -= 1
  ns = midi_io.midi_to_note_sequence(midi)

  if len(ns.instrument_infos) <= 0:
    print('No Singing found:', fn_midi)
    return example_index
  elif len(ns.instrument_infos) > 1:
    print('Multiple Singing found:', fn_midi)
    return example_index

  velocities = [note.velocity for note in ns.notes]
  velocity_max = np.max(velocities)
  velocity_min = np.min(velocities)
  new_velocity_tuple = music_pb2.VelocityRange(min=velocity_min, max=velocity_max)

  if split == 'train':
    splits = find_split_points(
        ns,
        samples,
        hparams['sample_rate'],
        hparams['min_length'],
        hparams['max_length'],
    )

    for start, end in zip(splits[:-1], splits[1:]):
      if end - start < hparams['min_length']:
        continue

      new_ns = sequences_lib.extract_subsequence(ns, start, end)
      new_wav_data = audio_io.crop_wav_data(wav_data, hparams['sample_rate'], start, end - start)
      samples_slice, _ = librosa.load(
          fn_audio,
          sr=hparams['sample_rate'],
          offset=start,
          duration=(end - start),
      )
      if np.sum(np.abs(samples_slice)) > 100 and len(new_ns.notes) > 0:
        # append example to hdf
        # https://www.programiz.com/python-programming/methods/built-in/repr
        group = hdf.create_group(str(example_index))
        group.create_dataset('id', data=fn_midi.encode('utf-8'))
        group.create_dataset('sequence', data=repr(new_ns.SerializeToString()))
        group.create_dataset('score', data=''.encode('utf-8'))
        group.create_dataset(
            'audio',
            data=audio_io.wav_data_to_samples(new_wav_data, hparams['sample_rate']),
            compression='gzip',
        )
        group.create_dataset('velocity_range', data=new_velocity_tuple.SerializeToString())
        example_index += 1
  else:
    # append example to hdf
    # https://www.programiz.com/python-programming/methods/built-in/repr
    group = hdf.create_group(str(example_index))
    group.create_dataset('id', data=fn_midi.encode('utf-8'))
    group.create_dataset('sequence', data=repr(ns.SerializeToString()))
    group.create_dataset('score', data=''.encode('utf-8'))
    group.create_dataset(
        'audio',
        data=audio_io.wav_data_to_samples(wav_data, hparams['sample_rate']),
        compression='gzip',
    )
    group.create_dataset('velocity_range', data=new_velocity_tuple.SerializeToString())
    example_index += 1

  return example_index


def create_examples_hdf(fn_audio, fn_midi, example_index, hdf, split, hparams):
  samples, sample_rate = librosa.load(fn_audio, sr=hparams['sample_rate'])
  samples = librosa.util.normalize(samples, norm=np.inf)
  wav_data = audio_io.samples_to_wav_data(samples, sample_rate)

  ns = midi_io.midi_file_to_note_sequence(fn_midi)

  velocities = [note.velocity for note in ns.notes]
  velocity_max = np.max(velocities)
  velocity_min = np.min(velocities)
  new_velocity_tuple = music_pb2.VelocityRange(min=velocity_min, max=velocity_max)

  if split == 'train':
    splits = find_split_points(
        ns,
        samples,
        hparams['sample_rate'],
        hparams['min_length'],
        hparams['max_length'],
    )

    for start, end in zip(splits[:-1], splits[1:]):
      if end - start < hparams['min_length']:
        continue

      new_ns = sequences_lib.extract_subsequence(ns, start, end)
      new_wav_data = audio_io.crop_wav_data(wav_data, hparams['sample_rate'], start, end - start)
      samples_slice, _ = librosa.load(
          fn_audio,
          sr=hparams['sample_rate'],
          offset=start,
          duration=(end - start),
      )
      if np.sum(np.abs(samples_slice)) > 100:
        # append example to hdf
        # https://www.programiz.com/python-programming/methods/built-in/repr
        group = hdf.create_group(str(example_index))
        group.create_dataset('id', data=fn_midi.encode('utf-8'))
        group.create_dataset('sequence', data=repr(new_ns.SerializeToString()))
        group.create_dataset('score', data=''.encode('utf-8'))
        group.create_dataset(
            'audio',
            data=audio_io.wav_data_to_samples(new_wav_data, hparams['sample_rate']),
            compression='gzip',
        )
        group.create_dataset('velocity_range', data=new_velocity_tuple.SerializeToString())
        example_index += 1
  else:
    # append example to hdf
    # https://www.programiz.com/python-programming/methods/built-in/repr
    group = hdf.create_group(str(example_index))
    group.create_dataset('id', data=fn_midi.encode('utf-8'))
    group.create_dataset('sequence', data=repr(ns.SerializeToString()))
    group.create_dataset('score', data=''.encode('utf-8'))
    group.create_dataset(
        'audio',
        data=audio_io.wav_data_to_samples(wav_data, hparams['sample_rate']),
        compression='gzip',
    )
    group.create_dataset('velocity_range', data=new_velocity_tuple.SerializeToString())
    example_index += 1

  return example_index


def find_split_points(note_sequence, samples, sample_rate, min_length, max_length):
  """Returns times at which there are no notes.

    The general strategy employed is to first check if there are places in the
    sustained pianoroll where no notes are active within the max_length window;
    if so the middle of the last gap is chosen as the split point.

    If not, then it checks if there are places in the pianoroll without sustain
    where no notes are active and then finds last zero crossing of the wav file
    and chooses that as the split point.

    If neither of those is true, then it chooses the last zero crossing within
    the max_length window as the split point.

    If there are no zero crossings in the entire window, then it basically gives
    up and advances time forward by max_length.

    Args:
        note_sequence: The NoteSequence to split.
        samples: The audio file as samples.
        sample_rate: The sample rate (samples/second) of the audio file.
        min_length: Minimum number of seconds in a split.
        max_length: Maximum number of seconds in a split.

    Returns:
        A list of split points in seconds from the beginning of the file.
    """

  if not note_sequence.notes:
    return []

  end_time = note_sequence.total_time

  note_sequence_sustain = sequences_lib.apply_sustain_control_changes(note_sequence)

  ranges_nosustain = _find_inactive_ranges(note_sequence)
  ranges_sustain = _find_inactive_ranges(note_sequence_sustain)

  nosustain_starts = [x[0] for x in ranges_nosustain]
  sustain_starts = [x[0] for x in ranges_sustain]

  nosustain_ends = [x[1] for x in ranges_nosustain]
  sustain_ends = [x[1] for x in ranges_sustain]

  split_points = [0.]

  while end_time - split_points[-1] > max_length:
    max_advance = split_points[-1] + max_length

    # check for interval in sustained sequence
    pos = bisect.bisect_right(sustain_ends, max_advance)
    if pos < len(sustain_starts) and max_advance > sustain_starts[pos]:
      split_points.append(max_advance)

    # if no interval, or we didn't fit, try the unmodified sequence
    elif pos == 0 or sustain_starts[pos - 1] <= split_points[-1] + min_length:
      # no splits available, use non sustain notes and find close zero crossing
      pos = bisect.bisect_right(nosustain_ends, max_advance)

      if pos < len(nosustain_starts) and max_advance > nosustain_starts[pos]:
        # we fit, great, try to split at a zero crossing
        zxc_start = nosustain_starts[pos]
        zxc_end = max_advance
        last_zero_xing = _last_zero_crossing(samples, int(np.floor(zxc_start * sample_rate)),
                                             int(np.ceil(zxc_end * sample_rate)))
        if last_zero_xing:
          last_zero_xing = float(last_zero_xing) / sample_rate
          split_points.append(last_zero_xing)
        else:
          # give up and just return where there are at least no notes
          split_points.append(max_advance)

      else:
        # there are no good places to cut, so just pick the last zero crossing
        # check the entire valid range for zero crossings
        start_sample = int(np.ceil((split_points[-1] + min_length) * sample_rate)) + 1
        end_sample = start_sample + \
            (max_length - min_length) * sample_rate
        last_zero_xing = _last_zero_crossing(samples, start_sample, end_sample)

        if last_zero_xing:
          last_zero_xing = float(last_zero_xing) / sample_rate
          split_points.append(last_zero_xing)
        else:
          # give up and advance by max amount
          split_points.append(max_advance)
    else:
      # only advance as far as max_length
      new_time = min(np.mean(ranges_sustain[pos - 1]), max_advance)
      split_points.append(new_time)

  if split_points[-1] != end_time:
    split_points.append(end_time)

  # ensure that we've generated a valid sequence of splits
  for prev, curr in zip(split_points[:-1], split_points[1:]):
    assert curr > prev
    assert curr - prev <= max_length + 1e-8
    if curr < end_time:
      assert curr - prev >= min_length - 1e-8
  assert end_time - split_points[-1] < max_length

  return split_points


def _find_inactive_ranges(note_sequence):
  """Returns ranges where no notes are active in the note_sequence."""
  start_sequence = sorted(note_sequence.notes, key=lambda note: note.start_time, reverse=True)
  end_sequence = sorted(note_sequence.notes, key=lambda note: note.end_time, reverse=True)

  notes_active = 0

  time = start_sequence[-1].start_time
  inactive_ranges = []
  if time > 0:
    inactive_ranges.append(0.)
    inactive_ranges.append(time)
  start_sequence.pop()
  notes_active += 1
  # Iterate through all note on events
  while start_sequence or end_sequence:
    if start_sequence and (start_sequence[-1].start_time < end_sequence[-1].end_time):
      if notes_active == 0:
        time = start_sequence[-1].start_time
        inactive_ranges.append(time)
      notes_active += 1
      start_sequence.pop()
    else:
      notes_active -= 1
      if notes_active == 0:
        time = end_sequence[-1].end_time
        inactive_ranges.append(time)
      end_sequence.pop()

  # if the last note is the same time as the end, don't add it
  # remove the start instead of creating a sequence with 0 length
  if inactive_ranges[-1] < note_sequence.total_time:
    inactive_ranges.append(note_sequence.total_time)
  else:
    inactive_ranges.pop()

  assert len(inactive_ranges) % 2 == 0

  inactive_ranges = [(inactive_ranges[2 * i], inactive_ranges[2 * i + 1])
                     for i in range(len(inactive_ranges) // 2)]
  return inactive_ranges


def _last_zero_crossing(samples, start, end):
  """Returns the last zero crossing in the window [start, end)."""
  samples_greater_than_zero = samples[start:end] > 0
  samples_less_than_zero = samples[start:end] < 0
  samples_greater_than_equal_zero = samples[start:end] >= 0
  samples_less_than_equal_zero = samples[start:end] <= 0

  # use np instead of python for loop for speed
  xings = np.logical_or(
      np.logical_and(samples_greater_than_zero[:-1], samples_less_than_equal_zero[1:]),
      np.logical_and(samples_less_than_zero[:-1], samples_greater_than_equal_zero[1:])).nonzero()[0]

  return xings[-1] + start if xings.size > 0 else None
