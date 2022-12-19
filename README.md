# Dual Task Monophonic Singing Transcription (DTMST)

Note tracking for amateur singing transcritpion using neural networks.

## Installation

Create conda environment:
```
conda create -n dtmst python=3.8
conda activate dtmst
```

Install pip packages:
```
pip install -r requirements.txt
```

## Transcribe audio file

```
cd scripts/
python transcribe.py
```

## Evaluate model

```
cd scripts/
python evaluate.py
```

## Train model

```
cd scripts/
python train.py
```

## Attribution

If you use this code or dataset in your research, please cite us via the following BibTeX:

```
@article{schwabe2022dual,
  author={schwabe, markus and murgul, sebastian and heizmann, michael},
  journal={journal of the audio engineering society},
  title={dual task monophonic singing transcription},
  year={2022},
  volume={70},
  number={12},
  pages={1038-1047},
  doi={https://doi.org/10.17743/jaes.2022.0040},
  month={december},
}
```
