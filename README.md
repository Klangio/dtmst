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