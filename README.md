# CFRP Wavefield Separation


This repository contains the PyTorch implementation and datasets for the paper: "Decoupling propagation and weave-induced modulation in laser-ultrasonic wavefields enables robust anisotropy characterization of woven CFRP".

## Repository layout

```
.
├─ data/                 # Zenodo data pointer 
├─ outputs/              # generated artifacts 
├─ scripts/              # helper scripts
└─ src/                  # core library code
```
## Data
Raw laser-ultrasonic wavefield data are hosted on Zenodo:
DOI: https://doi.org/10.5281/zenodo.18168499

## Setup

Use the provided `environment.yml` or install dependencies from `requirements.txt`.

```bash
conda env create -f environment.yml
```

## Quick start

```bash
python scripts/download_data.py --record <zenodo_record_id>
python scripts/reproduce_figures.py --mode train
```

## Citation

See `CITATION.cff` for citation metadata.
