# CFRP Wavefield Separation


This repository contains the PyTorch implementation and datasets for the paper: "Decoupling propagation and texture-induced artifacts to improve time-of-flight reliability in scanning laser-ultrasonic measurements of woven CFRP".

## 📂 Repository Structure

```
.
├─ data/                 # Zenodo data pointer and metadata for raw wavefields
├─ outputs/              # generated artifacts (figures, logs, checkpoints)
├─ scripts/              # runnable utilities (data download, training, plotting)
├─ src/                  # core library code (models, datasets, training loops)
├─ environment.yml       # conda environment definition
├─ README.md             # project overview and usage instructions
└─ CITATION.cff          # citation metadata for the paper
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
