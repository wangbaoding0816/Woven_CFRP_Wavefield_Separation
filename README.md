# CFRP Wavefield Separation

This repository contains code for wavefield separation in CFRP plates using a two-branch B-PINN model with a frequency-domain structural prior.

## Repository layout

```
.
├─ data/                 # Zenodo data pointer (no data stored here)
├─ outputs/              # generated artifacts (gitignored)
├─ scripts/              # helper scripts (download + reproduction)
└─ src/                  # core library code
```

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
