# woven_cfrp_wavefield_separation
Code for reproducing wavefield separation and surface microstructure characterization in woven CFRP using laser ultrasonics.
## Overview
This repository contains the PyTorch implementation and datasets for the paper: 
"Decoupling propagation and weave-induced modulation in laser-ultrasonic wavefields enables robust anisotropy characterization of woven CFRP".


## Data
Raw laser-ultrasonic wavefield data are hosted on Zenodo:
DOI: https://doi.org/10.5281/zenodo.18168499

## Quick start
```bash
conda env create -f environment.yml
conda activate cfrp
python scripts/reproduce_figures.py
