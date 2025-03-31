# TEmPT: **T**emporal **E**sti**m**ation of **P**roportional **T**ransition

This repo is a work in progress and contains all the scripts used for the Servius et al. manuscript "On the estimation of state transitions of proportional data with uncertainty quantification".

The calcualtion of the point estimates for the application examples are in each respective folder as well as the boostrap implementation. An example script for the parallelised bootstrap, will be added in due coarse and can be requested before then.

| foldername      | description                                                                       |
|-----------------|-----------------------------------------------------------------------------------|
| `appli_CSR`     | application scripts for "Antibody Isotype Dynamics in COVID-19 Vaccination".      |
| `appli_cellType`| application scripts for "Evolution of B Cell Maturation in COVID-19 Vaccination". |
| `appli_simData` | application for the simulated closed system and the system with ingress.          |

### Note:

(1) Each folder is self-contained with it's own function file `mm_temptations_functions.py`, these are all the same and were originally symlinks to a single file.

(2) ...
