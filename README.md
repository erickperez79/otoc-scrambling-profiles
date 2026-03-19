# Spatial Scrambling Profiles and System-Size Scaling of the Scrambling Residual Ω

**Paper 1b — Kaelion Project**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19105623.svg)](https://doi.org/10.5281/zenodo.19105623)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--3228--4847-brightgreen)](https://orcid.org/0009-0006-3228-4847)

## Overview

This repository contains all data, code, and figures for:

> E. F. Perez-Eugenio, "Spatial Scrambling Profiles and System-Size Scaling of the Scrambling Residual Ω across Dynamical Regimes: Exact Simulation and Cross-Platform Quantum Hardware Validation" (2026)

The paper extends the Ω diagnostic introduced in Paper I to multi-distance operator measurements, characterizing spatial scrambling structure across five dynamical models and validating the protocol across three quantum hardware platforms (IBM Heron, Rigetti Ankaa-3, AQT IBEX Q1).

**Paper I (prerequisite):** https://doi.org/10.5281/zenodo.18752608

## Repository Structure

```
├── code/                          # Simulation and analysis scripts
│   ├── P1_KI_chaotic_scaling_v2_dense.py     # KI chaotic N=4-12, dense trace
│   ├── P1_KI_chaotic_scaling_v3_typicality.py # KI chaotic N=14-18, typicality
│   ├── P1b_5models_kaggle_N10_12.py           # 5 models N=10-12
│   └── paper1b_generate_figures.py            # Reproduces all figures from JSON
│
├── data/                          # Authoritative datasets
│   ├── paper1b_master_5models.json            # 5 models N=4-12, D_max=5N
│   ├── paper1b_5models_N4-8_D5N.json         # N=4-8 with per-seed SYK
│   ├── syk_n12_results.json                   # SYK N=12, 10 seeds
│   ├── syk_q6_results.json                    # SYK q=6 N=4-12
│   ├── ki_chaotic_scaling_v3_combined.json    # KI chaotic N=4-18 authoritative
│   ├── aqt_results.json                       # 12 AQT circuits, raw
│   ├── aqt_analysis_final.json                # 48 AQT measurements with diffs
│   ├── rigetti_results.json                   # 26 Rigetti circuits, raw
│   ├── rigetti_analysis_final.json            # Rigetti analysis
│   └── exact_reference_values.json            # Exact simulation reference values
│
├── figures/                       # Publication figures
│   ├── fig1_omega_profiles.png    # Spatial scrambling profiles
│   ├── fig2_crossplatform.png     # Cross-platform hardware comparison
│   ├── fig3_beta_function.png     # Effective beta-function
│   └── fig4_omega_scaling.png     # System-size scaling
│
├── docs/                          # Manuscript files
│   ├── paper1b_main.tex           # Main manuscript (v4)
│   ├── paper1b_supplemental.tex   # Supplemental material
│   └── kaelion_v8.bib             # Bibliography
│
├── README.md
├── LICENSE
├── CITATION.cff
└── .gitignore
```

## Key Results

- **Spatial profiles:** Three qualitatively distinct patterns across dynamical classes — monotonically increasing (KI chaotic, butterfly velocity), U-shaped (KI mixing, novel), flat (SYK, all-to-all)
- **Cross-platform:** Ion trap (AQT IBEX Q1) achieves 15× lower error than superconducting (Rigetti Ankaa-3) for deep chaotic circuits at d=3
- **System-size scaling:** Algebraic form Ω = 1/(1+AN^c) with c≈1 favored over power-law by ΔAIC = −3.30
- **Autonomous flow:** β_eff(Ω) depends only on Ω, not on system size N — first empirical characterization of an autonomous accessibility flow from quantum hardware

## Models and Parameters

All simulations use **Conjunto B** parameters: OBC, J=1.0, h=0.5, b=0.5 (KI chaotic), unless otherwise noted. Five models: KI chaotic, KI mixing (J=0.1), KI integrable (J=0), KI dual-unitary (J=π/4), SYK-inspired (random ZZ all-to-all).

## Reproducing the Results

```bash
# Generate all figures from JSON data
python code/paper1b_generate_figures.py

# Run KI chaotic scaling (N=4-12, dense trace)
python code/P1_KI_chaotic_scaling_v2_dense.py

# Run KI chaotic scaling (N=14-18, typicality)
python code/P1_KI_chaotic_scaling_v3_typicality.py
```

Requirements: Python 3.8+, numpy, scipy, matplotlib, qiskit (for IBM circuits)

## Hardware Access

- **IBM Quantum:** ibm_marrakesh (156 qubits, Heron processor) — via IBM Quantum account
- **Rigetti Ankaa-3** and **AQT IBEX Q1:** via AWS Braket

## Related Work

- Paper I (prerequisite): https://doi.org/10.5281/zenodo.18752608
- OSF project: https://osf.io/6z5sb/
- GitHub: https://github.com/erickperez79

## Author

**E. F. Perez-Eugenio**
Independent Researcher, León, Guanajuato, Mexico
ORCID: [0009-0006-3228-4847](https://orcid.org/0009-0006-3228-4847)
Email: erick.fpe79@gmail.com

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

Please cite this work using the information in [CITATION.cff](CITATION.cff).
