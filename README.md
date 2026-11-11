# Template-update policies for behavioural continuous authentication under drift, poisoning and differential privacy

Code, configuration and results for the manuscript *"Who Should Teach the Template? An Evaluation of Update Policies for Behavioral Continuous Authentication Under Drift, Poisoning, and Differential Privacy"* (A. K. Aseres, L. K. Tegegne, A. M. Beyene).

## Contents
| Path | What it is |
|---|---|
| `notebook.ipynb` | The complete study (Sections 1-19: pre-specified study; Section 20: revision experiments). Restart-and-run reproducible. |
| `notebook_executed.ipynb` | The same notebook with all outputs. |
| `pacau_core.py`, `analysis_helpers.py`, `pacau_rev.py` | Simulator, statistics and revision extensions (the notebook also writes these files). |
| `outputs/preregistration.json` | Original analysis plan (not externally registered; see manuscript Section VI-D). |
| `outputs/revision/revision_plan.json` | Revision analysis plan, written before any revision run. |
| `outputs/revision/cmu_partitions_by_seed.json` | Subject roles (background, calibration, attacker/probe, victim) for every seed, incl. pilot seed 999. |
| `outputs/revision/config_methods_conditions.json` | Default configuration, every policy switch and every attack condition. |
| `outputs/**/*.csv` | Per-run results and all summary/test tables reported in the paper. |
| `outputs/revision/artefact_hashes.json` | SHA-256 of every output file and module. |

Attack implementations: `run_stream` (C1-C4) in `pacau_core.py`; `run_stream2` (C3n, C5, C6, C3s, handover) in `pacau_rev.py`.

## Running
```
pip install -r requirements.txt
PACA_ROOT=$(pwd) PACA_JOBS=4 jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook_executed.ipynb --ExecutePreprocessor.timeout=-1
```
The notebook downloads the CMU keystroke benchmark and Touchalytics from commit-pinned public mirrors and verifies SHA-256 hashes. With the cached CSVs in `outputs/` present it only re-runs the analyses (minutes); `PACA_FORCE=1` recomputes every simulation (about 2-3 CPU-hours; parallel with `PACA_JOBS`).

## Data
No data are redistributed here. CMU: Killourhy & Maxion, DSN 2009. Touchalytics: Frank et al., IEEE TIFS 2013. Confirm each dataset's terms with its original authors before reuse.

## Licence
Choose and add a licence before publishing (e.g. MIT for code, CC-BY 4.0 for result files).
