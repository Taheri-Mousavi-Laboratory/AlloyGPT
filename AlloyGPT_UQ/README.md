# AlloyGPT-UQ for IN718

Source code for the AlloyGPT-UQ workflow on IN718 (Alloy 718):

1. a **physics-based yield strength model** that computes yield strength (YS) from CALPHAD equilibrium phase data and
   supplies the YS labels, and
2. a **GPT-style causal language model** (AlloyGPT) trained on an "alloy language" for IN718. The same model does
   **forward prediction** (composition → mean / standard deviation / spread of yield strength) and **inverse design**
   (yield-strength targets → composition).

## No data or weights

The phase-equilibrium data and the training data were generated with Thermo-Calc, and its license does not allow
redistribution. This repository therefore contains **no dataset files, no tokenizer trained on the data, and no trained
weights**. It shows how the code works. If you generate comparable data yourself, you can run every step with it.

The yield strength notebook runs end to end on random placeholder data by default (the numbers it produces have no
physical meaning). The training notebook is kept with the outputs of the original run (environment check, dataset
summaries, training log and the example generations printed during training) so readers can see what a run looks like.

## Contents

| File | Description |
|---|---|
| [`0_yield_strength_model.ipynb`](./0_yield_strength_model.ipynb) | yield strength model as one function, `compute_yield_strength(phase_data, params)`; data loading, plots, optional export of the composition/YS table |
| [`1_train_AlloyGPT_IN718.ipynb`](./1_train_AlloyGPT_IN718.ipynb) | dataset → sentences → tokens → GPT training (with resume from a checkpoint); includes the original run's outputs |
| [`2_evaluate_AlloyGPT_IN718.ipynb`](./2_evaluate_AlloyGPT_IN718.ipynb) | loss curves, forward prediction, inverse design, and the round-trip check on the test split |
| [`AlloyGPT/UtilityPack.py`](./AlloyGPT/UtilityPack.py) | folder and logging helpers |
| [`AlloyGPT/DataPack_v5.py`](./AlloyGPT/DataPack_v5.py) | train/valid/test split, record → sentence templates, tokenizer loading and tokenization |
| [`AlloyGPT/ModelPack_v4.py`](./AlloyGPT/ModelPack_v4.py) | GPT model and generation |
| [`AlloyGPT/TrainPack_v3.py`](./AlloyGPT/TrainPack_v3.py) | training loop (gradient accumulation, warm-up + cosine LR, number-token-weighted loss, checkpoints) |
| [`AlloyGPT/TestPack_v3.py`](./AlloyGPT/TestPack_v3.py) | sentence parsing and test-set helpers (imported by the training notebook) |

## Yield strength model

The yield strength model is the same model used in
[Certification-driven uncertainty-aware alloy design](https://github.com/Taheri-Mousavi-Laboratory/Certification-driven-uncertainty-aware-alloy-design)
(`Python_scripts/Yield strength modeling/yield_strength_model.ipynb`). In this repository the whole model is wrapped in
a single function:

```python
results = compute_yield_strength(phase_data)                          # default parameters
results = compute_yield_strength(phase_data, MaterialParams(r_gp=15e-9))  # override any parameter
```

`results` holds the bulk composition, the γ′ and γ″ volume fractions, each strengthening contribution and the total
`YS` (MPa) for every configuration.

The yield strength is the sum of four contributions:

$$\sigma_y = \sigma_{\gamma'} + \sigma_{\gamma''} + \sigma_{ss} + \sigma_0$$

| Term | Mechanism |
|---|---|
| σ<sub>γ′</sub> | Coherency strengthening by spherical γ′, weakly coupled dislocation pairs |
| σ<sub>γ″</sub> | Coherency strengthening by disc-shaped γ″, strongly coupled dislocation pairs |
| σ<sub>ss</sub> | Solid-solution strengthening of the γ matrix (root-sum-square of element contributions) |
| σ<sub>0</sub> | Constant base term |

Steps for each configuration:

1. **Lattice parameter of each phase** from a Vegard-type mixing rule over its composition, with Ni as the balance element.
2. **Volume fractions** of γ′ and γ″, converted from CALPHAD phase amounts using molar volumes (4 atoms per cell).
3. **Interparticle spacing** from the volume fraction and the mean planar precipitate radius, r̄ = √(2/3)·r. For γ″, r is
   the equivalent radius (a²c)<sup>1/3</sup> of the disc.
4. **Critical resolved shear stress** for edge and screw dislocations, taking the larger of the two and converting to YS
   with the Taylor factor M.

The equations are written out in the notebook's Model section.

### Default parameters

| Parameter | Symbol | Value |
|---|---|---|
| Shear modulus | G | 75.2 GPa |
| Burgers vector | b | 0.254 nm |
| Poisson ratio | ν | 0.33 |
| Taylor factor | M | 2.6 |
| γ′ radius | r | 10 nm |
| γ′ lattice misfit | δ | 0.004 |
| γ″ disc radius / half-thickness | a / c | 50 nm / 10 nm |
| γ″ tetragonal misfit | δ″ | 0.0286 |
| Base strength | σ<sub>0</sub> | 22.34 MPa |

All parameters, including the lattice-parameter and solid-solution coefficients, are fields of `MaterialParams` in the
notebook.

### Input format

To run on your own Thermo-Calc data, export the equilibrium results as HDF5 files `batchconfig0.hdf5`,
`batchconfig1.hdf5`, … and set `USE_SYNTHETIC_DATA = False`, `DATA_DIR` and `N_FILES` in the notebook's configuration
cell. Each file contains the datasets below, one row per alloy configuration; rows from all files are concatenated.

| Key | Shape | Content |
|---|---|---|
| `comp` | (n, 7) | Bulk alloy composition, columns `Ni, Cr, Nb, Mo, Ti, Al, Co` (Fe is the balance) |
| `FCC_1`, `FCC_1_comp` | (n,), (n, 7) | γ matrix: phase amount, phase composition |
| `FCC_2`, `FCC_2_comp` | (n,), (n, 7) | Candidate γ′ phase |
| `FCC_3`, `FCC_3_comp` | (n,), (n, 7) | Candidate γ′ phase |
| `BCT`, `BCT_comp` | (n,), (n, 7) | γ″ phase |

Phase compositions (`*_comp`) use the column order `Fe, Cr, Nb, Mo, Ti, Al, Co`, with Ni taken as the balance. For
each configuration, γ′ is whichever of `FCC_2` and `FCC_3` has the larger phase amount.

Optional: set `OUTPUT_PATH` (`.csv` or `.arrow`) to save the composition/YS table, and provide
`data/literature_ys.csv` (one column, `YS_MPa`) to plot a comparison against reported values.

## AlloyGPT

Built on the AlloyGPT code by Bo Ni:
B. Ni, B. Glaser, S. M. Taheri-Mousavi, "End-to-end prediction and design of additively manufacturable alloys using a
generative AlloyGPT model", *npj Computational Materials* **11**, 294 (2025). https://doi.org/10.1038/s41524-025-01768-2

The GPT model in `AlloyGPT/ModelPack_v4.py` is adapted from [nanoGPT](https://github.com/karpathy/nanoGPT)
(Andrej Karpathy, MIT License).

### What you need to provide

1. **A dataset** (`.arrow`) with one row per alloy and these columns:
   - composition (wt.%): `TiMassPC`, `CrMassPC`, `MoMassPC`, `AlMassPC`, `NbMassPC`, `CoMassPC`, `NiMassPC`
   - properties: `Mean YS`, `Std YS`, `Delta_YS_expected`

   Set its path in `Data_PKeys['local_data_file']` in the training notebook (cell "Parameter Keys").
2. **A BPE tokenizer** for the alloy language at `assets/cBPE_tokenizer_ver1.json` (loaded by
   `DataPack.build_tokenizer`, with `</s>` as token id 0). The loss weighting expects the number characters
   `0-9 . + - e` to be single tokens.

## Running

```
pip install -r requirements.txt
```

Run the notebooks from the repository root, in order:

0. `0_yield_strength_model.ipynb`: runs on CPU. With the default `USE_SYNTHETIC_DATA = True` it writes random HDF5
   files to `synthetic_data/` and runs the full model on them.
1. `1_train_AlloyGPT_IN718.ipynb` (CUDA GPU; the training code assumes CUDA): set `CKeys['IF_FirstRun'] = 1` for a new
   run. **A new run deletes `./training_resu/`.** To resume, set it to `2` and choose `CKeys['Resume_from_where']`.
   The model is imported as `AlloyGPT.ModelPack_v3`; the version in this repository is `AlloyGPT/ModelPack_v4.py`
   (same `GPT` model, `generate` can stop at `</s>`), so change that import (and `DataPack_v3` → `DataPack_v5` in the
   tokenizer check) when you run it.
2. `2_evaluate_AlloyGPT_IN718.ipynb`: reads `./training_resu/` and writes to `./validation_results/`.

Everything the notebooks write (synthetic data, data splits, dataloaders, checkpoints, predictions) goes to folders that
`.gitignore` excludes, so a `git add .` will not pick them up.

## Data availability

The Thermo-Calc phase-equilibrium data and the derived training data are not included, in accordance with the
Thermo-Calc software license. Researchers with a Thermo-Calc license can regenerate equivalent inputs in the formats
described above.

## Reference

```bibtex
@paper{JixuanHasanBoSMohadeseh2026Uncertainty,
      title={Uncertainty-informed alloy design for additive manufacturing via a multi-functional large language model}, 
      author={Jixuan Dong, Hasan Al Jame, Bo Ni and S. Mohadeseh Taheri-Mousavi},
      journal={Additive Manufacturing Letters},
      year={2026},
      pages={100422},
      url={https://doi.org/10.1016/j.addlet.2026.100422}, 
}
