# AlloyGPT-UQ for IN718

Source code for AlloyGPT-UQ: a GPT-style causal language model (AlloyGPT) trained on an "alloy language" for IN718
(Alloy 718). The same model does **forward prediction** (composition → mean / standard deviation / spread of yield
strength) and **inverse design** (yield-strength targets → composition).

The yield strength (YS) labels used for training were computed with the physics-based yield strength model from
[Certification-driven uncertainty-aware alloy design](https://github.com/Taheri-Mousavi-Laboratory/Certification-driven-uncertainty-aware-alloy-design)
(see [Yield strength labels](#yield-strength-labels)).

## About the notebooks and data

> [!IMPORTANT]
> **The notebooks only show how the code works and what its output looks like.** They are not a way to reproduce
> the published results from data in this repository, because no data is provided.

All results in this work come from data computed with **Thermo-Calc**. The Thermo-Calc license does not allow us to
redistribute that data, so this repository contains **no dataset files, no tokenizer trained on the data, and no
trained weights**.

Both notebooks keep the outputs of the original runs (environment check, dataset summaries, training log, example
generations, evaluation plots) so readers can see the expected output. They cannot be re-run without your own data.

**Generating your own data:** anyone with a Thermo-Calc license can compute equivalent data and prepare it in the
format described in [What you need to provide](#what-you-need-to-provide). With that data, every notebook runs as shown.

## Contents

| File | Description |
|---|---|
| [`1_train_AlloyGPT_IN718.ipynb`](./1_train_AlloyGPT_IN718.ipynb) | dataset → sentences → tokens → GPT training (with resume from a checkpoint); includes the original run's outputs |
| [`2_evaluate_AlloyGPT_IN718.ipynb`](./2_evaluate_AlloyGPT_IN718.ipynb) | loss curves, forward prediction, inverse design, and the round-trip check on the test split |
| [`AlloyGPT_UQ/`](./AlloyGPT_UQ) | source code: data processing, GPT model, training loop, testing and utility functions |

## Yield strength labels

The yield strength values behind the training labels (`Mean YS`, `Std YS`, `Delta_YS_expected`) were computed from
Thermo-Calc equilibrium phase data with the physics-based yield strength model in
[Certification-driven uncertainty-aware alloy design](https://github.com/Taheri-Mousavi-Laboratory/Certification-driven-uncertainty-aware-alloy-design)
(`Python_scripts/Yield strength modeling/yield_strength_model.ipynb`). The model itself is not included in this
repository; see that repository for its code and documentation.

## AlloyGPT

Built on the AlloyGPT code by Bo Ni:
B. Ni, B. Glaser, S. M. Taheri-Mousavi, "End-to-end prediction and design of additively manufacturable alloys using a
generative AlloyGPT model", *npj Computational Materials* **11**, 294 (2025). https://doi.org/10.1038/s41524-025-01768-2

The GPT model in `AlloyGPT_UQ/ModelPack_UQ.py` is adapted from [nanoGPT](https://github.com/karpathy/nanoGPT)
(Andrej Karpathy, MIT License).

### What you need to provide

1. **A dataset** (`.arrow`) with one row per alloy and these columns:
   - composition (wt.%): `TiMassPC`, `CrMassPC`, `MoMassPC`, `AlMassPC`, `NbMassPC`, `CoMassPC`, `NiMassPC`
   - properties: `Mean YS`, `Std YS`, `Delta_YS_expected`

   Set its path in `Data_PKeys['local_data_file']` in the training notebook (cell "Parameter Keys").
2. **A BPE tokenizer** for the alloy language at `assets/cBPE_tokenizer_ver1.json` (loaded by
   `DataPack.build_tokenizer`, with `</s>` as token id 0). The loss weighting expects the number characters
   `0-9 . + - e` to be single tokens.

## Installation

Clone the repository using `git`:
```bash
git clone https://github.com/Taheri-Mousavi-Laboratory/AlloyGPT.git
cd AlloyGPT
```

Create a virtual environment
```bash
conda env create --prefix=./AlloyGPT_env --file=AlloyGPT_env_environment.yml
conda activate ./AlloyGPT_env
```

Then, install:
```bash
cd AlloyGPT_UQ
pip install -r requirements.txt
```

Run the notebooks from inside the `AlloyGPT_UQ/` folder, on a CUDA GPU (the training code assumes CUDA):

1. `1_train_AlloyGPT_IN718.ipynb`: set `CKeys['IF_FirstRun'] = 1` for a new run. **A new run deletes `./training_resu/`.**
   To resume, set it to `2` and choose `CKeys['Resume_from_where']`.
2. `2_evaluate_AlloyGPT_IN718.ipynb`: reads `./training_resu/` and writes to `./validation_results/`.

Everything the notebooks write (data splits, dataloaders, checkpoints, predictions) goes to folders that `.gitignore`
excludes, so a `git add .` will not pick them up.

## Data availability

The results in this work are based on data computed with Thermo-Calc. In accordance with the Thermo-Calc software
license, neither the Thermo-Calc data nor the training data derived from it are provided. The notebooks are included
only to show how the code works and its expected output. Researchers with a Thermo-Calc license can generate data in
the format described above and run the full workflow with it.

## Reference

```bibtex
@article{JixuanHasanBoSMohadeseh2026Uncertainty,
      title={Uncertainty-informed alloy design for additive manufacturing via a multi-functional large language model},
      author={Jixuan Dong and Hasan Al Jame and Bo Ni and S. Mohadeseh Taheri-Mousavi},
      journal={Additive Manufacturing Letters},
      year={2026},
      pages={100422},
      url={https://doi.org/10.1016/j.addlet.2026.100422},
}
```
