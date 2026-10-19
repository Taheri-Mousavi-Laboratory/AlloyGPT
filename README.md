# AlloyGPT: End-to-end prediction and design of additively manufacturable alloys using an autoregressive language model

Bo Ni<sup>1</sup>, Benjamin Glaser<sup>1</sup>, S. Mohadeseh Taheri-Mousavi<sup>1,2,3</sup>

<sup>1</sup> Department of Materials Science and Engineering, Carnegie Mellon University

<sup>2</sup> Department of Mechanical Engineering, Carnegie Mellon University

<sup>3</sup> Lead contact: smtaherimousavi@cmu.edu

Rapid progress in additive manufacturing of alloys opens opportunities in controlling compositions and microstructures at voxel-size resolution in complex geometries, thus unlocking unprecedented design and performance in various critical engineering applications. However, to fully exploit such potential, capable yet efficient models for navigating the vast design spaces of alloy compositions, structures and properties are of great research interest. Here, we present AlloyGPT, an autoregressive alloy-specific language model, that learns the composition-structure-property relationship and generates novel designs for additively manufacturable alloys. Specifically, we develop efficient grammar to convert physics-rich alloy datasets into readable text records for both forward prediction and inverse design tasks. Then, we construct a customized tokenizer and generative pre-trained transformer (GPT) model to master this alloy-specific language through autoregressive training. At deployment, our model can accurately predict multiple phase structures and properties based on given alloy compositions, achieving R2 values ranging from 0.86 to 0.99 for the test set. When tested beyond the learned composition domain, this performance only degrades gradually in a stable manner. Given the desired properties and structures, the same model can suggest multiple alloy compositions that meet the design goals. And the balance between composition diversity and design accuracy can be further tuned stably. Our AlloyGPT model presents a novel way of integrating comprehensive knowledge of alloys in terms of language and can simultaneously solve forward prediction and inverse design tasks with accuracy, diversity and robustness. This fundamental language model will open new avenues to accelerate knowledge integration and material design for pure or gradient structural alloys manufactured by traditional and additive manufacturing. 

![plot](./assets/TOC.svg)

## Update: AlloyGPT-UQ for IN718

The [`AlloyGPT_UQ/`](./AlloyGPT_UQ) folder extends AlloyGPT to **uncertainty-aware prediction and design of IN718
(Alloy 718)**. The model is trained to predict the **mean, standard deviation and spread of yield strength** for a
given composition (forward prediction), and to generate compositions that meet targets on these quantities (inverse
design). The folder contains:

- `0_yield_strength_model.ipynb`: the physics-based yield strength model that supplies the YS labels (the same model
  used in [Certification-driven uncertainty-aware alloy design](https://github.com/Taheri-Mousavi-Laboratory/Certification-driven-uncertainty-aware-alloy-design))
- `1_train_AlloyGPT_IN718.ipynb`: training on the IN718 alloy language
- `2_evaluate_AlloyGPT_IN718.ipynb`: forward prediction, inverse design and the round-trip check
- `AlloyGPT/`: the source code version used for this work

> [!NOTE]
> The AlloyGPT-UQ notebooks only show how the code works and its expected output. The data come from Thermo-Calc
> and **are not provided** under its license. Users with a Thermo-Calc license can generate data in the same format
> and run the full workflow. See [`AlloyGPT_UQ/README.md`](./AlloyGPT_UQ/README.md) for details and setup.

## Installation

Create a virtual environment

```bash
conda env create --prefix=./AlloyGPT_env --file=AlloyGPT_env_environment.yml
conda activate ./AlloyGPT_env

```

Install:
```bash
pip install git+https://github.com/Taheri-Mousavi-Laboratory/AlloyGPT.git

```
If you want to create an editable installation, clone the repository using `git`:
```bash
git clone https://github.com/Taheri-Mousavi-Laboratory/AlloyGPT.git
cd AlloyGPT
```
Then, install:
```bash
pip install -r requirements.txt
pip install -e .
```
### Directory structure
```
ModeShapeDiffusionDesign/
│
├── VibeGen/                                    # Source code directory
│   ├── DataPack.py
│   ├── ModelPack.py
│   ├── TrainPack.py
│   ├── UtilityPack.py
│   └── TestPack.py
│
├── AlloyGPT_UQ/                                # AlloyGPT-UQ extension for IN718 (see its README)
│   ├── AlloyGPT/                               # source code version used for AlloyGPT-UQ
│   ├── 0_yield_strength_model.ipynb
│   ├── 1_train_AlloyGPT_IN718.ipynb
│   ├── 2_evaluate_AlloyGPT_IN718.ipynb
│   ├── README.md
│   └── requirements.txt
|
├── 0_AlloyGPT_Inferring_demo.ipynb             # demo 1: for making inferences
├── 1_AlloyGPT_training_demo.ipynb              # demo 2: for training
│
├── setup.py                                    # The setup file for packaging
├── requirements.txt                            # List of dependencies
├── README.md                                   # Documentation
├── assets/                                     # Support materials
└── ...
```

## Usage

### Inference notebook

In the following example, we will demonstrate how to solve the forward prediction and inverse design tasks with the same trained AlloyGPT model. Note that, you will need the password to access the trained model due to restrictions of [Thermo-Calc EULA](https://thermocalc.com/wp-content/uploads/Documentation/Terms_Conditions/eula.pdf).

```
0_AlloyGPT_Inferring_demo.ipynb
```

### Training notebook

In the following example, we will demonstrate how to train the AlloyGPT model. Note that, you will need the password to access the dataset due to restrictions of [Thermo-Calc EULA](https://thermocalc.com/wp-content/uploads/Documentation/Terms_Conditions/eula.pdf).

```
1_AlloyGPT_training_demo.ipynb 
```

### Pretrained models
Due to the restrictions of [Thermo-Calc EULA](https://thermocalc.com/wp-content/uploads/Documentation/Terms_Conditions/eula.pdf), the trained model will only be made available based on reasonable request. User tokens will be asked for to access the trained model when running the notebooks listed above.

### Reference

```bibtex
@paper{BoBenjaminSMohadeseh2025AlloyGPT,
      title={ AlloyGPT: End-to-end prediction and design of additively manufacturable alloys using an autoregressive language model}, 
      author={Bo Ni, Benjamin Glaser and S. Mohadeseh Taheri-Mousavi},
      year={2025},
      eprint={preprint},
      archivePrefix={Research Square},
      url={https://doi.org/10.21203/rs.3.rs-6067058/v1}, 
}
@paper{JixuanHasanBoSMohadeseh2026Uncertainty,
      title={Uncertainty-informed alloy design for additive manufacturing via a multi-functional large language model}, 
      author={Jixuan Dong, Hasan Al Jame, Bo Ni and S. Mohadeseh Taheri-Mousavi},
      journal={Additive Manufacturing Letters},
      year={2026},
      pages={100422},
      url={https://doi.org/10.1016/j.addlet.2026.100422}, 
}
```

Our implementation is inspired by the [nanoGPT](https://github.com/karpathy/nanoGPT) repository by [Andrej Karpathy](https://github.com/karpathy).
