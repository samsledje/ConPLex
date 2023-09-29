# ConPLex

![ConPLex Schematic](assets/images/Fig2_Schematic.png)

[![ConPLex Releases](https://img.shields.io/github/v/release/samsledje/ConPLex?include_prereleases)](https://github.com/samsledje/ConPLex/releases)
[![PyPI](https://img.shields.io/pypi/v/conplex-dti)](https://pypi.org/project/conplex-dti/)
[![Build](https://github.com/samsledje/ConPLex/actions/workflows/build.yml/badge.svg)](https://github.com/samsledje/ConPLex/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/conplex/badge/?version=latest)](https://conplex.readthedocs.io/en/main/?badge=main)
[![License](https://img.shields.io/github/license/samsledje/ConPLex)](https://github.com/samsledje/ConPLex/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

🚧🚧 Please note that ConPLex is currently a pre-release and is actively being developed. For the code used to generate our PNAS results, see the [manuscript code](https://github.com/samsledje/ConPLex_dev) 🚧🚧

 - [Homepage](http://conplex.csail.mit.edu)
 - [Documentation](https://d-script.readthedocs.io/en/main/)

## Abstract

Sequence-based prediction of drug-target interactions has the potential to accelerate drug discovery by complementing experimental screens. Such computational prediction needs to be generalizable and scalable while remaining sensitive to subtle variations in the inputs. However, current computational techniques fail to simultaneously meet these goals, often sacrificing performance on one to achieve the others. We develop a deep learning model, ConPLex, successfully leveraging the advances in pre-trained protein language models ("PLex") and employing  a novel  protein-anchored contrastive co-embedding ("Con") to outperform state-of-the-art approaches. ConPLex achieves high accuracy, broad adaptivity to unseen data, and specificity against decoy compounds. It makes predictions of binding based on the distance between learned representations, enabling predictions at the scale of massive compound libraries and the human proteome. Experimental testing of 19 kinase-drug interaction predictions validated 12 interactions, including four with sub-nanomolar affinity, plus a novel strongly-binding EPHB1 inhibitor ($K_D = 1.3nM$). Furthermore, ConPLex embeddings are interpretable, which enables us to visualize the drug-target embedding space and use embeddings to characterize the function of human cell-surface proteins. We anticipate ConPLex will facilitate novel drug discovery by making highly sensitive in-silico drug screening feasible at genome scale.

## Installation

### Install from PyPI

You should first have a version of [`cudatoolkit`](https://anaconda.org/nvidia/cudatoolkit) compatible with your system installed. Then run

```bash
pip install conplex-dti
conplex-dti --help
```

### Compile from Source

```bash
git clone https://github.com/samsledje/ConPLex.git
cd ConPLex
conda create -n conplex-dti python=3.9
conda activate conplex-dti
make poetry-download
export PATH="[poetry  install  location]:$PATH"
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
make install
conplex-dti --help
```

## Usage

### Download benchmark data sets and pre-trained models

```bash
conplex-dti download --to datasets --benchmarks davis bindingdb biosnap biosnap_prot biosnap_mol dude
```

```bash
conplex-dti download --to . --models ConPLex_v1_BindingDB
```

### Run benchmark training

```bash
conplex-dti train --run-id TestRun --config config/default_config.yaml
```

### Make predictions with a trained model

```bash
conplex-dti predict --data-file [pair predict file].tsv --model-path ./models/ConPLex_v1_BindingDB.pt --outfile ./results.tsv
```

Format of `[pair predict file].tsv` should be `[protein ID]\t[molecule ID]\t[protein Sequence]\t[molecule SMILES]`

### Visualize co-embedding space

```bash
...
```

## Reference

If you use ConPLex, please cite [Contrastive learning in protein language space predicts interactions between drugs and protein targets](https://www.pnas.org/doi/10.1073/pnas.2220778120) by Rohit Singh*, Samuel Sledzieski*, Bryan Bryson, Lenore Cowen and Bonnie Berger.

```bash
@article{singh2023contrastive,
  title={Contrastive learning in protein language space predicts interactions between drugs and protein targets},
  author={Singh, Rohit and Sledzieski, Samuel and Bryson, Bryan and Cowen, Lenore and Berger, Bonnie},
  journal={Proceedings of the National Academy of Sciences},
  volume={120},
  number={24},
  pages={e2220778120},
  year={2023},
  publisher={National Acad Sciences}
}
```

Thanks to Ava Amini, Kevin Yang, and Sevahn Vorperian from MSR New England for suggesting the use of the triplet distance contrastive loss function without the sigmoid activation. The default has now been changed. For the original formulation with the sigmoid activation, you can set the `--use-sigmoid-cosine` flag during training.

### Manuscript Code

Code used to generate results in the manuscript can be found in the [development repository](https://github.com/samsledje/ConPLex_dev)
