<div align="center">
  <img class="darkmode" style="width: 500px;" src="https://github.com/moldyn/normi/blob/main/docs/hero_dark.svg?raw=true#gh-dark-mode-only" />
  <img class="lightmode" style="width: 500px;" src="https://github.com/moldyn/normi/blob/main/docs/hero.svg?raw=true#gh-light-mode-only" />

  <p>
    <a href="https://github.com/wemake-services/wemake-python-styleguide" alt="wemake-python-styleguide">
        <img src="https://img.shields.io/badge/style-wemake-000000.svg" /></a>
    <a href="https://beartype.rtfd.io" alt="bear-ified">
        <img src="https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg" /></a>
    <a href="https://pypi.org/project/normi" alt="PyPI">
        <img src="https://img.shields.io/pypi/v/normi" /></a>
    <a href="https://anaconda.org/conda-forge/normi" alt="conda version">
	<img src="https://img.shields.io/conda/vn/conda-forge/normi" /></a>
    <a href="https://pepy.tech/project/normi" alt="Downloads">
        <img src="https://static.pepy.tech/badge/normi" /></a>
    <a href="https://github.com/moldyn/normi/actions/workflows/pytest.yml" alt="GitHub Workflow Status">
        <img src="https://img.shields.io/github/actions/workflow/status/moldyn/normi/pytest.yml?branch=main"></a>
    <a href="https://codecov.io/gh/moldyn/normi" alt="Code coverage">
        <img src="https://codecov.io/gh/moldyn/normi/branch/main/graph/badge.svg?token=KNWDAUXIGI" /></a>
    <a href="https://github.com/moldyn/normi/actions/workflows/codeql.yml" alt="CodeQL">
        <img src="https://github.com/moldyn/normi/actions/workflows/codeql.yml/badge.svg?branch=main" /></a>
    <a href="https://img.shields.io/pypi/pyversions/normi" alt="PyPI - Python Version">
        <img src="https://img.shields.io/pypi/pyversions/normi" /></a>
    <a href="https://moldyn.github.io/normi" alt="Docs">
        <img src="https://img.shields.io/badge/MkDocs-Documentation-brightgreen" /></a>
    <a href="https://github.com/moldyn/normi/blob/main/LICENSE" alt="License">
        <img src="https://img.shields.io/github/license/moldyn/normi" /></a>
  </p>

  <p>
    <a href="https://moldyn.github.io/NorMI">Docs</a> •
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="https://moldyn.github.io/NorMI/faq">FAQ</a>
  </p>
</div>

# NorMI: Nonparametric Normalized Mutual Information Estimator Based on *k*-NN Statistics
This software provides an extension to the Kraskov-Estimator to allow normalizing the mutual information.

The method will be published soon as:
> **Adaptive Entropy-Based Normalization for (High-Dimensional) Mutual Information**  
> D. Nagel, G. Diez, and G. Stock,  in prep.

If you use this software package, please cite the above mentioned paper.

## Features
- Intuitive usage via [module](#module---inside-a-python-script) and via [CI](#ci---usage-directly-from-the-command-line)
- Sklearn-style API for fast integration into your Python workflow
- No magic, only a  single parameter which can be optimized via cross-validation
- Extensive [documentation](https://moldyn.github.io/NorMI) and detailed discussion in publication

## Installation
The package is called `normi` and is available via [PyPI](https://pypi.org/project/normi) or [conda](https://anaconda.org/conda-forge/normi). To install it, simply call:
```bash
python3 -m pip install --upgrade normi
```
or
```
conda install -c conda-forge normi
```
or for the latest dev version
```bash
# via ssh key
python3 -m pip install git+ssh://git@github.com/moldyn/NorMI.git

# or via password-based login
python3 -m pip install git+https://github.com/moldyn/NorMI.git
```

### Shell Completion
Using the `bash`, `zsh` or `fish` shell click provides an easy way to provide shell completion, checkout the [docs](https://click.palletsprojects.com/en/8.0.x/shell-completion).
In the case of bash you need to add following line to your `~/.bashrc`
```bash
eval "$(_NORMI_COMPLETE=bash_source normi)"
```

## Usage
In general one can call the module directly by its entry point `$ normi` or by calling the module `$ python -m normi`. The latter method is preferred to ensure using the desired python environment. For enabling the shell completion, the entry point needs to be used.

### CI - Usage Directly from the Command Line
The module brings a rich CI using [click](https://click.palletsprojects.com).
For a complete list of all options please see the
[docs](https://moldyn.github.io/NorMI/reference/cli/).
```bash
python -m normi /
  --input input_file  / # ascii file of shape (n_samples, n_features)
  --output output_file  / # creates ascii file of shape (n_features, n_features)
  --n-dims / # this allows to treat every n_dims columns as a high dimensional feature
  --verbose

```

### Module - Inside a Python Script
```python
from normi import NormalizedMI

# Load file
# X is np.ndarray of shape (n_samples, n_features)

nmi = NormalizedMI()
nmi_matrix = nmi.fit_transform(X)
...
```

## Credits

- Logo generated with DALL·E 3 by @gegabo
