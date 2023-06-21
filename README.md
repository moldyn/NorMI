<div align="center">
  <!--
  <img class="darkmode" style="width: 400px;" src="https://github.com/moldyn/normalized-mi/blob/main/docs/logo_large_dark.svg?raw=true#gh-dark-mode-only" />
  <img class="lightmode" style="width: 400px;" src="https://github.com/moldyn/normalized-mi/blob/main/docs/logo_large_light.svg?raw=true#gh-light-mode-only" />
  -->

  <p>
    <a href="https://github.com/wemake-services/wemake-python-styleguide" alt="wemake-python-styleguide">
        <img src="https://img.shields.io/badge/style-wemake-000000.svg" /></a>
    <a href="https://beartype.rtfd.io" alt="bear-ified">
        <img src="https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg" /></a>
    <a href="https://pypi.org/project/normalized-mi" alt="PyPI">
        <img src="https://img.shields.io/pypi/v/normalized-mi" /></a>
    <a href="https://anaconda.org/conda-forge/normalized-mi" alt="conda version">
	<img src="https://img.shields.io/conda/vn/conda-forge/normalized-mi" /></a>
    <a href="https://pepy.tech/project/normalized-mi" alt="Downloads">
        <img src="https://pepy.tech/badge/normalized-mi" /></a>
    <a href="https://github.com/moldyn/normalized-mi/actions/workflows/pytest.yml" alt="GitHub Workflow Status">
        <img src="https://img.shields.io/github/actions/workflow/status/moldyn/normalized-mi/pytest.yml?branch=main"></a>
    <a href="https://codecov.io/gh/moldyn/normalized-mi" alt="Code coverage">
        <img src="https://codecov.io/gh/moldyn/normalized-mi/branch/main/graph/badge.svg?token=KNWDAUXIGI" /></a>
    <a href="https://github.com/moldyn/normalized-mi/actions/workflows/codeql.yml" alt="CodeQL">
        <img src="https://github.com/moldyn/normalized-mi/actions/workflows/codeql.yml/badge.svg?branch=main" /></a>
    <a href="https://img.shields.io/pypi/pyversions/normalized-mi" alt="PyPI - Python Version">
        <img src="https://img.shields.io/pypi/pyversions/normalized-mi" /></a>
    <a href="https://moldyn.github.io/normalized-mi" alt="Docs">
        <img src="https://img.shields.io/badge/MkDocs-Documentation-brightgreen" /></a>
    <a href="https://github.com/moldyn/normalized-mi/blob/main/LICENSE" alt="License">
        <img src="https://img.shields.io/github/license/moldyn/normalized-mi" /></a>
  </p>

  <p>
    <a href="https://moldyn.github.io/normalized-mi">Docs</a> •
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="https://moldyn.github.io/normalized-mi/faq">FAQ</a>
  </p>
</div>

# Nonparametric Normalized Mutual Information Estimator Based on $k$-NN Statistics
This software provides an extension to the Kraskov-Estimator to allow normalizing the mutual information.

The method will be published soon:
> **Nonparametric Normalized Mutual Information Estimator to Identify Functional Dynamics in Proteins**  
> D. Nagel, G. Diez, and G. Stock,  

If you use this software package, please cite the above mentioned paper.

## Features
- Intuitive usage via [module](#module---inside-a-python-script) and via [CI](#ci---usage-directly-from-the-command-line)
- Sklearn-style API for fast integration into your Python workflow
- No magic, only a  single parameter which can be optimized via cross-validation
- Extensive [documentation](https://moldyn.github.io/normalized-mi) and detailed discussion in publication

## Installation
The package is called `normalized-mi` and is available via [PyPI](https://pypi.org/project/normalized-mi) or [conda](https://anaconda.org/conda-forge/normalized-mi). To install it, simply call:
```bash
python3 -m pip install --upgrade normalized-mi
```
or
```
conda install -c conda-forge normalized-mi
```

or for the latest dev version
```bash
# via ssh key
python3 -m pip install git+ssh://git@github.com/moldyn/normalized-mi.git

# or via password-based login
python3 -m pip install git+https://github.com/moldyn/normalized-mi.git
```

### Shell Completion
Using the `bash`, `zsh` or `fish` shell click provides an easy way to provide shell completion, checkout the [docs](https://click.palletsprojects.com/en/8.0.x/shell-completion).
In the case of bash you need to add following line to your `~/.bashrc`
```bash
eval "$(_NORMALIZED_MI_COMPLETE=bash_source normalized-mi)"
```

## Usage
In general one can call the module directly by its entry point `$ normalized-mi` or by calling the module `$ python -m normalized-mi`. The latter method is preferred to ensure using the desired python environment. For enabling the shell completion, the entry point needs to be used.

### CI - Usage Directly from the Command Line
The module brings a rich CI using [click](https://click.palletsprojects.com).
Each module and submodule contains a detailed help, which can be accessed by
...

tba

### Module - Inside a Python Script
```python
from normalized-mi import NormalizedMI

# Load file
# X is np.ndarray of shape (n_samples, n_features)

nmi = NormalizedMI()
nmi.fit(X)
...
```
