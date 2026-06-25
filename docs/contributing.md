# Welcome to the `normi` Contributing Guide

This guide will give you an overview of the contribution workflow from opening an issue and creating a PR. To get an overview of the project, read the [module overview][normi].

## Issues

### Create a new issue

If you spot a bug, want to request a new functionality, or have a question on how to use the module, please [search if an issue already exists](https://github.com/moldyn/normi/issues). If a related issue does not exist, feel free to [open a new issue](https://github.com/moldyn/normi/issues/new/choose).

### Solve an issue

If you want to contribute and do not how, feel free to scan through the [existing issues](https://github.com/moldyn/normi/issues).

## Create a new pull request
### Create a fork

If you want to request a change, you first have to [fork the repository](https://github.com/moldyn/normi/fork).

### Setup a development environment

First you need to install `uv`. Please follow the installation instruction provided on their page [docs.astral.sh/uv](https://docs.astral.sh/uv/).

```bash
# create the virtual environment and install all dependency groups
uv sync --all-groups

# install the pre-commit hooks (runs ruff on every commit)
uv run pre-commit install
```

With the hooks installed, `ruff` lints and formats your staged files automatically on each `git commit`. To run them manually on the whole project at any time:
```bash
uv run pre-commit run --all-files
```

### Make changes and run tests

Apply your changes and check if you followed the coding style by running
```bash
uv run ruff check
uv run ruff format --check
```
Most issues can be fixed automatically with `uv run ruff check --fix` and `uv run ruff format`.

If you add a new function/method/class please ensure that you add a test function, as well. Running the test simply by
```bash
uv run pytest
```
Ensure that the coverage does not decrease.

### Open a pull request

Now you are ready to open a pull request and please do not forget to add a description.
