"""CLI of normalized-mi.

MIT License
Copyright (c) 2023, Daniel Nagel
All rights reserved.

"""
import click
import numpy as np

from nmi import NormalizedMI

# setup matplotlibs rcParam

PRECISION = ['half', 'single', 'double']
PRECISION_TO_DTYPE = {
    'half': np.float16,
    'single': np.float32,
    'double': np.float64,
}

HELP_STR = f"""Normalized MI v{nmi.__version__}

\b
Estimating the normalized mutual information based on k-nn statistics.
Copyright (c) 2023, Daniel Nagel
"""


@click.command(
    help='Estimating NMI matrix of coordinates.',
    no_args_is_help=True,
)
@click.option(
    '-i',
    '--input',
    'input_file',
    required=True,
    type=click.Path(exists=True),
    help=(
        'Path to input file. Needs to be of shape (n_samples, n_features).'
        ' All command lines need to start with "#". By default np.float16'
        ' is used for the datatype.'
    ),
)
@click.option(
    '-o',
    '--output',
    'output_file',
    required=True,
    type=click.Path(),
    help=(
        'Path to output file. Will be a matrix of shape (n_features, '
        'n_features).'
    ),
)
@click.option(
    '--precision',
    default='single',
    show_default=True,
    type=click.Choice(PRECISION, case_sensitive=True),
    help=(
        'Precision used for calculation. Lower precision reduces memory '
        'impact but may lead to overflow errors.'
    ),
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Activate verbose mode.',
)
def main(
    input_file,
    output_file,
    precision,
    verbose,
):
    if verbose:
        click.echo('\nNormalized MI\n~~~ Initialize class')

    raise NotImplementedError('sorry nothing implemented yet')


if __name__ == '__main__':
    main()
