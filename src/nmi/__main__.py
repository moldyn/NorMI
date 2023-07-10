"""CLI of normalized-mi.

MIT License
Copyright (c) 2023, Daniel Nagel
All rights reserved.

"""
import click
import numpy as np

from nmi import INVMEASURES, NORMS, NormalizedMI, __version__
from nmi._utils import savetxt

# setup matplotlibs rcParam

PRECISION = ['half', 'single', 'double']
PRECISION_TO_DTYPE = {
    'half': np.float16,
    'single': np.float32,
    'double': np.float64,
}

HELP_STR = f"""Normalized MI v{__version__}

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
    'output_basename',
    required=True,
    type=click.Path(),
    help=(
        'Path to output basename. Will be a matrix of shape (n_features, '
        'n_features).'
    ),
)
@click.option(
    '--norm',
    default='joint',
    show_default=True,
    type=click.Choice(NORMS, case_sensitive=False),
    help='Normalization method of the mutual information.',
)
@click.option(
    '--inv-measure',
    default='radius',
    show_default=True,
    type=click.Choice(INVMEASURES, case_sensitive=False),
    help='Invariant measure to rescale the entropies.',
)
@click.option(
    '--n-dims',
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help=(
        'Dimension of each feature. Assumes the first nth colums belong to the'
        ' first feature.'
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
    input_file, output_basename, norm, inv_measure, n_dims, precision, verbose,
):
    # load file
    if verbose:
        click.echo(f'\nNormalized MI\n~~~ Load file: {input_file}')
    input = np.loadtxt(input_file, dtype=PRECISION_TO_DTYPE[precision])

    if verbose:
        click.echo('~~~ Initialize class')
    nmi = NormalizedMI(
        normalize_method=norm,
        invariant_measure=inv_measure,
        verbose=verbose,
        n_dims=n_dims,
    )
    if verbose:
        click.echo('~~~ Fit class')
    nmi.fit(input)

    # save results
    if verbose:
        click.echo(f'~~~ Save files: {output_basename}.nmi/.mi/.hxy/.hx/.hy')
    kwargs = {'fmt': '%.5f'}
    savetxt(f'{output_basename}.nmi', nmi.nmi_, **kwargs)
    savetxt(f'{output_basename}.mi', nmi.mi_, **kwargs)
    savetxt(f'{output_basename}.hx', nmi.hx_, **kwargs)
    savetxt(f'{output_basename}.hy', nmi.hy_, **kwargs)
    savetxt(f'{output_basename}.hxy', nmi.hxy_, **kwargs)


if __name__ == '__main__':
    main()
