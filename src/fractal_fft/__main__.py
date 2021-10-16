"""Command-line interface."""
import click
import numpy as np

from fractal_fft.fractal_fft import FractalFFT


@click.command()
@click.version_option()
def main() -> None:
    """Fractal Fft."""
    A = 2 * np.identity(2, dtype=np.int32)
    B = np.identity(2, dtype=np.int32)
    C = np.identity(2, dtype=np.int32)
    fractal_fft = FractalFFT(A, B, C)
    print(fractal_fft.E(2, 1).shape)
    print(fractal_fft.D(2, 1).shape)
    print(fractal_fft.M(2))


if __name__ == "__main__":
    main(prog_name="fractal-fft")  # pragma: no cover
