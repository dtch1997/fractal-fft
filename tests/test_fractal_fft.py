"""Test suite for FractalFFT."""
import numpy as np

from fractal_fft.fractal_fft import FractalFFT


def test_fractal_fft() -> None:
    """Check that FractalFFT runs with no errors on a toy example."""
    A = 2 * np.identity(2, dtype=np.int32)
    B = np.identity(2, dtype=np.int32)
    C = np.identity(2, dtype=np.int32)
    fractal_fft = FractalFFT(A, B, C)
    _ = fractal_fft.M(2)
