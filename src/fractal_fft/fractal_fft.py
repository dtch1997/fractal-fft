"""Code for FractalFFT implementation."""
from typing import Any
from typing import Dict

import numpy as np

TWO_PI = 2 * np.pi


def is_integer_matrix(x: Any) -> bool:
    """Check whether a variable is an integer 2D matrix."""
    return isinstance(x, np.ndarray) and len(x.shape) == 2 and x.dtype == np.int32


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize a matrix by subtracting first column from all columns."""
    return x - x[:, 0]  # type: ignore


def is_invertible(x: np.ndarray) -> bool:
    """Check whether a matrix is invertible."""
    try:
        _ = np.linalg.inv(x)
        return True
    except np.linalg.LinAlgError:
        return False


class FractalFFT:
    """Implementation of Fractal FFT algorithm.

    Described in 'A Fast Fourier Transform for Fractal Approximations'
    Reference: https://arxiv.org/pdf/1607.03690.pdf
    """

    def __init__(self, Ainv: np.ndarray, b: np.ndarray, c: np.ndarray):
        """Initializes FractalFFT.

        Args:
            Ainv: d x d matrix. Inverse of matrix A. Must be a 2D integer matrix
            b: d x K matrix, each column is a vector b_k. Must be a 2D integer matrix
            c: d x K matrix, each column is a vector c_j. Must be a 2D integer matrix

        Raises:
            RuntimeError: if arguments are invalid
        """
        # TODO: check Hadamard, invertible
        self._validate(Ainv, b, c)
        self.A = np.linalg.inv(Ainv)
        self.B = Ainv.T
        self.b = normalize(b)
        self.c = normalize(c)
        self.K = b.shape[0]
        self._cache: Dict[Any, np.ndarray] = {}
        if not is_invertible(self.M(1)):
            raise RuntimeError("A, b, c must be such that M(1) is invertible")

    @staticmethod
    def _validate(Ainv: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
        """Check that Ainv, B, C are valid matrices."""
        if not (
            is_integer_matrix(Ainv) and is_integer_matrix(b) and is_integer_matrix(c)
        ):
            raise RuntimeError("Ainv, B, C must be integer matrices")

        if (
            not Ainv.shape[0] == Ainv.shape[1] == b.shape[0] == c.shape[0]
        ) or not b.shape[1] == c.shape[1]:
            raise RuntimeError("Ainv, B, C must be dxd, dxK, dxK respectively")

        if not is_invertible(Ainv):
            raise RuntimeError("Ainv must be invertible")

        A = np.linalg.inv(Ainv)
        _, s, _ = np.linalg.svd(A)
        if not s[0] < 1:
            raise RuntimeError("A must have spectral norm smaller than 1")

    def clear(self) -> None:  # pragma: no cover
        """Clear internal cache memory."""
        self._cache.clear()

    def E(self, N: int, m: int) -> np.ndarray:
        """Calculate matrix E."""
        key = ("E", N, m)
        if key in self._cache:
            return self._cache[key]
        else:
            E = self._E(N, m)
            self._cache[key] = E
            return E

    def _E(self, N: int, m: int) -> np.ndarray:
        e = self.c.T @ (self.A ** N) @ self.b[:, m]
        e = np.stack([e] * self.K, axis=1)
        return np.exp(-TWO_PI * 1j * e)  # type: ignore

    def D(self, N: int, m: int) -> np.ndarray:
        """Calculate matrix D."""
        key = ("D", N, m)
        if key in self._cache:
            return self._cache[key]
        else:
            D = self._D(N, m)
            self._cache[key] = D
            return D

    def _D(self, N: int, m: int) -> np.ndarray:
        if N == 1:
            return np.identity(1)
        elif m == 0:
            # This is true because b_0 is always fixed to be 0
            return np.identity(self.K)
        else:
            return np.kron(self.D(N - 1, m), self.E(N, m))  # type: ignore

    def M(self, N: int) -> np.ndarray:
        """Calculate matrix M."""
        key = ("M", N)
        if key in self._cache:
            return self._cache[key]
        else:
            M = self._M(N)
            self._cache[key] = M
            return M

    def _M(self, N: int) -> np.ndarray:
        if N == 1:
            m = self.c.T @ self.A @ self.b
            return np.exp(TWO_PI * 1j * m)  # type: ignore
        else:
            M_1 = self.M(1)
            M_prev = self.M(N - 1)
            dim_prev = M_prev.shape[0]
            M_N = np.zeros(
                shape=(dim_prev * self.K, dim_prev * self.K), dtype=np.csingle
            )
            # Calculate M_N blockwise
            for col in range(self.K):
                col_start_idx = col * dim_prev
                col_end_idx = (col + 1) * dim_prev
                D_N_col_M_prev = self.D(N, col) @ M_prev
                for row in range(self.K):
                    row_start_idx = row * dim_prev
                    row_end_idx = (row + 1) * dim_prev
                    M_N[row_start_idx:row_end_idx, col_start_idx:col_end_idx] = (
                        M_1[row, col] * D_N_col_M_prev
                    )
            return M_N
