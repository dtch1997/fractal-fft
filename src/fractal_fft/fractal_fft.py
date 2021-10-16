"""Code for FractalFFT implementation."""
import numpy as np

TWO_PI = 2 * np.pi


class FractalFFT:
    """Implementation of Fractal FFT algorithm.

    Described in 'A Fast Fourier Transform for Fractal Approximations'
    Reference: https://arxiv.org/pdf/1607.03690.pdf
    """

    def __init__(self, Ainv: np.ndarray, b: np.ndarray, c: np.ndarray):
        """Initializes FractalFFT.

        Args:
            Ainv: d x d matrix
            b: K x d matrix, each column is a vector b_k
            c: K x d matrix, each column is a vector c_j
        """
        # TODO: check Hadamard, invertible
        self._validate(Ainv, b, c)
        self.A = np.linalg.inv(Ainv)
        self.b = b
        self.c = c
        self._normalize()
        self.K = b.shape[0]
        self._cache = {}

    @staticmethod
    def _validate(Ainv, b, c):
        """Check that Ainv, B, C are valid matrices."""

        def is_integer_matrix(x):
            return (
                isinstance(x, np.ndarray) and len(x.shape) == 2 and x.dtype == np.int32
            )

        if (
            not is_integer_matrix(Ainv)
            and is_integer_matrix(b)
            and is_integer_matrix(c)
        ):
            raise RuntimeError("Ainv, B, C must be integer matrices")

        if not Ainv.shape[0] == Ainv.shape[1] == b.shape[1] == c.shape[1]:
            raise RuntimeError("Ainv, B, C must be dxd, Kxd, Kxd respectively")

        try:
            A = np.linalg.inv(Ainv)
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Ainv must be invertible") from e

        _, s, _ = np.linalg.svd(A)
        if not s[0] < 1:
            raise RuntimeError("A must have spectral norm smaller than 1")

    def _normalize(self):
        """Normalize to ensure b_0, c_0 are 0."""
        self.b = self.b - self.b[:, 0]
        self.c = self.c - self.c[:, 0]

    def clear(self):
        """Clear internal cache memory."""
        self._cache.clear()

    def E(self, N, m):
        """Calculate matrix E."""
        key = ("E", N, m)
        if key in self._cache:
            return self._cache[key]
        else:
            E = self._E(N, m)
            self._cache[key] = E
            return E

    def _E(self, N, m):
        e = self.c.T @ (self.A ** N) @ self.b[:, m]
        e = np.stack([e] * self.K, axis=1)
        return np.exp(-TWO_PI * 1j * e)

    def D(self, N, m):
        """Calculate matrix D."""
        key = ("D", N, m)
        if key in self._cache:
            return self._cache[key]
        else:
            D = self._D(N, m)
            self._cache[key] = D
            return D

    def _D(self, N, m):
        if m == 0:
            # This is true because b_0 is always fixed to be 0
            return np.identity(self.K)
        elif N == 1:
            return 1
        else:
            return np.kron(self.D(N - 1, m), self.E(N, m))

    def M(self, N):
        """Calculate matrix M."""
        key = ("M", N)
        if key in self._cache:
            return self._cache[key]
        else:
            M = self._M(N)
            self._cache[key] = M
            return M

    def _M(self, N):
        if N == 1:
            m = self.c.T @ self.A @ self.b
            return np.exp(TWO_PI * 1j * m)
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
