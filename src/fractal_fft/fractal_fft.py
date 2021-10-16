"""Code for FractalFFT implementation."""
import numpy as np

TWO_PI = 2 * np.pi


class FractalFFT:
    """Implementation of Fractal FFT algorithm.

    Described in 'A Fast Fourier Transform for Fractal Approximations'
    Reference: https://arxiv.org/pdf/1607.03690.pdf
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray):
        """Initializes FractalFFT.

        Args:
            A: d x d matrix
            B: K x d matrix, each column is a vector
            C: K x d matrix, each column is a vector
        """
        # TODO: error handling
        # TODO: check Hadamard, invertible
        self.A = A
        self.B = B
        self.C = C
        self._normalize()
        self.K = B.shape[0]
        self._cache = {}

    def _normalize(self):
        """Normalize to ensure b_0, c_0 are 0."""
        self.B = self.B - self.B[:, 0]
        self.C = self.C - self.C[:, 0]

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
        e = self.C.T @ (self.A ** N) @ self.B[:, m]
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
            m = self.C.T @ self.A @ self.B
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
