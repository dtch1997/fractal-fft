import numpy as np

TWO_PI = 2 * np.pi

class FractalFFT:
    """ 
    Implementation of Fractal FFT method described in
    'A Fast Fourier Transform for Fractal Approximations'
    Reference: https://arxiv.org/pdf/1607.03690.pdf

    Args:
        :param A: d x d matrix
        :param B: K x d matrix, each column is a vector
        :param C: K x d matrix, each column is a vector
    """
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray):
        # TODO: error handling
        # TODO: check Hadamard, invertible
        self.A = A
        self.B = B
        self.C = C
        self._normalize()
        self.K = B.shape[0]

    def _normalize(self):
        """ Normalize to ensure b_0, c_0 are 0 """
        self.B = self.B - self.B[:,0]
        self.C = self.C - self.C[:,0]

    def E(self, N, m):
        # TODO: implement caching to reduce computation
        return self._E(N, m)

    def _E(self, N, m):
        e = self.C.T @ (self.A ** N) @ self.B[:,m]
        e = np.stack([e] * self.K, axis=1)
        return np.exp(-TWO_PI * 1j * e)

    def D(self, N, m):
        # TODO: implement caching to reduce computation
        return self._D(N, m)

    def _D(self, N, m):
        if m == 0:
            # This is true because b_0 is always fixed to be 0
            return np.identity(self.K)
        elif N == 1:
            return 1
        else:
            return np.kron(self.D(N-1, m), self.E(N, m))

    def M(self, N):
        # TODO: implement caching to reduce computation
        return self._M(N)

    def _M(self, N):
        if N == 1:
            m = self.C.T @ self.A @ self.B
            return np.exp(TWO_PI * 1j * m)
        else:
            M_1 = self.M(1)
            M_prev = self.M(N-1)
            dim_prev = M_prev.shape[0]
            M_N = np.zeros(shape = (dim_prev * self.K, dim_prev * self.K), dtype = np.csingle)
            # Calculate M_N blockwise
            for col in range(self.K):
                col_start_idx = col * dim_prev 
                col_end_idx = (col + 1) * dim_prev
                D_N_col_M_prev = self.D(N, col) @ M_prev
                for row in range(self.K):
                    row_start_idx = row * dim_prev 
                    row_end_idx = (row + 1) * dim_prev
                    M_N[row_start_idx:row_end_idx, col_start_idx: col_end_idx] = M_1[row, col] * D_N_col_M_prev
            return M_N

if __name__ == "__main__":
    A = np.identity(2)
    B = np.random.uniform(size=(2,2))
    C = np.random.uniform(size=(2,2))
    fractal_fft = FractalFFT(A, B, C)
    print(fractal_fft.E(2,1).shape)
    print(fractal_fft.D(2,1).shape)
    print(fractal_fft.M(2))