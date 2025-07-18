import numpy as np

class SystemProperties:
    @staticmethod
    def is_linear(system, x1, x2, a1, a2):
        left = system(a1 * x1 + a2 * x2)
        right = a1 * system(x1) + a2 * system(x2)
        return np.allclose(left, right)

    @staticmethod
    def is_time_invariant(system, x, shift):
        n = np.arange(len(x))
        x_shifted = np.roll(x, shift)
        y = system(x)
        y_shifted = np.roll(y, shift)
        return np.allclose(system(x_shifted), y_shifted)

    @staticmethod
    def is_causal(system, x):
        y_full = system(x)
        for n in range(1, len(x)):
            x_trunc = np.copy(x)
            x_trunc[n:] = 0
            y_trunc = system(x_trunc)
            if not np.allclose(y_trunc[:n+1], y_full[:n+1]):
                return False
        return True

    @staticmethod
    def is_stable(system, x):
        y = system(x)
        return np.all(np.abs(y) <= 1.1 * np.max(np.abs(x))) 