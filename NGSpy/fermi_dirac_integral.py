import numpy as np
from ngsolve import sqrt, exp, IfPos

a = 9.6
b = 2.13
c = 2.4
g = 2 / np.sqrt(np.pi)  # 1 / Gamma(3/2)
eps = 1e-8
clip_exp = 40.0


def F_half_aymerich_humet_np(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (
        3 * np.sqrt(2) * (b + x + (np.abs(x - b) ** c + a) ** (1 / c)) ** (-1.5)
        + g * np.exp(-x)
    )


def _abs_smooth(y):
    return sqrt(y * y + eps * eps)


def _clamp(val, bound):
    return IfPos(val - bound, bound, IfPos(-bound - val, -bound, val))


def _safe_exp(x):
    x_clip = _clamp(x, clip_exp)
    return exp(x_clip)


def F_half_aymerich_humet_ng(x):
    return 1.0 / (
        3 * sqrt(2) * (b + x + (_abs_smooth(x - b) ** c + a) ** (1 / c)) ** (-1.5)
        + g * _safe_exp(-x)
    )
