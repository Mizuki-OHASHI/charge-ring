import numpy as np
import matplotlib.pyplot as plt


def fermi_dirac_integral_1(x: np.ndarray) -> np.ndarray:
    """Fermi-Dirac integral of half-integer order (j=1/2) approximation"""
    return np.piecewise(
        x,
        [x > 25],
        [
            lambda x: (2 / np.sqrt(np.pi))
            * ((2 / 3) * x**1.5 + (np.pi**2 / 12) * x**-0.5),
            lambda x: np.exp(x) / (1 + 0.27 * np.exp(x)),
        ],
    )


def fermi_dirac_integral_2(x: np.ndarray) -> np.ndarray:
    """
    Fermi-Dirac integral of order 1/2 (j=1/2) using the approximation
    by Aymerich-Humet et al. (1981).
    
    This approximation is continuous and accurate across all regimes,
    transitioning smoothly from the Boltzmann limit (exp(x) for x << 0)
    to the degenerate limit (Sommerfeld expansion, first-order term).
    """
    
    # Aymerich-Humet et al. (1981) approximation parameters
    a1 = 6.316
    a2 = 12.92
    
    # Pre-factor for the degenerate limit (4 / (3 * sqrt(pi)))
    C_deg = 0.75224956896

    result = np.piecewise(
        x,
        [x < -10.0],  # Non-degenerate regime
        [
            lambda x: np.exp(x),
            lambda x: 1.0 / (np.exp(-x) + (C_deg * (x**2 + a1 * x + a2)**0.75)**(-1.0))
        ]
    )
    
    return result

xx = np.linspace(-10, 50, 500)
yy1 = fermi_dirac_integral_1(xx)
yy2 = fermi_dirac_integral_2(xx)
plt.plot(xx, yy1, label="Approximation 1", color="red")
plt.plot(xx, yy2, label="Approximation 2", linestyle="--", color="blue")
plt.yscale("log")
plt.xlabel("x")
plt.ylabel("F_{1/2}(x)")
plt.title("Fermi-Dirac Integral of Half-Integer Order (j=1/2)")
plt.grid(True)
plt.savefig("fermi_dirac_integral_j_half.png", dpi=300)