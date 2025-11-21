import numpy as np
import matplotlib.pyplot as plt
import mpl_backend_ssh  # noqa: F401

data = [
    ["_dia1E11V027", "200 nm", "-k", {"lw": 1}],
    ["_thick", "500 nm", "-r", {"lw": 1}],
    ["_thick2", "1000 nm", "--b", {"lw": 1}],
]

plt.figure(figsize=(8, 6))
for folder, label, style, kwargs in data:
    filepath = f"{folder}/line_profile_vertical.txt"
    loaded = np.loadtxt(filepath, comments="#")
    mask = loaded[:, 0] <= 2
    z = loaded[mask, 0]  # in nm
    potential = loaded[mask, 1]  # in mV
    plt.plot(z, potential, style, label=label, **kwargs)

plt.xlabel("Z / nm")
plt.ylabel("Potential / mV")
plt.legend()
plt.title("Vertical Potential Profiles")
plt.grid(True)
plt.show()
