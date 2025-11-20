import numpy as np
import matplotlib.pyplot as plt

d = np.loadtxt("electrostatic_energy.txt", comments="#")
mask = np.abs(d[:, 0]) < 2
V_tip = d[mask, 0]
energy = d[mask, 1]
min_energy_arg = np.argmin(energy)
print(
    f"Minimum energy at V_tip = {V_tip[min_energy_arg]} V: {energy[min_energy_arg]} J"
)
plt.plot(V_tip, energy, "b.")
plt.axvline(
    x=V_tip[min_energy_arg],
    color="r",
    linestyle="--",
    label=f"Min at {V_tip[min_energy_arg]} V",
)
plt.xlabel("Tip Voltage (V)")
plt.ylabel("Electrostatic Energy (J)")
plt.title("Electrostatic Energy vs Tip Voltage")
plt.grid()
plt.legend()
plt.savefig("electrostatic_energy_plot.png")
