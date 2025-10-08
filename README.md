# Charge Ring Simulation by Finite Element Method

## Setup

### Requirements

Note: This code has been tested only on MacOS (M3) and WSL2 (Ubuntu 22.04).

### Install dependencies

Conda maybe too heavy, so we use `micromamba` here.

```sh
micromamba create -n fem2 -c conda-forge python=3.11 gmsh python-gmsh fenics-dolfinx adios4dolfinx numpy scipy matplotlib -y
```

### Activate environment

```sh
micromamba activate fem2
```