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
python main.py [--help]
```

## Run using GPU

FEniCSx managed by conda does not support GPU acceleration. You need to build FEniCSx from source with GPU support. The easiest way is to use Docker.

To check if your GPU is visible to Docker, run the command below. You should see your GPU details if everything is set up correctly.

```sh
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

Then pull the Docker image with FEniCSx and GPU support.

```sh
docker pull dolfinx/dolfinx:stable
```

Run the Docker container with GPU support and mount the repository root directory to `/root/shared` in the container.

```sh
docker run -it --rm --gpus all -v "$(pwd)":/root/shared dolfinx/dolfinx:stable bash
```

**Inside the Docker container:**

```sh
cd /root/shared
# Install additional dependencies
pip install scipy adios4dolfinx
mpiexec -n 1 python main.py --petsc-args -vec_type cuda -mat_type aijcusparse
```
