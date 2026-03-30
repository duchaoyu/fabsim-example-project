# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Initial clone (required for submodules)
git clone --recursive https://github.com/DavidJourdan/fabsim-example-project

# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all targets
cmake --build build

# Build a specific target
cmake --build build --target membrane_example

# Run tests
cd build && ctest
```

## Running Executables

All executables are in `build/` after compilation:

```bash
./build/rod_example              # Discrete elastic rod simulation
./build/membrane_example         # StVK FEM membrane simulation
./build/orthotropic_example      # Orthotropic material membrane
./build/gradient                 # Gradient/Hessian numerical verification
./build/best_fit_thickness       # Genetic algorithm for thickness optimization
```

## Architecture

This project is an example/research codebase demonstrating **FabSim** — a library for simulating elastic structures (rods, membranes, shells) with optimization support. All simulations use Polyscope for interactive 3D visualization.

### Three-layer stack

1. **fabsim** (`external/fabsim/include/fsim/`): Material models
   - Element classes: `StVKElement`, `NeoHookeanElement`, `OrthotropicStVKMembrane`, `ElasticRod`
   - Model classes compose elements into assemblies: `ElasticMembrane`, `ElasticShell`, `RodCollection`
   - All models expose `energy()`, `gradient()`, `hessian()` interface used by the solver

2. **optim** (`external/optim/`): Nonlinear solvers
   - Newton solver and L-BFGS; optionally uses CHOLMOD for fast sparse linear solves
   - Called from `src/` with a model and parameters

3. **polyscope** (`external/polyscope/`): OpenGL-based viewer
   - Each `src/` file registers meshes/curves and adds GUI callbacks via `polyscope::state::userCallback`

### Source files in `src/`

- `membrane.cpp`, `membrane_orthotropic.cpp`, `composite.cpp`, `inflation.cpp`, `rod.cpp`: Interactive simulators
- `genetic_algorithm*.cpp`, `genetic_knitting_directions.cpp`: Inverse-design optimizations (best-fit thickness, mesh calibration, height/direction optimization)
- `gradient_hessian_check.cpp`: Finite-difference validation of analytical derivatives
- `test_file.cpp`: Ad-hoc testing

### Data

Mesh inputs are in `data/` as `.off` files. The default mesh is `data/mesh.off`.

### Dependencies

- **Eigen3** (required): Linear algebra, must be installed system-wide
- **OpenMP** (optional): Parallel Hessian assembly in fabsim
- **CHOLMOD** (optional): Fast sparse solver in optim
