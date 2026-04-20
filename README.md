
# SRRM: Improving Recursive Transport Surrogates in the Small-Discrepancy Regime

This repository contains the official implementation for the paper: *"SRRM: Improving Recursive Transport Surrogates in the Small-Discrepancy Regime"*. It includes the core algorithm based on mass-median axis-recursive partitioning, along with scripts to reproduce various applications and numerical experiments.

## 📂 Repository Structure

Our repository is organized by specific applications and core algorithm components:

* **`base.cpp`**: The core C++ integration file. This file implements our proposed SRRM algorithm and serves as the unified wrapper that integrates both our method and the third-party baselines (BSP-OT and HCP) for fair comparison.
* **`setup.py`**: Python build script for compiling the C++ core (`base.cpp`) into a callable Python module.
* **Experiment Directories**: 
  * `3D_point_classification/`
  * `Autoencoder_small/`
  * `color_transfer/`
  * `shape_bijection_3d/`
  * `simu_flow/`
  *(These folders contain the Python scripts and configurations to reproduce the respective experiments presented in the paper).*
* **Third-Party Libraries**: `BSP-OT-main/` and `HCP-main/` (See Acknowledgments below).

## 🤝 Third-Party Libraries & Acknowledgments

To ensure full reproducibility of our comparative experiments, we have included the source code of two excellent open-source libraries in the root directory. We sincerely thank the original authors.

### 1. BSP-OT (MIT License)
* **Directory:** `BSP-OT-main/`
* **Source:** [baptiste-genest/BSP-OT](https://github.com/baptiste-genest/BSP-OT)
* **Usage:** We utilize the BSP-OT framework in two ways, integrated via `base.cpp`:
  1. **Baseline Comparison:** Evaluated as a state-of-the-art baseline in our experiments.
  2. **Direct Integration:** Our SRRM implementation directly calls the specific plan-merging logic from this library (`BSPOT::BijectiveMatching merged = BSPOT::MergePlans(...)`) to handle bijective matching aggregation. The novel mass-median partitioning logic of SRRM is independently implemented in `base.cpp`.

### 2. HCP - Hilbert Curve Projection (GPL-3.0 License)
* **Directory:** `HCP-main/`
* **Source:** [sherlockLitao/HCP](https://github.com/sherlockLitao/HCP)
* **Usage:** Included **strictly as a baseline** for performance comparison. It is wrapped in `base.cpp` solely to be executed alongside SRRM under identical computational environments. We use the original, unmodified logic provided by the authors.



## 🚀 How to Run

### 1. Prerequisites 
To compile the core C++ algorithm (`base.cpp`), please ensure you have the following installed:
* Python 3.8+
* A C++ compiler supporting C++20 (e.g., GCC 10+, MSVC, Clang)
* **pybind11**: Install via `pip install pybind11`
* **Eigen 3.4+**: Required for matrix operations.

### 2. Path Configuration
Before compiling, you need to point the compiler to your local Eigen directory. 
Open `setup.py` and modify the `include_dirs` list to match your local installation paths for Eigen. (Note: The path to `BSP-OT-main` is already configured relatively).

### 3. Compilation 
Once the paths are correctly set, open your terminal/command prompt in the root directory of this repository and run the following command to compile the C++ extension in-place:

```bash
python setup.py build_ext --inplace
