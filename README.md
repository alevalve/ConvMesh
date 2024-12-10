# ConvMesh

**GitHub Repository**: [ConvMesh](https://github.com/alevalve/ConvMesh.git)

## Overview

**ConvMesh** is a Python-based framework that uses convex optimization and PyTorch3D to process and optimize 3D meshes. The project includes two main functionalities:
1. **Improved Mesh Optimization**: Refines a source mesh to match a target mesh using convex optimization techniques.
2. **Generate Mesh**: Deforms and renders 3D meshes with loss-based optimization using PyTorch3D.

---

## Repository Structure

- `improved_mesh.py`: Optimizes a raw 3D mesh to align with a target mesh using convex optimization.
- `generate_mesh.py`: Implements deformable mesh optimization and rendering with PyTorch3D.
- `README.md`: Project documentation.
- `requirements.txt`: List of dependencies required for running the project.
- `dolphin.obj`: Dolphin mesh from Facebook AI.

---

## Installation

Follow these steps to set up and run the project:

### 1. Clone the Repository

```bash
git clone https://github.com/alevalve/ConvMesh.git
cd ConvMesh
```

### 2. Set Up the Environment

It is recommended to use a virtual environment to manage dependencies.

#### Create a Virtual Environment

```bash
python -m venv conv_mesh
source conv_mesh/bin/activate  # For Windows: conv_mesh\Scripts\activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

If you're using a CPU-only system (no CUDA/GPU), install PyTorch without CUDA:

```bash
pip install torch torchvision torchaudio
```

#### Install PyTorch3D

Install PyTorch3D for CPU-only usage:

```bash
pip install pytorch3d
```

---

## Usage

### 1. Improved Mesh Optimization (`improved_mesh.py`)

This script refines a source mesh to align with a target mesh using convex optimization.

#### Example Command

```bash
python improved_mesh.py --raw_mesh path/to/raw.obj --target_mesh path/to/target.obj --output_mesh path/to/output.obj --num_samples 2500 --delta 0.1 --lambd 0.3 --device cpu
```

#### Parameters

- `--raw_mesh`: Path to the raw/source `.obj` mesh file.
- `--target_mesh`: Path to the target `.obj` mesh file.
- `--output_mesh`: Path to save the optimized mesh.
- `--num_samples`: Number of points to sample from the meshes (default: 2500).
- `--delta`: Threshold for smoothness constraints (default: 0.1).
- `--lambd`: Regularization weight (default: 0.3).
- `--device`: Device to use (`cpu` or `cuda`).

---

### 2. Generate Mesh (`generate_mesh.py`)

This script deforms a 3D mesh using PyTorch3D's loss-based optimization and renders the output.

#### Example Command

```bash
python generate_mesh.py --target_mesh path/to/target.obj --output_mesh path/to/generated.obj --loss_output path/to/loss.csv --device cpu
```

#### Parameters

- `--target_mesh`: Path to the target `.obj` mesh file.
- `--output_mesh`: Path to save the generated mesh.
- `--loss_output`: Path to save loss values as a CSV (optional).
- `--device`: Device to use (`cpu` or `cuda`).
- Other parameters include optimization weights (`--w_chamfer`, `--w_edge`, `--w_normal`, `--w_laplacian`).

---

## Example Workflow

1. Optimize a raw mesh using `improved_mesh.py`.
2. Use the optimized mesh as input to `generate_mesh.py` for further deformation or rendering.

---

## Troubleshooting

- **Installation Issues**: Ensure all dependencies in `requirements.txt` are installed and compatible with your Python version.
- **PyTorch3D Errors**: Refer to the [PyTorch3D Installation Guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

---

## References

- [PyTorch3D Documentation](https://pytorch3d.readthedocs.io/)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [CVXPY Documentation](https://www.cvxpy.org/)
