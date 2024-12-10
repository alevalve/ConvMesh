# Use convex optimization to improve mesh

import os
import torch
import argparse
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
import cvxpy as cp
from generate_mesh import *
sys.path.append(os.path.abspath(''))

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
  print("Warning, is running in CPU")




class ConvexOptim:
    def __init__(self, raw_mesh, target_mesh, num_samples, device="cpu"):
        """
        Initialize the ConvexOptim class with raw and target meshes.

        Args:
            raw_mesh (Meshes): The raw/source mesh.
            target_mesh (Meshes): The target mesh.
            num_samples (int): Number of points to sample from the target mesh.
            device (str): Device to use ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.raw_mesh = raw_mesh
        self.target_mesh = target_mesh
        self.num_samples = num_samples

    def process_mesh(self):
        """
        Process the raw and target meshes to extract vertices, faces, and target points.

        Returns:
            vertices_target (np.ndarray): Target mesh vertices.
            n_vertices (int): Number of vertices in the raw mesh.
            vertices_cvx (cp.Variable): CVXPY variable for convex optimization.
            faces (np.ndarray): Faces of the raw mesh.
        """
        # Sample points from the target mesh
        target_points = sample_points_from_meshes(self.target_mesh, self.num_samples).squeeze().cpu().numpy()

        # Extract vertices and faces from the raw mesh
        vertices = self.raw_mesh.verts_packed().cpu().numpy()
        faces = self.raw_mesh.faces_packed().cpu().numpy()

        # Extract vertices from the target mesh
        vertices_target = self.target_mesh.verts_packed().cpu().numpy()

        # Initialize CVXPY variable for vertices
        n_vertices = vertices.shape[0]
        vertices_cvx = cp.Variable((n_vertices, 3))

        return vertices_target, n_vertices, vertices_cvx, faces, target_points

    def conv_constraints(self, delta=0.1, lambd=0.3):
        """
        Define the constraints and the objective for convex optimization.

        Args:
            delta (float): Maximum allowed distance for smoothness constraints.

        Returns:
            constraints (list): List of CVXPY constraints for optimization.
            objective (cvxpy.Expression): The objective function for optimization.
        """
        # Process the mesh to get vertices and faces
        vertices_target, n_vertices, vertices_cvx, faces, target_points = self.process_mesh()

        # Define smoothness constraints (Laplacian-like)
        smoothness_constraints = []
        for face in faces:
            for i in range(3):
                smoothness_constraints.append(
                    cp.norm(vertices_cvx[face[i]] - vertices_cvx[face[(i + 1) % 3]]) <= delta
                )

        # Regularization term to keep vertices close to their target positions
        regularization_term = cp.sum_squares(vertices_cvx - vertices_target)

        # Approximate alignment loss
        approx_alignment_loss = cp.sum(cp.norm(vertices_cvx - target_points.mean(), axis=1))

        # Define the objective function (minimize alignment loss + regularization)
        objective = cp.Minimize(approx_alignment_loss + lambd * regularization_term)

        return smoothness_constraints, objective
    
    def training(self, verbose=False, delta=0.1, lamb=0.3):
        """
        Train the convex optimization problem.

        Args:
            verbose (bool): If True, print optimization details.
            delta (float): Smoothness constraint threshold.
            lamb (float): Regularization weight (can be used to adjust balance).

        Returns:
            dict: Dictionary containing the optimization status and result.
        """
        # Get constraints and objective
        smoothness_constraints, objective = self.conv_constraints(delta=delta)

        # Solve the optimization problem
        problem = cp.Problem(objective, smoothness_constraints)
        result = problem.solve(verbose=verbose)

        # Return the optimization status and result
        return {
            "status": problem.status,
            "optimal_value": problem.value,
        }



    
def main(args):
    device = torch.device(args.device)

    # Load the raw and target meshes
    raw_mesh = load_obj(args.raw_mesh)[0].to(device)
    target_mesh = load_obj(args.target_mesh)[0].to(device)
    
    # Normalize the meshes
    raw_manager = MeshManager(args.raw_mesh, device=device)
    target_manager = MeshManager(args.target_mesh, device=device)
    
    raw_mesh = raw_manager.get_mesh()
    target_mesh = target_manager.get_mesh()

    # Initialize ConvexOptim
    convex_optimizer = ConvexOptim(
        raw_mesh=raw_mesh,
        target_mesh=target_mesh,
        num_samples=args.num_samples,
        device=args.device,
    )

    # Train the model
    result = convex_optimizer.training(
        verbose=args.verbose,
        delta=args.delta,
        lambd=args.lambd,
    )

    # Save results
    print(f"Optimization Status: {result['status']}")
    print(f"Optimal Value: {result['optimal_value']}")

    # Save the processed raw mesh
    raw_manager.mesh = raw_mesh
    raw_manager.save(args.output_mesh)
    print(f"Processed mesh saved to {args.output_mesh}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convex optimization for 3D meshes")

    # Input and output files
    parser.add_argument("--raw_mesh", type=str, required=True, help="Path to the raw mesh (source mesh) file")
    parser.add_argument("--target_mesh", type=str, required=True, help="Path to the target mesh file")
    parser.add_argument("--output_mesh", type=str, required=True, help="Path to save the optimized mesh")

    # Optimization parameters
    parser.add_argument("--num_samples", type=int, default=2500, help="Number of points to sample from the meshes")
    parser.add_argument("--delta", type=float, default=0.1, help="Smoothness constraint threshold (delta)")
    parser.add_argument("--lambd", type=float, default=0.3, help="Regularization weight (lambda)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output during optimization")

    # Device configuration
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., 'cpu' or 'cuda')")

    args = parser.parse_args()

    main(args)

        








  







