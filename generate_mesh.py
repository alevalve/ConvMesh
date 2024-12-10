# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import torch
import pandas as pd
import argparse
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

# Build class for Mesh preprocessing

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
  print("Warning, is running in CPU")

class MeshManager:
    def __init__(self, obj_file, device=device):
        self.device = torch.device(device)
        self.mesh = self.load_and_preprocess(obj_file)

    def _load_and_preprocess(self, file_path):
        verts, faces, _ = load_obj(file_path)
        faces_idx = faces.verts_idx.to(self.device)
        verts = verts.to(self.device)

        # Normalize the mesh for simplicity
        center = verts.mean(0)
        verts = verts - center
        scale = verts.abs().max()
        verts = verts / scale

        return Meshes(verts=[verts], faces=[faces_idx])
    
    def save(self, output_path):
        verts, faces = self.mesh.get_mesh_verts_faces(0)
        save_obj(output_path, verts, faces)

    def get_mesh(self):
        return self.mesh

class MeshDeformer:
    def __init__(self, optimizer, optimizer_type, device=device):
        self.device = torch.device(device)
        self.src_mesh = ico_sphere(4, device)
        self.deform_verts = torch.full(
            self.src_mesh.verts_packed().shape, 0.0, device=self.device, requires_grad=True
        )
        self.optimizer = optimizer("SGD", [self.deform_verts], 1.0, 0.9)

    def optimize(self, target_mesh, iterations=250, weights=None, num_samples=2500):
        """
        Optimize the source mesh to deform towards the target mesh.

        Args:
            target_mesh (Meshes): Target mesh to match.
            iterations (int): Number of optimization iterations.
            weights (dict): Weights for loss terms. Keys: "chamfer", "edge", "normal", "laplacian".
            num_samples (int): Number of points to sample from the meshes during loss computation.

        Returns:
            new_src_mesh (Meshes): The deformed source mesh.
            losses (dict): Loss values recorded during optimization.
        """

        if weights is None:
            weights =  {"chamfer": 1.0, "edge": 1.0, "normal": 0.01, "laplacian": 0.1}
        
        losses = {"chamfer": [], "edge": [], "normal": [], "laplacian":[]}

        for _ in range(iterations):
            self.optimizer.zero_grad()

            # Deform the source_mesh
            new_src_mesh = self.src_mesh.offset_verts(self.deform_verts)
            
            # Compute loss
            loss, loss_dict = self.compute_loss(new_src_mesh, target_mesh, weights, num_samples)
            loss.backward()
            self.optimizer.step()

            # Recod loss values
            for key in losses:
                losses[key].append(loss_dict[key].item())

        return new_src_mesh, losses
    
    def compute_loss(self, src_mesh, target_mesh, weights, num_samples=2500):
        sample_trg = sample_points_from_meshes(target_mesh, num_samples)
        sample_src = sample_points_from_meshes(src_mesh, num_samples)

        loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
        loss_edge = mesh_edge_loss(src_mesh)
        loss_normal = mesh_normal_consistency(src_mesh)
        loss_laplacian = mesh_laplacian_smoothing(src_mesh)

        total_loss = (
            weights["chamfer"] * loss_chamfer + 
            weights["edge"] * loss_edge +
            weights["normal"] * loss_normal +
            weights["laplacian"] * loss_laplacian
        )

        return total_loss, {
            "chamfer":loss_chamfer,
            "edge": loss_edge,
            "normal": loss_normal, 
            "laplacian": loss_laplacian
        }

def create_optimizer(optimizer_type, params, lr, momentum):
        if optimizer_type == "SGD":
            return torch.optim.SGD(params, lr=lr, momentum=momentum)
        elif optimizer_type == "ADAM":
            return torch.optim.Adam(params, lr=lr)
        else:
            raise NotImplementedError(f"Unsupported optimizer type: {optimizer_type}")



def main(args):

    # Set the device

    device = torch.device(args.device)

    # Load the target mesh

    trg_manager = MeshManager(args.target_mesh, device=device)
    optimizer = create_optimizer(args.optimizer, [], args.learning_rate, args.momentum)
    deformer = MeshDeformer(optimizer=optimizer, optimizer_type=args.optimizer, device=device)

    # Optimize the source mesh to match the target

    weights = {
        "chamfer": args.w_chamfer,
        "edge": args.w_edge,
        "normal": args.w_normal,
        "laplacian": args.w_laplacian,
    }

    new_mesh, losses = deformer.optimize(
        trg_manager.get_mesh(),
        weights=weights,
        num_samples=args.num_samples,
    )

    # Save the new mesh

    trg_manager.mesh = new_mesh
    trg_manager.save(args.output_mesh)
    print(f"Deformed mesh saved to {args.output_mesh}")
    
    # Save the loss data if specified
    if args.loss_output:
        pd.DataFrame(losses).to_csv(args.loss_output, index=False)
        print(f"Losses saved to {args.loss_output}")

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Deform a 3D mesh using PyTorch3D")

    parser.add_argument("--target_mesh", type=str, required=True, help="Path to the target file")
    parser.add_argument("--output_mesh", type=str, required=True, help="Path to store the new mesh")
    parser.add_argument("--loss_output", type=str, default=None, help="Path to save the loss data as a csv file")

    #Optimization parameters
    parser.add_argument("--iteration", type=int, default=250, help="Number of iterations")
    parser.add_argument("--num_samples", type=int, default=2500, help="Number of points to sample from the meshes")
    parser.add_argument("--device",type=str, default='cuda', help='Device to use')
    parser.add_argument("--optimizer", type=str, default='SGD', choices=["SGD", "ADAM"], help="Optimizer type")
    parser.add_argument("--learning_rate", type=float, default=1.0, help="Learning rate for the optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer (ignored for ADAM)")

    # Loss weights
    parser.add_argument("--w_chamfer", type=float, default=1.0, help="Weight for chamfer loss")
    parser.add_argument("--w_edge", type=float, default=1.0, help="Weight for edge loss")
    parser.add_argument("--w_normal", type=float, default=0.01, help="Weight for normal consistency loss")
    parser.add_argument("--w_laplacian", type=float, default=0.1, help="Weight for Laplacian smoothing loss")

    args = parser.parse_args()

    main(args)
    

            
    

