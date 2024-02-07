# 6. Do Something Fun (10 points)

import argparse
import imageio
import pytorch3d
import torch
import numpy as np
from tqdm import tqdm

from starter.utils import get_device, get_mesh_renderer

def load_mesh(mesh_path, device=None):
    """
    Load a mesh from the specified path.

    Args:
        mesh_path (str): Path to the mesh file.
        device (str): Device to use (if None, use GPU if available).

    Returns:
        pytorch3d.structures.Meshes: Loaded mesh.
    """
    if device is None:
        device = get_device()

    # Load mesh
    vertices, faces, _ = pytorch3d.io.load_obj(mesh_path)
    faces = faces.verts_idx
    vertices = vertices.unsqueeze(0)
    faces = faces.unsqueeze(0)

    # Define colors (adjust this if needed)
    if mesh_path == "data/Pikachu OBJ.obj":
        mean = vertices.mean(1, keepdim=True)
        vertices = vertices - mean
        color1 = [1, 1, 0]
        color2 = [1, 1, 0.3]
        z_min = vertices[:, :, 2].min()
        z_max = vertices[:, :, 2].max()
        alpha = (vertices[:, :, 2] - z_min) / (z_max - z_min)
        colors = alpha.unsqueeze(-1) * torch.tensor(color2) + (1 - alpha.unsqueeze(-1)) * torch.tensor(color1)
    elif mesh_path == "data/Pokeball.obj":
        mean = vertices.mean(1, keepdim=True)
        vertices = vertices - mean
        relative_rotation = pytorch3d.transforms.euler_angles_to_matrix(torch.tensor([-0.35, 0, -0.3]), "XYZ")
        relative_rotation = relative_rotation.to(vertices.dtype)
        vertices = vertices @ relative_rotation
        color_bottom = [0.7, 0.7, 0.7]  # Light gray for bottom
        color_upper = [1, 0, 0]  # Red for upper
        color_stripe = [0, 0, 0]  # Black for stripe
        y_min = vertices[:, :, 1].min()
        y_max = vertices[:, :, 1].max()
        stripe_height = 0.07 * (y_max - y_min)
        stripe_center = (y_max + y_min) / 2
        colors = torch.zeros_like(vertices) * torch.tensor(color_stripe)
        colors[vertices[:, :, 1] > stripe_center + stripe_height / 2] = torch.tensor(color_upper, dtype=torch.float32)
        colors[vertices[:, :, 1] < stripe_center - stripe_height / 2] = torch.tensor(color_bottom, dtype=torch.float32)

    # Move tensors to the same device
    vertices = vertices.to(device)
    faces = faces.to(device)
    colors = colors.to(device)

    # Create textures
    textures = pytorch3d.renderer.TexturesVertex(colors)

    return pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=textures
    ).to(device)


def scene(
    image_size=256,
    output_file="output/scene.gif",
    device=None
):
    """
    Animates a scene where a ball is thrown towards Pikachu.

    Args:
        image_size (int): Size of the output images.
        output_file (str): Path to the output gif file.
        device (str): Device to use (if None, use GPU if available).

    Returns:
        None (saves a gif to output_file).
    """

    # Use the specified device or get the available device
    if device is None:
        device = get_device()

    # Load meshes
    pikachu_mesh = load_mesh("data/Pikachu OBJ.obj", device)
    pokeball_mesh = load_mesh("data/Pokeball.obj", device)

    # Set up renderer, lights, and camera
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, 10.0]], device=device)
   
    # Create gif
    images = []
    num_frames = 50
    i = 0
    # Pokeball moving (parabolic trajectory)
    vertices = pokeball_mesh._verts_padded
    for frame in tqdm(range(num_frames)):
        # Meshes
        k = frame / num_frames
        R_1 = pytorch3d.transforms.euler_angles_to_matrix(torch.tensor([-k*5*np.pi, 0, 0]), "XYZ").to(device)
        T_1 = torch.tensor([0.0, 15*k*(1-k), 17*(1-k)]).float().to(device)
        rot_pokeball_mesh = pokeball_mesh.update_padded(vertices @ R_1.T + T_1)
        combined_mesh = pytorch3d.structures.join_meshes_as_scene([pikachu_mesh, rot_pokeball_mesh])
        # Camera
        R, T = pytorch3d.renderer.look_at_view_transform(dist=15, elev=20, azim=40+i, device=device)
        T = T + torch.tensor([-4, 0, 0]).float().to(device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        i += 1
        # Render
        combined_rend = renderer(combined_mesh, cameras=cameras, lights=lights)
        image = combined_rend[0, ..., :3].cpu().numpy().clip(0, 1)
        images.append((image * 255).astype("uint8"))
    # Pikachu shrinking
    vertices = pikachu_mesh._verts_padded
    for frame in tqdm(range(num_frames)):
        # Meshes
        k = frame / num_frames
        new_pikachu_mesh = pikachu_mesh.update_padded(vertices * (1 - k))
        combined_mesh = pytorch3d.structures.join_meshes_as_scene([new_pikachu_mesh, pokeball_mesh])
        # Camera
        R, T = pytorch3d.renderer.look_at_view_transform(dist=15, elev=20, azim=40+i, device=device)
        T = T + torch.tensor([-4, 0, 0]).float().to(device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        i += 1
        # Render
        combined_rend = renderer(combined_mesh, cameras=cameras, lights=lights)
        image = combined_rend[0, ..., :3].cpu().numpy().clip(0, 1)
        images.append((image * 255).astype("uint8"))

    # Save the gif
    imageio.mimwrite(output_file, images, fps=30, loop=1000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_file", type=str, default="output/scene.gif")
    args = parser.parse_args()
    scene(
        image_size=args.image_size,
        output_file=args.output_file
    )

