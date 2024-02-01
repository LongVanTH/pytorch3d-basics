# 5.2 Parametric Functions (10 + 5 points)

import argparse
import imageio
import numpy as np
import pytorch3d
import torch
import mcubes
from tqdm import tqdm

from starter.utils import get_device, get_mesh_renderer, get_points_renderer


def render_torus_mesh_360(
    image_size=256,
    voxel_size=64,
    num_views=50,
    fps=15,
    output_file="output/torus_mesh.gif",
    device=None
):
    """
    Renders a torus mesh and creates a 360-degree gif.

    Args:
        image_size (int): Size of the output images.
        voxel_size (int): Voxel size for discretizing 3D space.
        num_views (int): Number of equally spaced azimuthal views.
        fps (int): Frames per second.
        output_file (str): Path to the output gif file.
        device (str): Device to use (if None, use GPU if available).

    Returns:
        None (saves a gif to output_file).
    """

    if device is None:
        device = get_device()

    min_value = -2
    max_value = 2
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3, indexing="ij")
    # Implicit function for a torus
    R = 1
    r = 1/2
    torus_implicit_function = (R - torch.sqrt(X ** 2 + Y ** 2)) ** 2 + Z ** 2 - r ** 2

    # Extract the mesh using marching cubes
    voxels = mcubes.smooth(torus_implicit_function)
    vertices, faces = mcubes.marching_cubes(voxels, isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    # Normalize vertex coordinates
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value

    # Create mesh structure
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))
    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)

    # Setup lights and renderer
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    # Save gif
    images = []
    for azimuth in tqdm(np.linspace(0, 360, num_views, endpoint=False)):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=4, elev=0, azim=azimuth, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        image = rend[0, ..., :3].cpu().numpy().clip(0, 1)
        images.append((image * 255).astype("uint8"))

    imageio.mimwrite(output_file, images, fps=fps, loop=1000)


def render_tanglecube_mesh_360(
    image_size=256,
    voxel_size=64,
    num_views=50,
    fps=15,
    output_file="output/tanglecube_mesh.gif",
    device=None
):
    """
    Renders a tanglecube mesh and creates a 360-degree gif.

    Args:
        image_size (int): Size of the output images.
        voxel_size (int): Voxel size for discretizing 3D space.
        num_views (int): Number of equally spaced azimuthal views.
        fps (int): Frames per second.
        output_file (str): Path to the output gif file.
        device (str): Device to use (if None, use GPU if available).

    Returns:
        None (saves a gif to output_file).
    """

    if device is None:
        device = get_device()

    min_value = -3.5
    max_value = 3.5
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3, indexing="ij")
    # Implicit function for a tanglecube
    tanglecube_implicit_function = X**4 - 5*X**2 + Y**4 - 5*Y**2 + Z**4 - 5*Z**2 + 11.8

    # Extract the mesh using marching cubes
    voxels = mcubes.smooth(tanglecube_implicit_function)
    vertices, faces = mcubes.marching_cubes(voxels, isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    # Normalize vertex coordinates
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value

    # Create mesh structure
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))
    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)

    # Setup lights and renderer
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -8.]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    # Save gif
    images = []
    for azimuth in tqdm(np.linspace(0, 360, num_views, endpoint=False)):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=8, elev=0, azim=azimuth, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        image = rend[0, ..., :3].cpu().numpy().clip(0, 1)
        images.append((image * 255).astype("uint8"))

    imageio.mimwrite(output_file, images, fps=fps, loop=1000)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a mesh surface from an implicit equation and create a 360-degree gif.")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--surface", type=str, default="torus", help="Surface to render (torus or tanglecube).", choices=["torus", "tanglecube"])
    parser.add_argument("--voxel_size", type=int, default=64, help="Voxel size for discretizing 3D space.")
    parser.add_argument("--num_views", type=int, default=60, help="Number of equally spaced azimuthal views.")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second.")
    parser.add_argument("--output_file", type=str, default="output/torus_mesh.gif", help="Path to the output gif file.")
    args = parser.parse_args()

    if args.surface == "torus":
        render_torus_mesh_360(
            image_size=args.image_size,
            voxel_size=args.voxel_size,
            num_views=args.num_views,
            fps=args.fps,
            output_file=args.output_file,
        )
    else:
        render_tanglecube_mesh_360(
            image_size=args.image_size,
            voxel_size=args.voxel_size,
            num_views=args.num_views,
            fps=args.fps,
            output_file=args.output_file,
        )