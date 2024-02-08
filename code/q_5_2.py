# 5.2 Parametric Functions (10 + 5 points)

import argparse
import imageio
import numpy as np
import pytorch3d
import torch
from tqdm import tqdm

from starter.utils import get_device, get_points_renderer


def render_torus_360(
    image_size=256,
    num_samples=200,
    num_views=50,
    fps = 15,
    output_file="output/torus.gif",
    device=None
):
    """
    Renders a torus point cloud and creates a 360-degree gif.

    Args:
        image_size (int): Size of the output images.
        num_samples (int): Number of samples for torus point cloud.
        num_views (int): Number of equally spaced azimuthal views.
        fps (int): Frames per second.
        output_file (str): Path to the output gif file.
        device (str): Device to use (if None, use GPU if available).

    Returns:
        None (saves a gif to output_file).
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta, indexing='ij')

    R = 1
    r = 1/2
    x = (R + r * torch.cos(Theta)) * torch.cos(Phi)
    y = (R + r * torch.cos(Theta)) * torch.sin(Phi)
    z = r * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    renderer = get_points_renderer(image_size=image_size, device=device)
    
    # Save gif
    images = []
    for azimuth in tqdm(np.linspace(0, 360, num_views, endpoint=False)):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=4, elev=0, azim=azimuth, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(torus_point_cloud, cameras=cameras)
        image = rend[0, ..., :3].cpu().numpy()
        images.append((image * 255).astype("uint8"))

    imageio.mimwrite(output_file, images, fps=fps, loop=1000)


def render_klein_bottle_360(
    image_size=256,
    num_samples=200,
    num_views=50,
    fps=15,
    output_file="output/klein_bottle.gif",
    device=None
):
    """
    Renders a Klein bottle point cloud and creates a 360-degree gif.

    Args:
        image_size (int): Size of the output images.
        num_samples (int): Number of samples for the Klein bottle point cloud.
        num_views (int): Number of equally spaced azimuthal views.
        fps (int): Frames per second.
        output_file (str): Path to the output gif file.
        device (str): Device to use (if None, use GPU if available).

    Returns:
        None (saves a gif to output_file).
    """

    if device is None:
        device = get_device()

    # Klein bottle parametric representation
    u = torch.linspace(0, np.pi, num_samples)
    v = torch.linspace(0, 2 * np.pi, num_samples)
    U, V = torch.meshgrid(u, v, indexing='ij')

    x = -2/15 * torch.cos(U) * (3 * torch.cos(V) - 30 * torch.sin(U) + 90 * torch.cos(U)**4 * torch.sin(U) - 60 * torch.cos(U)**6 * torch.sin(U) + 5 * torch.cos(U) * torch.cos(V) * torch.sin(U))
    y = -1/15 * torch.sin(U) * (3 * torch.cos(V) - 3 * torch.cos(U)**2 * torch.cos(V) - 48 * torch.cos(U)**4 * torch.cos(V) + 48 * torch.cos(U)**6 * torch.cos(V) - 60 * torch.sin(U) + 5 * torch.cos(U) * torch.cos(V) * torch.sin(U) - 5 * torch.cos(U)**3 * torch.cos(V) * torch.sin(U) - 80 * torch.cos(U)**5 * torch.cos(V) * torch.sin(U) + 80 * torch.cos(U)**7 * torch.cos(V) * torch.sin(U))
    z = 2/15 * (3 + 5 * torch.cos(U) * torch.sin(U)) * torch.sin(V)
    y = y - 2

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    klein_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    renderer = get_points_renderer(image_size=image_size, device=device)

    # Save gif
    images = []
    for azimuth in tqdm(np.linspace(0, 360, num_views, endpoint=False)):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=5, elev=0, azim=azimuth, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(klein_point_cloud, cameras=cameras)
        image = rend[0, ..., :3].cpu().numpy()
        images.append((image * 255).astype("uint8"))

    imageio.mimsave(output_file, images, fps=fps, loop=1000)

    # Save gif (another view)
    images = []
    for azimuth in tqdm(np.linspace(0, 360, num_views, endpoint=False)):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=5, elev=0, azim=azimuth, device=device)
        R_relative=[[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        R = torch.tensor(R_relative).float().to(device) @ R @ torch.tensor(R_relative).float().to(device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(klein_point_cloud, cameras=cameras)
        image = rend[0, ..., :3].cpu().numpy()
        images.append((image * 255).astype("uint8"))

    imageio.mimsave(output_file.replace(".gif", "_2.gif"), images, fps=fps, loop=1000)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a point cloud from a parametric equation and create a 360-degree gif.")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--surface", type=str, default="torus", help="Surface to render (torus or klein).", choices=["torus", "klein"])
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples for point cloud.")
    parser.add_argument("--num_views", type=int, default=60, help="Number of equally spaced azimuthal views.")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second.")
    parser.add_argument("--output_file", type=str, default="output/pointcloud.gif", help="Path to the output gif file.")
    args = parser.parse_args()

    if args.surface == "torus":
        render_torus_360(
            image_size=args.image_size,
            num_samples=args.num_samples,
            num_views=args.num_views,
            fps=args.fps,
            output_file=args.output_file,
        )
    else:
        render_klein_bottle_360(
            image_size=args.image_size,
            num_samples=args.num_samples,
            num_views=args.num_views,
            fps=args.fps,
            output_file=args.output_file,
        )