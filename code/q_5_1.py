# 5.1 Rendering Point Clouds from RGB-D Images (10 points)

import argparse
import imageio
import pytorch3d
import torch
import numpy as np
from tqdm import tqdm

from starter.utils import get_device, get_points_renderer, unproject_depth_image
from starter.render_generic import load_rgbd_data

def render_pointcloud(
    data_path="data/rgbd_data.pkl",
    image_size=256,
    use_pointcloud=1,
    num_points=1000,
    distance=7.0,
    radius=0.01,
    num_views=50,
    fps = 15,
    output_file="output/pointcloud.gif",
    device=None,
):
    """
    Render point clouds from RGB-D images.

    Args:
        data_path (str): Path to the RGB-D data file.
        image_size (int): Size of the output images.
        use_pointcloud (int): Which point cloud to use (1, 2, or 3 for union).
        num_points (int): Number of points to render (0 for all).
        distance (float): Distance of the camera from the origin.
        radius (float): Radius of the points.
        num_views (int): Number of equally spaced azimuthal views.
        fps (int): Frames per second.
        output_file (str): Path to the output gif file.
        device (str): Device to use (if None, use GPU if available).
        
    Returns:
        None (saves a gif to output_file).
    """

    if device is None:
        device = get_device()
    data = load_rgbd_data(path=data_path)
    
    # Select which point clouds to use
    rgb1 = torch.Tensor(data['rgb1'])
    mask1 = torch.Tensor(data['mask1'])
    depth1 = torch.Tensor(data['depth1'])
    camera1 = data['cameras1']
    points1, rgb1 = unproject_depth_image(rgb1, mask1, depth1, camera1)
    
    rgb2 = torch.Tensor(data['rgb2'])
    mask2 = torch.Tensor(data['mask2'])
    depth2 = torch.Tensor(data['depth2'])
    camera2 = data['cameras2']
    points2, rgb2 = unproject_depth_image(rgb2, mask2, depth2, camera2)

    if use_pointcloud == 1:
        points = points1
        rgb = rgb1
    elif use_pointcloud == 2:
        points = points2
        rgb = rgb2
    else:
        points = torch.cat([points1, points2], dim=0)
        rgb = torch.cat([rgb1, rgb2], dim=0)

    # Select a subset of points if specified
    if 0 < num_points < points.shape[0]:
        indices = torch.randperm(points.shape[0])[:num_points]
        points = points[indices]
        rgb = rgb[indices]

    points[:, 1] = -points[:, 1] # flip y axis
    points[:, 0] = -points[:, 0] # flip x axis

    pointcloud = pytorch3d.structures.Pointclouds(
        points=points.unsqueeze(0),
        features=rgb[:, :3].unsqueeze(0),
    ).to(device)

    # Initialize cameras with specified parameters
    theta_values = np.linspace(0, 360, num_views, endpoint=False)+180
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=distance,
        elev=0,
        azim=theta_values,
    )
    # R_relative_1=[[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    # R = torch.tensor(R_relative_1).float() @ R
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device,
    )

    renderer = get_points_renderer(image_size=image_size, device=device, radius=radius)
    
    # Save gif
    images = []
    for i in tqdm(range(num_views)):
        image = renderer(pointcloud, cameras=cameras[i])
        image = image.cpu().numpy()[0, ..., :3]
        images.append((image * 255).astype("uint8"))

    imageio.mimsave(output_file, images, fps=fps, loop=1000)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render point clouds from RGB-D images.")
    parser.add_argument("--data_path", type=str, default="data/rgbd_data.pkl", help="Path to the RGB-D data file.")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--use_pointcloud", type=int, default=1, help="Which point cloud to use (1, 2, or 3 for union).", choices=[1, 2, 3])
    parser.add_argument("--num_points", type=int, default=10000, help="Number of points to render (0 for all).")
    parser.add_argument("--distance", type=float, default=7.0, help="Distance of the camera from the origin.")
    parser.add_argument("--radius", type=float, default=0.01, help="Radius of the points.")
    parser.add_argument("--num_views", type=int, default=10, help="Number of equally spaced azimuthal views.")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second.")
    parser.add_argument("--output_file", type=str, default="output/pointcloud.gif", help="Path to the output gif file.")
    args = parser.parse_args()

    render_pointcloud(
        data_path=args.data_path,
        image_size=args.image_size,
        use_pointcloud=args.use_pointcloud,
        num_points=args.num_points,
        distance=args.distance,
        radius=args.radius,
        num_views=args.num_views,
        fps=args.fps,
        output_file=args.output_file
    )
