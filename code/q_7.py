# 7. Sampling Points on Meshes (Extra Credit) (10 points)

import argparse
import torch
import pytorch3d
import imageio
import numpy as np
from tqdm import tqdm

from starter.utils import get_device, get_points_renderer


def render_360_view(
        pointcloud, 
        image_size=256,
        number_of_frames=30,
        fps = 15,
        output_file="output/render_360.gif"
    ):
    """
    Render a 360-degree gif of a point cloud.

    Args:
        pointcloud (pytorch3d.structures.Pointclouds): Point cloud to render.
        image_size (int): Size of the output images.
        number_of_frames (int): Number of frames in the gif.
        fps (int): Frames per second.
        output_file (str): Path to the output gif file.

    Returns:
        None (saves a gif to output_file).
    """
    device = get_device()
    renderer = get_points_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -2.0]], device=device)

    images = []
    theta_values = [360 * (t*t/(2*(t*t-t)+1)) + 180 for t in np.linspace(0, 1, number_of_frames, endpoint=False)]
    for i in tqdm(range(number_of_frames)):
        theta = theta_values[i]
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(3.0, 0, theta)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R[0].t().unsqueeze(0), T=T[0].unsqueeze(0),
            fov=60, device=device
        )

        rend = renderer(pointcloud, cameras=cameras, lights=lights)
        image = rend[0, ..., :3].cpu().numpy().clip(0, 1)
        images.append((image * 255).astype("uint8"))

    imageio.mimwrite(output_file, images, fps=fps, loop=1000)


def sample_points_on_meshes(mesh, num_points=1000):
    """
    Sample points on a mesh.

    Args:
        mesh (pytorch3d.structures.Meshes): Mesh to sample points from.
        num_points (int): Number of points to sample.

    Returns:
        torch.Tensor: Sampled points.
    """
    verts = mesh.verts_padded()[0]
    faces = mesh.faces_padded()[0]
    areas = mesh.faces_areas_packed()

    face_idx = torch.multinomial(areas/areas.sum(), num_points, replacement=True)
    face_verts = verts[faces[face_idx]] # vertices of the sampled faces

    u = torch.sqrt(torch.rand(num_points))
    v = torch.rand(num_points)
    points = (1 - u.view(-1, 1)) * face_verts[:, 0] + u.view(-1, 1) * (1 - v.view(-1, 1)) * face_verts[:, 1] + \
             u.view(-1, 1) * v.view(-1, 1) * face_verts[:, 2]

    return points


def points_to_pointcloud(points):
    """
    Convert points to a point cloud.

    Args:
        points (torch.Tensor): Points to convert.

    Returns:
        pytorch3d.structures.Pointclouds: Point cloud.
    """

    device = points.device
    color = (points - points.min()) / (points.max() - points.min())

    pointcloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    return pointcloud


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a 360-degree view of a point cloud sampled from a mesh.")
    parser.add_argument("--mesh_path", type=str, default="data/cow.obj", help="Path to the input mesh (obj format).")
    parser.add_argument("--output_file", type=str, default="output/cow_360_points_1000.gif", help="Path to the output gif file.")
    parser.add_argument("--image_size", type=int, default=256, help="Size of the output images.")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames in the gif.")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second.")
    parser.add_argument("--num_points", type=int, default=1000, help="Number of points to sample on the mesh.")
    
    args = parser.parse_args()

    device = get_device()
    
    verts, faces, _ = pytorch3d.io.load_obj(args.mesh_path)
    faces = faces.verts_idx
    mesh = pytorch3d.structures.Meshes(verts=[verts], faces=[faces])
    
    points = sample_points_on_meshes(mesh, num_points=args.num_points)
    pointcloud = points_to_pointcloud(points).to(device)

    render_360_view(
        pointcloud,
        image_size=args.image_size,
        number_of_frames=args.frames,
        fps=args.fps,
        output_file=args.output_file
    )