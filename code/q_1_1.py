# 1.1. 360-degree Renders (5 points)

import argparse
import imageio
import torch
import pytorch3d
import numpy as np
from tqdm import tqdm

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh

def render_cow_360(
    cow_path="data/cow.obj",
    image_size=256,
    color=[0.7, 0.7, 1],
    device=None,
    number_of_frames=30,
    rotation_mode="linear",
    rotation_camera="true",
):
    """
    Render a 360-degree gif of a cow.

    Args:
        cow_path (str): Path to the cow mesh.
        image_size (int): Size of the output images.
        color (list): RGB color values.
        device (str): Device to use (if None, use GPU if available).
        number_of_frames (int): Number of frames in the gif.
        rotation_mode (str): Rotation mode, either "linear" or "ease-in-out".
        rotation_camera (bool): If True, rotate the camera. Otherwise, rotate the object.

    Returns:
        list: List of images representing the 360-degree view.
    """

    if device is None:
        device = get_device()

    # Renderer
    renderer = get_mesh_renderer(image_size=image_size)

    # Vertices, faces, and textures
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    
    # Lights
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    
    # Rotation
    if rotation_mode == "ease-in-out":
        theta_values = [360 * (t*t/(2*(t*t-t)+1)) + 180 for t in np.linspace(0, 1, number_of_frames, endpoint=False)]
    elif rotation_mode == "linear":
        theta_values = [360 * t + 180 for t in np.linspace(0, 1, number_of_frames, endpoint=False)]

    # Loop instead of batch (to avoid memory issues)
    images = []
    theta_values = np.array(theta_values)
    for i in tqdm(range(number_of_frames)):
        theta = theta_values[i]
        if rotation_camera == "true":
            R, T = pytorch3d.renderer.cameras.look_at_view_transform(3.0, 0, theta)
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(
                R=R[0].t().unsqueeze(0), T=T[0].unsqueeze(0),
                fov=60, device=device
            )
            mesh_r = mesh
        else:
            theta = -theta - 180
            relative_rotation = pytorch3d.transforms.euler_angles_to_matrix(torch.tensor([0, theta*np.pi/180, 0]), "XYZ")
            relative_rotation = relative_rotation.to(vertices.dtype)
            rotated_vertices = vertices @ relative_rotation
            mesh_r = pytorch3d.structures.Meshes(
                verts=rotated_vertices,
                faces=faces,
                textures=pytorch3d.renderer.TexturesVertex(textures),
            ).to(device)
            R = torch.eye(3, device=device).unsqueeze(0)
            T = torch.tensor([[0, 0, 3]], device=device)
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(
                R=R, T=T,
                device=device
            )
        rend = renderer(mesh_r, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]
        rend = (rend * 255).astype("uint8")  # for imageio.mimwrite
        images.append(rend)

    """# Batch (can cause memory issues)
    if rotation_camera == "true":
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=3,
            elev=0,
            azim=theta_values,
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T,
            device=device
        )
        mesh_r = mesh.extend(number_of_frames)
    else:
        theta_values = -np.array(theta_values) - 180
        relative_rotations = [
            pytorch3d.transforms.euler_angles_to_matrix(
                torch.tensor([0, theta * np.pi / 180, 0]), "XYZ"
            ).to(vertices.dtype)
            for theta in theta_values
        ]
        rotated_vertices = vertices @ torch.stack(relative_rotations)
        mesh_r = pytorch3d.structures.Meshes(
            verts=rotated_vertices,
            faces=faces.repeat(number_of_frames, 1, 1),
            textures=pytorch3d.renderer.TexturesVertex(textures).extend(number_of_frames),
        ).to(device)
        # Alternative implementation
        # mesh_r = []
        # theta_values = -np.array(theta_values) - 180
        # for theta in theta_values:
        #     relative_rotation = pytorch3d.transforms.euler_angles_to_matrix(torch.tensor([0, theta*np.pi/180, 0]), "XYZ")
        #     relative_rotation = relative_rotation.to(vertices.dtype)
        #     rotated_mesh = mesh.update_padded(vertices @ relative_rotation)
        #     mesh_r.append(rotated_mesh)
        # mesh_r = pytorch3d.structures.join_meshes_as_batch(mesh_r)
        R = torch.eye(3, device=device).unsqueeze(0).repeat(number_of_frames, 1, 1) # (num_views, 3, 3)
        T = torch.tensor([[0, 0, 3]], device=device).repeat(number_of_frames, 1) # (num_views, 3)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T,
            device=device
        )
        
    rend = renderer(mesh_r, cameras=cameras, lights=lights)
    renders = rend.cpu().numpy()[:, ..., :3]  # (num_views, H, W, 3)
    images = [(render * 255).astype("uint8") for render in renders]"""
    
    return images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="output/cow_360.gif")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=30)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--rotation_mode", type=str, default="linear", choices=["ease-in-out", "linear"])
    parser.add_argument("--rotation_camera", type=str, default="true", choices=["true", "false"], help="If true, rotate the camera. Otherwise, rotate the object.")
    args = parser.parse_args()

    my_images = render_cow_360(
        cow_path=args.cow_path,
        image_size=args.image_size,
        number_of_frames=args.num_frames,
        rotation_mode=args.rotation_mode,
        rotation_camera=args.rotation_camera
    )

    imageio.mimwrite(args.output_path, my_images, fps=args.fps, loop=1000)