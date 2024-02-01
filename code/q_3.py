# 3. Re-texturing a mesh (10 points)

import argparse
import torch
import pytorch3d
import imageio
import numpy as np
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def retexture_cow(
    cow_path="data/cow.obj",
    color1=[1, 1, 0],  # Front color
    color2=[1, 0, 1],  # Back color
    image_size=256,
    device=None,
    number_of_frames=30,
    fps = 15,
    output_file="output/cow_retextured.gif"
):
    """
    Re-texture a cow mesh and render a 360-degree gif.

    Args:
        cow_path (str): Path to the cow mesh.
        color1 (list): RGB color values for the front.
        color2 (list): RGB color values for the back.
        image_size (int): Size of the output images.
        device (str): Device to use (if None, use GPU if available).
        number_of_frames (int): Number of frames in the gif.
        fps (int): Frames per second.
        output_file (str): Path to the output gif file.

    Returns:
        None (saves a gif to output_file).
    """

    if device is None:
        device = get_device()

    # Load cow mesh
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    # Find z_min and z_max
    z_min = vertices[:, :, 2].min()
    z_max = vertices[:, :, 2].max()

    # Calculate color based on z-coordinate
    alpha = (vertices[:, :, 2] - z_min) / (z_max - z_min)
    colors = alpha.unsqueeze(-1) * torch.tensor(color2) + (1 - alpha.unsqueeze(-1)) * torch.tensor(color1)

    # Create TexturesVertex
    textures = pytorch3d.renderer.TexturesVertex(colors)

    # Create Meshes object
    cow_mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=textures
    )
    cow_mesh.to(device)

    renderer = get_mesh_renderer(image_size=image_size)

    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    theta_values = [360 * t + 180 for t in np.linspace(0, 1, number_of_frames, endpoint=False)]
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
    mesh_r = cow_mesh.extend(number_of_frames).to(device)

    rend = renderer(mesh_r,
                    cameras=cameras,
                    lights=lights
                    )
    renders = rend.cpu().numpy()[:, ..., :3]
    images = [(render * 255).astype("uint8") for render in renders]

    imageio.mimwrite(output_file, images, fps=fps, loop=1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_file", type=str, default="output/cow_retextured.gif")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=30)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    retexture_cow(
        cow_path=args.cow_path,
        image_size=args.image_size,
        number_of_frames=args.num_frames,
        fps=args.fps,
        output_file=args.output_file
    )