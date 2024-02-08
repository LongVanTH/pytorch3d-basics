# 1.2 Re-creating the Dolly Zoom (10 points)

import argparse
import imageio
import numpy as np
import pytorch3d
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw

from starter.utils import get_device, get_mesh_renderer

def dolly_zoom(
    image_size=256,
    num_frames=100,
    fps=30,
    device=None,
    output_file="output/dolly.gif",
    look_from_behind="false",
):
    """
    Create a dolly zoom effect.
    
    Args:
        image_size (int): Size of the output images.
        num_frames (int): Number of frames in the gif.
        fps (int): Frames per second.
        device (str): Device to use (if None, use GPU if available).
        output_file (str): Path to the output gif file.
        look_from_behind (str): If "true", look from behind the object.
        
    Returns:
        None (saves a gif to output_file).
    """

    if device is None:
        device = get_device()

    mesh = pytorch3d.io.load_objs_as_meshes(["data/cow_on_plane.obj"])
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    fovs = torch.linspace(5, 120, num_frames)
    screen_width = 6

    # Loop instead of batch for simplicity (and to avoid memory issues)
    renders = []
    for fov in tqdm(fovs):
        distance = screen_width / (2 * np.tan(np.radians(fov) / 2))
        T = [[0, 0, distance]]

        if look_from_behind == "false":
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov, T=T, device=device)
        else:
            R, _ = pytorch3d.renderer.cameras.look_at_view_transform(distance, 0, 0)
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(
                R=R[0].t().unsqueeze(0), T=T, fov=fov, device=device
            )

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()
        renders.append(rend)
    
    """# Batch
    distances = screen_width / (2 * torch.tan(np.radians(fovs) / 2))
    Ts = torch.tensor([[0, 0, dist] for dist in distances], device=device)

    if look_from_behind == "false":
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            fov=fovs, T=Ts, device=device
        )
    else:
        Rs, _ = pytorch3d.renderer.cameras.look_at_view_transform(distances, 0, 0)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=Rs, T=Ts, fov=fovs, device=device
        )

    rend = renderer(mesh.extend(num_frames), cameras=cameras, lights=lights)
    renders = rend[..., :3].cpu().numpy()"""

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"fov: {fovs[i]:.2f}", fill=(255, 0, 0))
        images.append(np.array(image))

    imageio.mimwrite(output_file, images, fps=fps, loop=1000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output_file", type=str, default="output/dolly.gif")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--look_from_behind", type=str, default="false", choices=["true", "false"], help="If true, look from behind the object.")
    args = parser.parse_args()

    dolly_zoom(
        image_size=args.image_size,
        num_frames=args.num_frames,
        fps=args.fps,
        output_file=args.output_file,
        look_from_behind=args.look_from_behind,
    )

