# 2.1 Constructing a Tetrahedron (5 points)
# 2.2 Constructing a Cube (5 points)

import argparse
import imageio
import numpy as np
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer


def create_tetrahedron_mesh():
    # Define vertices of the tetrahedron
    vertices = torch.tensor([
        [1, 1, 1],  # Vertex 0
        [-1, -1, 1],  # Vertex 1
        [-1, 1, -1],  # Vertex 2
        [1, -1, -1],  # Vertex 3
    ], dtype=torch.float32)
    vertices = vertices.unsqueeze(0)

    # Define faces of the tetrahedron (triangle indices)
    faces = torch.tensor([
        [0, 1, 2],  # Face 0
        [0, 2, 3],  # Face 1
        [0, 3, 1],  # Face 2
        [1, 3, 2],  # Face 3
    ], dtype=torch.int64)
    faces = faces.unsqueeze(0)

    """# Create a single-color texture
    textures = torch.ones_like(vertices)
    color = [0.7, 0.7, 1]
    textures = textures * torch.tensor(color)

    # Create a multi-color texture
    textures = torch.tensor([
        [1, 0, 0],  # Vertex 0
        [0, 1, 0],  # Vertex 1
        [0, 0, 1],  # Vertex 2
        [1, 1, 0],  # Vertex 3
    ], dtype=torch.float32)
    textures = textures.unsqueeze(0)

    # Create a Meshes object
    tetrahedron_mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )"""

    # Define colors for each face
    colors_per_face = torch.tensor([
        [1.0, 0.0, 0.0],  # Red for Face 0
        [0.0, 1.0, 0.0],  # Green for Face 1
        [0.0, 0.0, 1.0],  # Blue for Face 2
        [1.0, 1.0, 0.0],  # Yellow for Face 3
    ], dtype=torch.float32)

    # Match the required shape (N,F,R,R,C)
    colors_per_face = colors_per_face.unsqueeze(0).unsqueeze(2).unsqueeze(2)

    # Create a TexturesAtlas object
    tetrahedron_mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesAtlas(colors_per_face),
    )

    return tetrahedron_mesh
    

def create_cube_mesh():
    # Define vertices of the cube
    vertices = torch.tensor([
        [-1, -1, -1],  # Vertex 0
        [-1, 1, -1],   # Vertex 1
        [1, 1, -1],    # Vertex 2
        [1, -1, -1],   # Vertex 3
        [-1, -1, 1],   # Vertex 4
        [-1, 1, 1],    # Vertex 5
        [1, 1, 1],     # Vertex 6
        [1, -1, 1],    # Vertex 7
    ], dtype=torch.float32)
    vertices = vertices.unsqueeze(0)

    # Define faces of the cube (two triangles per face)
    faces = torch.tensor([
        [0, 1, 2],  # Face 0 (front)
        [0, 2, 3],  # Face 0 (front)
        [4, 5, 6],  # Face 1 (back)
        [4, 6, 7],  # Face 1 (back)
        [0, 1, 5],  # Face 2 (left)
        [0, 5, 4],  # Face 2 (left)
        [3, 2, 6],  # Face 3 (right)
        [3, 6, 7],  # Face 3 (right)
        [0, 4, 7],  # Face 4 (bottom)
        [0, 7, 3],  # Face 4 (bottom)
        [1, 5, 6],  # Face 5 (top)
        [1, 6, 2],  # Face 5 (top)
    ], dtype=torch.int64)
    faces = faces.unsqueeze(0)

    """# Create a single-color texture
    textures = torch.ones_like(vertices)
    color = [0.7, 0.7, 1]
    textures = textures * torch.tensor(color)
    
    # Create a Meshes object
    cube_mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )"""

    # Define colors for each face
    colors_per_face = torch.tensor([
        [1.0, 0.0, 0.0],  # Red for Face 0
        [1.0, 0.0, 0.0],  # Red for Face 0
        [0.0, 1.0, 0.0],  # Green for Face 1
        [0.0, 1.0, 0.0],  # Green for Face 1
        [0.0, 0.0, 1.0],  # Blue for Face 2
        [0.0, 0.0, 1.0],  # Blue for Face 2
        [1.0, 1.0, 0.0],  # Yellow for Face 3
        [1.0, 1.0, 0.0],  # Yellow for Face 3
        [1.0, 0.0, 1.0],  # Magenta for Face 4
        [1.0, 0.0, 1.0],  # Magenta for Face 4
        [0.0, 1.0, 1.0],  # Cyan for Face 5
        [0.0, 1.0, 1.0],  # Cyan for Face 5
    ], dtype=torch.float32)

    # Match the required shape (N,F,R,R,C)
    colors_per_face = colors_per_face.unsqueeze(0).unsqueeze(2).unsqueeze(2)

    # Create a TexturesAtlas object
    cube_mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesAtlas(colors_per_face),
    )

    return cube_mesh


def figure_360(
    mesh,
    image_size=256,
    device=None,
    distance=5,
    number_of_frames=90,
    fps=30,
    rotation_mode="linear",
    output_file="output/mesh_360.gif",
):
    """
    Render a 360-degree gif of a mesh.

    Args:
        mesh (Meshes): Meshes object.
        image_size (int): Size of the output images.
        device (str): Device to use (if None, use GPU if available).
        distance (float): Distance from the object.
        number_of_frames (int): Number of frames in the gif.
        fps (int): Frames per second.
        rotation_mode (str): Rotation mode, either "linear" or "ease-in-out".
        output_file (str): Path to the output gif file.

    Returns:
        list: List of images representing the 360-degree view.
    """

    if device is None:
        device = get_device()

    # Renderer
    renderer = get_mesh_renderer(image_size=image_size)

    # Lights
    lights = pytorch3d.renderer.PointLights(location=[[0, 5, -10]], device=device)

    # Rotation
    if rotation_mode == "ease-in-out":
        theta_values = [360 * (t*t/(2*(t*t-t)+1)) + 180 for t in np.linspace(0, 1, number_of_frames, endpoint=False)]
    elif rotation_mode == "linear":
        theta_values = [360 * t + 180 for t in np.linspace(0, 1, number_of_frames, endpoint=False)]

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=distance,
        elev=0,
        azim=theta_values,
    )
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )

    mesh_r = mesh.extend(number_of_frames).to(device)
    
    rend = renderer(mesh_r,
                    cameras=cameras,
                    lights=lights
                    )
    renders = rend.cpu().numpy()[:, ..., :3]  # (num_views, H, W, 3)
    images = [(render * 255).astype("uint8") for render in renders]

    imageio.mimwrite(output_file, images, fps=fps, loop=1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, default="tetrahedron", choices=["tetrahedron", "cube"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--number_of_frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--distance", type=float, default=5)
    parser.add_argument("--rotation_mode", type=str, default="linear", choices=["ease-in-out", "linear"])
    parser.add_argument("--output_file", type=str, default="output/mesh_360.gif")
    args = parser.parse_args()

    if args.mesh == "tetrahedron":
        mesh = create_tetrahedron_mesh()
    elif args.mesh == "cube":
        mesh = create_cube_mesh()

    figure_360(
        mesh,
        image_size=args.image_size,
        distance=args.distance,
        number_of_frames=args.number_of_frames,
        fps=args.fps,
        rotation_mode=args.rotation_mode,
        output_file=args.output_file,
    )