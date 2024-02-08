# 4. Camera Transformations (10 points)

import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer


def render_cow(
    cow_path="data/cow_with_axis.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    """
    Render a cow mesh with a given relative transformation.

    Args:
        cow_path (str): Path to the cow mesh.
        image_size (int): Size of the output images.
        R_relative (list): 3x3 rotation matrix.
        T_relative (list): 3D translation vector.
        device (str): Device to use (if None, use GPU if available).

    Returns:
        np.array: Rendered image.
    """

    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)

    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
    renderer = get_mesh_renderer(image_size=image_size)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.t().unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow_with_axis.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="output/transform.jpg")
    args = parser.parse_args()

    R_relative_0=[[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]
    R_relative_1=[[0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]]
    R_relative_2=[[0, 0, -1],
                [0, 1, 0],
                [1, 0, 0]]
    T_relative_0=[0, 0, 0]
    T_relative_1=[0, 0, 2]
    T_relative_2=[0.5, -0.5, 0]
    T_relative_3=[3, 0, 3]

    transforms = [(R_relative_1, T_relative_0),
                  (R_relative_2, T_relative_3),
                  (R_relative_0, T_relative_1),
                  (R_relative_0, T_relative_2)]
    
    for i, (R, T) in enumerate(transforms):
        image_render = render_cow(
            cow_path=args.cow_path,
            image_size=args.image_size,
            R_relative=R,
            T_relative=T,
        )
        plt.imsave(args.output_path.replace(".jpg", f"{i+1}.jpg"), image_render)