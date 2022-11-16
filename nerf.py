import os
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange

device = "mps" if torch.backends.mps.is_available() else "cpu"

# lego builder...standard nerf dataset lol
# images are 100x100
# poses are 4x4 pose matrices. top left is 3x3 rotation matrix
# last col is x, y, z, 1 translation vector
data = np.load('tiny_nerf_data.npz')
images = data["images"].astype("float32")
poses = data["poses"].astype("float32")
focal = data["focal"].astype("float32")

dirs = np.stack([np.sum([0, 0, -1]*pose[:3, :3], axis=-1) for pose in poses])
origins = poses[:, :3, -1]
# plotting data code
""" ax = plt.figure(figsize=(12, 8)).add_subplot(projection="3d")
_ = ax.quiver(
    origins[..., 0].flatten(),
    origins[..., 1].flatten(),
    origins[..., 2].flatten(),
    dirs[..., 0].flatten(),
    dirs[..., 1].flatten(),
    dirs[..., 2].flatten(),
    length=0.5,
    normalize=True
)
plt.show()

plt.imshow(images[13])
plt.show()
"""

# for a given image, the camera center is the same, but the
# direction of ray is different --> Pinhole camera model


def get_rays(height, width, focal_length, c2w):
    # get origin and direction of rays for each pixel and camera origin

    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32).to(c2w),
        torch.arange(height, dtype=torch.float32).to(c2w),
        indexing='ij'
    )
    i, j = i.transpose(-1, -2), j.transpose(-1, -2)
    directions = torch.stack([(i-width * 0.5)/focal_length,
                              -(j-height*0.5)/focal_length,
                              -torch.ones_like(i)], dim=-1)

    # applying camera pose to directions
    rays_d = torch.sum(directions[..., None, :]*c2w[:3, :3], dim=-1)

    # optical center (origin) is same for all directions
    # keep in mind that camera is different from image! (seems obvious but is important!)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d


num_train = 100
testimg_idx = 101
testimg, testpose = images[testimg_idx], poses[testimg_idx]

images = torch.from_numpy(data["images"][:num_train]).to(device)
poses = torch.from_numpy(data["poses"]).to(device)
focal = torch.from_numpy(data["focal"]).float().to(device)
testimg = torch.from_numpy(data["images"][testimg_idx]).to(device)
testpose = torch.from_numpy(data["poses"][testimg_idx]).to(device)

height, width = images.shape[1:3]
with torch.no_grad():
    ray_origin, ray_direction = get_rays(height, width, focal, testpose)

print("ray origin")
print(ray_origin.shape)
print(ray_origin[height // 2, width // 2, :])
print("\n")

print('ray Direction')
print(ray_direction.shape)
print(ray_direction[height // 2, width // 2, :])
print('\n')
