import ops
import cv2 as cv
import torch
import numpy as np
import os
import math


def CalAxisNoise(depth, normal, intrinsics):
    print(depth.shape)
    h, w = depth.shape
    noisy_depth_map = np.copy(depth)
    noisy_depth_map = noisy_depth_map.astype(np.float64)
    ray = np.tile(np.array([0, 0, 1], dtype=np.float32), (h, w, 1))
    ray1 = torch.tensor(ray.astype(np.float32))
    ray1 = ray1.unsqueeze(0).cuda()
    ray = ops.normalize(ray1)
    ray = ray[0].cpu().detach().numpy().astype(np.float32)
    theta = np.arccos(np.sum(ray * normal, axis=2))
    theta[np.where(noisy_depth_map == 0)] = 0.0
    #theta = np.clip(theta, 0.0, 1.5)   ##限制角度
    theta = np.clip(theta, 0.0, 1.3)
    D = (noisy_depth_map / 1000.0).astype(np.float32)
    deviation = 1.5 - 0.5 * D + 0.3 * D ** 2 + 0.1 * D ** 1.5 * (theta ** 2 / (1.58 - theta) ** 2)
    deviation[np.where(depth == 0)] = 0.0
    noise = np.random.normal(0, 1.0, h * w).reshape([h, w])
    noisy_depth_map = noisy_depth_map + noise * deviation

    return noisy_depth_map


W = 512
H = 424
cx = 256.512
cy = 207.939
fx = 364.276
fy = 364.276
Z = 1800.0
left = -Z * cx / fx  # Z*W/fx
right = Z * (W - cx) / fx  #
top = Z * cy / fy  # Z*H/fy
bottom = -Z * (H - cy) / fy  #
print(left, right, top, bottom)
dx = np.abs(left) / cx
dy = top / cy
#dx = 2.0
#dy = 2.0

intrinsics = [fx, fy, cx, cy]
depthdir = 'data/DZ0'
savedir = 'data/depth'

if not os.path.isdir(savedir):
    os.mkdir(savedir)
for filename in os.listdir(depthdir):
    print(depthdir + '/' + filename)
    depth = cv.imread(depthdir + '/' + filename, -1)
    m = depth > 0
    mask = np.zeros_like(depth).astype(np.uint8)
    mask[np.where(depth > 0)] = 255
    depth1 = torch.tensor(depth.astype(np.float32))
    depth1 = depth1.unsqueeze(-1)
    depth1 = depth1.unsqueeze(0)
    depth1 = depth1.permute(0, 3, 1, 2).cuda()
    normal = ops.depth2normal_ortho(depth1, dx, dy)
    normal = normal[0].cpu().detach().numpy().astype(np.float32)
    for times in range(1):
        axisnoise = CalAxisNoise(depth, normal, intrinsics)
        newmask = np.copy(mask)

        lateralnoise = np.zeros_like(depth)
        for k in range(1):
            contours, _ = cv.findContours(newmask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
            edges = np.empty((0, contours[0].shape[1], contours[0].shape[2]), dtype=np.int32)
            for cidx in range(len(contours)):
                edges = np.vstack((edges, contours[cidx]))
            sample = np.random.randint(low=0, high=edges.shape[0], size=int(0.1 * edges.shape[0]))

            for idx in range(sample.shape[0]):
                x = edges[sample[idx], 0, 0]
                y = edges[sample[idx], 0, 1]
                t = 1

                if y >= t and y < newmask.shape[0] - t and x >= t and x < newmask.shape[1] - t:
                    index = tuple(([y - t, y, y, y, y + t], [x, x - t, x, x + t, x]))
                    if idx % 2 == 0:
                        newmask[index] = 0
                        lateralnoise[index] = 0
                    else:
                        newmask[index] = 255
                        lateralnoise[index] = axisnoise[y, x]

            vis_contours, _ = cv.findContours(newmask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        lateralnoise[np.where(depth > 0)] = 0
        noisedD = (axisnoise + lateralnoise).astype(np.uint16) * m.astype(np.uint16)
        cv.imwrite(savedir + '/' + filename, noisedD)
