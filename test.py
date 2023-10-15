import argparse
from torch.utils.data import DataLoader
import model
import torch
import numpy as np
import loader
import time
import os
import ops
import cv2
import trimesh
from loader import TrainData
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeldir", type=str, default="model")
    parser.add_argument("--savedir", type=str, default="results")
    parser.add_argument("--depth_net_name", type=str, default="F2F_Model.pth")
    parser.add_argument("--back_net_name", type=str, default="B2B_Model.pth")
    parser.add_argument("--dataset_dir", type=str, default="data/")           ##
    parser.add_argument("--index_file", type=str, default="test.csv")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--low_thres", type=float, default=500.0)
    parser.add_argument("--up_thres", type=float, default=3000.0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    batch_size = args.batch_size
    model_dir = args.modeldir
    save_dir = args.savedir
    low_thres = args.low_thres
    up_thres = args.up_thres
    dataset_dir = args.dataset_dir

    # input resolution
    W = 512
    H = 424
    # perspective intrinsics
    cx = 256.512
    cy = 207.939
    fx = 364.276
    fy = 364.276
    # Orthographic parameters
    Z = 1800
    left = -Z * cx / fx
    right = Z * (W - cx) / fx
    top = Z * cy / fy
    bottom = -Z * (H - cy) / fy
    dx = np.abs(left) / cx
    dy = top / cy

    # init for the face generation
    crop_w = 424
    crop_h = 424
    fp_idx = np.zeros([crop_h, crop_w], dtype=np.int64)
    bp_idx = np.ones_like(fp_idx) * (crop_h * crop_w)
    for hh in range(crop_h):
        for ww in range(crop_w):
            fp_idx[hh, ww] = hh * crop_w + ww
            bp_idx[hh, ww] += hh * crop_w + ww

    # init X, Y coordinate tensors
    Y, X = torch.meshgrid(torch.tensor(range(crop_h)), torch.tensor(range(crop_w)))
    X = X.unsqueeze(0).unsqueeze(0).float().cuda()  # (B,H,W)
    Y = Y.unsqueeze(0).unsqueeze(0).float().cuda()
    x_cord = X * dx
    y_cord = Y * dy

    with torch.no_grad():
        # load pretrained models
        ngf = 32
        # load params
        F2F_D = model.AttentionNet(in_channel1=4, out_channel=1, ngf=ngf, num_heads=8, norm=True).cuda()
        state_dict = torch.load(os.path.join(model_dir, args.depth_net_name),map_location='cuda:0')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v

        F2F_D.load_state_dict(new_state_dict)

        ##
        B2B_D = model.UNet(in_channel=4, out_channel=1, ngf=ngf, upconv=False, norm=True).cuda()
        state_dict = torch.load(os.path.join(model_dir, args.back_net_name), map_location='cuda:0')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        B2B_D.load_state_dict(new_state_dict)
        ##
        F2F_D = F2F_D.eval()
        B2B_D = B2B_D.eval()

        start_time = time.time()
        t = time.time()
        test_data = TrainData(dataset_dir, train=False, transform=torch.tensor)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0,
                                 drop_last=True)
        ##
        i = 0
        ##
        # laod data
        for data in test_loader:
            # YOU NEED TO APPLY MASK FOR THE COLOR AND DEPTH DATA BEFORE INPUT
            color_f, depth_f,depth_b_smpl = data
            color_f = color_f.permute(0, 3, 1, 2).cuda()
            color_f = ops.crop_tensor(color_f, crop_h, crop_w)
            depth_f = depth_f.unsqueeze(-1)
            depth_f = depth_f.permute(0, 3, 1, 2).cuda()
            depth_f = ops.crop_tensor(depth_f, crop_h, crop_w)
            depth_b_smpl = depth_b_smpl.unsqueeze(-1)
            depth_b_smpl = depth_b_smpl.permute(0, 3, 1, 2).cuda()
            depth_b_smpl = torch.flip(depth_b_smpl, [3])
            depth_b_smpl = ops.crop_tensor(depth_b_smpl, crop_h, crop_w)
            ##
            # optional for kinect 2 data (bad mask)
            # color_batch = ops.erode(color_batch, 2)
            # color_batch = ops.dilate(color_batch, 3)

            mask_batch = (depth_f > 0.0).float()
            mask_batch_smpl = (depth_b_smpl > 0.0).float()
            ##
            ##
            depth_b_smpl = 2 * Z - depth_b_smpl
            depth_b_smpl = depth_b_smpl * mask_batch_smpl

            f_alb = ops.convert_color_to_m1_1(color_f).clamp(-1.0, 1.0)

            # fix the averange depth to 1750 mm (for better results)
            fix_p = 1750
            tmp1 = fix_p - torch.sum(depth_f) / torch.sum(mask_batch)
            depth_f = depth_f + tmp1
            depth_f = depth_f * mask_batch
            ##
            tmp2 = fix_p - torch.sum(depth_b_smpl) / torch.sum(mask_batch_smpl)
            depth_b_smpl = depth_b_smpl + tmp2
            depth_b_smpl = depth_b_smpl * mask_batch_smpl
            ##
            f_depth = ops.convert_depth_to_m1_1(depth_f, low_thres, up_thres).clamp(-1.0, 1.0)
            b_depth_smpl = ops.convert_depth_to_m1_1(depth_b_smpl, low_thres, up_thres).clamp(-1.0, 1.0)
            # front depth net
            dz0 = F2F_D(torch.cat((f_depth,f_alb), dim=1), inter_mode='bilinear')
            dz0_batch = ops.convert_depth_back_from_m1_1(dz0, low_thres, up_thres) * mask_batch
            #fix_p = 1755
            #tmp1 = fix_p - torch.sum(dz0_batch) / torch.sum(mask_batch)
            #dz0_batch = dz0_batch + tmp1
            #dz0_batch = dz0_batch * mask_batch
            # back depth net
            dz1 = B2B_D(torch.cat((b_depth_smpl,f_alb), dim=1), inter_mode='bilinear')
            dz1_batch = ops.convert_depth_back_from_m1_1(dz1, low_thres, up_thres) * mask_batch

            network_time = time.time() - t
            ##print(depth_f.shape)
            # convert the images to 3D mesh
            fpct = torch.cat((x_cord, y_cord, dz0_batch), dim=1)
            bpct = torch.cat((x_cord, y_cord, dz1_batch), dim=1)
            ##print(fpct.shape)  ##(1,3,424,424)
            # dilate for the edge point interpolation
            fpct = ops.dilate(fpct, 1)
            bpct = ops.dilate(bpct, 1)
            fpc = fpct[0].permute(1, 2, 0).detach().cpu().numpy()
            bpc = bpct[0].permute(1, 2, 0).detach().cpu().numpy()
            ##########################
            ops.remove_points(fpc, bpc)
            # get the edge region for the edge point interpolation
            mask_pc = fpc[:, :, 2] > low_thres
            mask_pc = mask_pc.astype(np.float32)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            eroded = cv2.erode(mask_pc, kernel)
            edge = (mask_pc - eroded).astype(bool)
            # interpolate 2 points for each edge point pairs
            fpc[edge, 2:3] = (fpc[edge, 2:3] * 2 + bpc[edge, 2:3] * 1) / 3
            bpc[edge, 2:3] = (fpc[edge, 2:3] * 1 + bpc[edge, 2:3] * 2) / 3
            fpc = fpc.reshape(-1, 3)
            bpc = bpc.reshape(-1, 3)
            if (np.sum(mask_pc) < 100):
                print('noimage')
                continue
            f_faces = ops.getfrontFaces(mask_pc, fp_idx)
            b_faces = ops.getbackFaces(mask_pc, bp_idx)
            edge_faces = ops.getEdgeFaces(mask_pc, fp_idx, bp_idx)
            faces = np.vstack((f_faces, b_faces, edge_faces))
            s = fpc.shape
            points = np.concatenate((fpc, bpc), axis=0)
            # reset center point and convert mm to m
            points[:, 0:3] = -(points[:, 0:3] - np.array([[crop_w / 2 * dx, crop_h / 2 * dy, Z]])) / 1000.0
            points[:, 0] = -points[:, 0]
            points[:, 1] = points[:, 1]
            vertices = points[:, 0:3]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            # mkdirs
            filename = str(i)
            output_obj_name = os.path.join(save_dir, 'obj', filename + '.obj')
            print('Output_file:', output_obj_name)
            if not os.path.isdir(os.path.join(save_dir, 'obj')):
                os.mkdir(os.path.join(save_dir, 'obj'))
            if not os.path.isdir(os.path.join(save_dir, 'df')):
                os.mkdir(os.path.join(save_dir, 'df'))
            if not os.path.isdir(os.path.join(save_dir, 'db')):
                os.mkdir(os.path.join(save_dir, 'db'))
            f_depth_path = os.path.join(save_dir, 'df')
            b_depth_path = os.path.join(save_dir, 'db')
            f_depth_batch = dz0_batch.permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint16)
            b_depth_batch = dz1_batch.permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint16)
            f_depth_batch = f_depth_batch[0]
            b_depth_batch = b_depth_batch[0]
            cv2.imwrite(os.path.join(f_depth_path, str(i) + "f_depth.png"), f_depth_batch.astype(np.uint16))
            cv2.imwrite(os.path.join(b_depth_path, str(i) + "b_depth.png"), b_depth_batch.astype(np.uint16))
            mesh.export(output_obj_name)  # save obj
            i = i + 1
