import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
from diff_gaussian_rasterization import \
    GaussianRasterizationSettings as GaussianCamera
from packaging import version
from pytorch_msssim import ms_ssim
from skimage import filters
from skimage.color import rgb2gray
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def normalize_3d_coordinate(p, bound):
    """Copy from nice-slam, licensed under the Apache License, Version 2.0.

    Normalize 3d coordinate to [-1, 1] range.

    Args:
        p: (N, 3) 3d coordinate
        bound: (3, 2) min and max of each dimension
    Returns:
        (N, 3) normalized 3d coordinate
    """
    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0] - bound[0, 0]) / (bound[0, 1] - bound[0, 0])) * 2 - 1.0
    p[:, 1] = ((p[:, 1] - bound[1, 0]) / (bound[1, 1] - bound[1, 0])) * 2 - 1.0
    p[:, 2] = ((p[:, 2] - bound[2, 0]) / (bound[2, 1] - bound[2, 0])) * 2 - 1.0
    return p


def random_select(num, k):
    """Random select k values from 0..num."""
    return list(np.random.permutation(np.array(range(num)))[:min(num, k)])


def get_rays_from_uv(i, j, c2w, fx, fy, cx, cy, device):
    """Copy from nice-slam, licensed under the Apache License, Version 2.0.

    Get corresponding rays from input uv.
    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)
    dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)],
                       -1).to(device)
    dirs = dirs.reshape(-1, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o.to(device), rays_d.to(device)


def select_uv(i, j, n, depth, color, device='cuda:0'):
    """Copy from nice-slam, licensed under the Apache License, Version 2.0.

    Select n uv from dense uv.
    """
    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n, ), device=device)
    indices = indices.clamp(0, i.shape[0])
    i = i[indices]  # (n)
    j = j[indices]  # (n)
    depth = torch.from_numpy(depth).to(device).reshape(-1, 1)
    color = torch.from_numpy(color).to(device).reshape(-1, 3)
    depth = depth[indices]  # (n)
    color = color[indices]  # (n,3)
    return i, j, depth, color


def get_sample_uv_with_grad(H0, H1, W0, W1, n, image, ratio=15):
    """Copy from Point-slam, licensed under the Apache License, Version 2.0.

    Sample n uv coordinates from an image region H0..H1, W0..W1
    image (numpy.ndarray): color image or estimated normal image
    Args:
        H0 (int): top start point in pixels
        H1 (int): bottom edge end in pixels
        W0 (int): left start point in pixels
        W1 (int): right edge end in pixels
        n (int): number of samples
        image (tensor): color image
        ratio (int): sample from top ratio * n pixels within the region.
    """
    intensity = rgb2gray(image)
    grad_y = filters.sobel_h(intensity)
    grad_x = filters.sobel_v(intensity)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    img_size = (image.shape[0], image.shape[1])
    selected_index = np.argpartition(grad_mag, -ratio * n,
                                     axis=None)[-ratio * n:]
    indices_h, indices_w = np.unravel_index(selected_index, img_size)
    mask = (indices_h >= H0) & (indices_h < H1) & (indices_w >=
                                                   W0) & (indices_w < W1)
    indices_h, indices_w = indices_h[mask], indices_w[mask]
    selected_index = np.ravel_multi_index(np.array((indices_h, indices_w)),
                                          img_size)
    samples = np.random.choice(range(0, indices_h.shape[0]),
                               size=n,
                               replace=False)

    return selected_index[samples]


def get_sample_uv(H0, H1, W0, W1, n, depth, color, device='cuda:0'):
    """Copy from nice-slam, licensed under the Apache License, Version 2.0.

    Sample n uv coordinates from an image region H0..H1, W0..W1.
    """
    depth = depth[H0:H1, W0:W1]
    color = color[H0:H1, W0:W1]
    i, j = torch.meshgrid(torch.linspace(W0, W1 - 1, W1 - W0).to(device),
                          torch.linspace(H0, H1 - 1, H1 - H0).to(device),
                          indexing='ij')
    i = i.t()  # transpose
    j = j.t()
    i, j, depth, color = select_uv(i, j, n, depth, color, device=device)
    return i, j, depth, color


def get_selected_index_with_grad(H0,
                                 H1,
                                 W0,
                                 W1,
                                 n,
                                 image,
                                 ratio=15,
                                 gt_depth=None,
                                 depth_limit=False):
    """Copy from Point-slam, licensed under the Apache License, Version 2.0.

    return uv coordinates with top color gradient from an image region
    H0..H1, W0..W1.

    Args:
        H0 (int): top start point in pixels
        H1 (int): bottom edge end in pixels
        W0 (int): left start point in pixels
        W1 (int): right edge end in pixels
        n (int): number of samples
        image (tensor): color image
        ratio (int): sample from top ratio * n pixels within the region.
        This should ideally be dependent on the image size in percentage.
        gt_depth (tensor): depth input, will be passed if using
        self.depth_limit
        depth_limit (bool): if True, limits samples where the gt_depth
        is smaller than 5 m
    Returns:
        selected_index (ndarray): index of top color gradient uv coordinates
    """
    intensity = rgb2gray(image.cpu().numpy())
    grad_y = filters.sobel_h(intensity)
    grad_x = filters.sobel_v(intensity)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # random sample from top ratio*n elements within the region
    img_size = (image.shape[0], image.shape[1])
    # try skip the top color grad. pixels
    selected_index = np.argpartition(grad_mag, -ratio * n,
                                     axis=None)[-ratio * n:]
    indices_h, indices_w = np.unravel_index(selected_index, img_size)
    mask = (indices_h >= H0) & (indices_h < H1) & (indices_w >=
                                                   W0) & (indices_w < W1)
    if gt_depth is not None:
        if depth_limit:
            mask_depth = torch.logical_and(
                (gt_depth[torch.from_numpy(indices_h).to(image.device),
                          torch.from_numpy(indices_w).to(image.device)] <=
                 5.0),
                (gt_depth[torch.from_numpy(indices_h).to(image.device),
                          torch.from_numpy(indices_w).to(image.device)] > 0.0))
        else:
            mask_depth = gt_depth[
                torch.from_numpy(indices_h).to(image.device),
                torch.from_numpy(indices_w).to(image.device)] > 0.0
        mask = mask & mask_depth.cpu().numpy()
    indices_h, indices_w = indices_h[mask], indices_w[mask]
    selected_index = np.ravel_multi_index(np.array((indices_h, indices_w)),
                                          img_size)

    return selected_index, grad_mag


def get_samples(camera,
                n,
                c2w,
                depth,
                color,
                device,
                Hedge=0,
                Wedge=0,
                depth_filter=False,
                return_index=False,
                depth_limit=None):
    """Copy from nice-slam, licensed under the Apache License, Version 2.0.

    Get n rays from the image region 0..H, 0..W.

    c2w is its camera pose and depth/color is the corresponding image tensor.
    """
    i, j, sample_depth, sample_color = get_sample_uv(Hedge,
                                                     camera.height - Hedge,
                                                     Wedge,
                                                     camera.width - Wedge,
                                                     n,
                                                     depth,
                                                     color,
                                                     device=device)
    rays_o, rays_d = get_rays_from_uv(i, j, c2w.to(device), camera.fx,
                                      camera.fy, camera.cx, camera.cy, device)
    if depth_filter:
        sample_depth = sample_depth.reshape(-1)
        mask = sample_depth > 0
        if depth_limit is not None:
            mask = mask & (sample_depth < depth_limit)
        rays_o, rays_d, sample_depth, sample_color = rays_o[mask], rays_d[
            mask], sample_depth[mask], sample_color[mask]
        i, j = i[mask], j[mask]

    if return_index:
        return rays_o, rays_d, sample_depth, sample_color, i.to(
            torch.int64), j.to(torch.int64)
    return rays_o, rays_d, sample_depth, sample_color


def get_samples_with_pixel_grad(camera,
                                n_color,
                                c2w,
                                depth,
                                color,
                                device,
                                Hedge=0,
                                Wedge=0,
                                depth_filter=True,
                                return_index=True,
                                depth_limit=None):
    """Copy from Point-slam, licensed under the Apache License, Version 2.0.

    Get n rays from the image region H0..H1, W0..W1 based on color
    gradients, normal map gradients and random selection H, W: height, width.

    fx, fy, cx, cy: intrinsics. c2w is its camera pose and depth/color is the
    corresponding image tensor.
    """
    H, W, fx, fy, cx, cy = (camera.height, camera.width, camera.fx, camera.fy,
                            camera.cx, camera.cy)

    assert (n_color > 0), 'invalid number of rays to sample.'

    index_color_grad, index_normal_grad = [], []
    if n_color > 0:
        index_color_grad = get_sample_uv_with_grad(Hedge,
                                                   camera.height - Hedge,
                                                   Wedge, camera.width - Wedge,
                                                   n_color, color)

    merged_indices = np.union1d(index_color_grad, index_normal_grad)
    i, j = np.unravel_index(merged_indices.astype(int), (H, W))
    i, j = torch.from_numpy(j).to(device).float(), torch.from_numpy(i).to(
        device).float()  # (i-cx), on column axis
    rays_o, rays_d = get_rays_from_uv(i, j, c2w.to(device), fx, fy, cx, cy,
                                      device)
    i, j = i.long(), j.long()

    depth = torch.from_numpy(depth).to(device)
    color = torch.from_numpy(color).to(device)
    sample_depth = depth[j, i].reshape(-1)
    sample_color = color[j, i].reshape(-1, 3)

    if depth_filter:
        mask = sample_depth > 0
        if depth_limit is not None:
            mask = mask & (sample_depth < depth_limit)
        rays_o, rays_d, sample_depth, sample_color = rays_o[mask], rays_d[
            mask], sample_depth[mask], sample_color[mask]
        i, j = i[mask], j[mask]

    if return_index:
        return rays_o, rays_d, sample_depth, sample_color, i.to(
            torch.int64), j.to(torch.int64)
    return rays_o, rays_d, sample_depth, sample_color


def get_rays(camera, c2w, device):
    """Copy from nice-slam, licensed under the Apache License, Version 2.0.

    Get rays for a whole image.
    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, camera.width - 1, camera.width),
                          torch.linspace(0, camera.height - 1, camera.height),
                          indexing='ij')
    i = i.t()  # transpose
    j = j.t()
    dirs = torch.stack([(i - camera.cx) / camera.fx,
                        -(j - camera.cy) / camera.fy, -torch.ones_like(i)],
                       -1).to(device)
    dirs = dirs.reshape(camera.height, camera.width, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d


def get_pointcloud(depth, camera, c2w, sampled_indices):
    """This function is modified from splaTAM, licensed under the BSD 3-Clause
    License."""
    fx, fy, cx, cy = (camera.fx, camera.fy, camera.cx, camera.cy)

    # Compute indices of sampled pixels
    xx = (sampled_indices[:, 1] - cx) / fx
    yy = (sampled_indices[:, 0] - cy) / fy
    depth_z = depth[sampled_indices[:, 0], sampled_indices[:, 1]]

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    pts4 = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1)
    pts = (c2w @ pts4.T).T[:, :3]

    # Remove points at camera origin
    A = torch.abs(torch.round(pts, decimals=4))
    B = torch.zeros((1, 3)).cuda().float()
    _, idx, counts = torch.cat([A, B], dim=0).unique(dim=0,
                                                     return_inverse=True,
                                                     return_counts=True)
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    invalid_pt_idx = mask[:len(A)]
    valid_pt_idx = ~invalid_pt_idx
    pts = pts[valid_pt_idx]

    return pts


@torch.no_grad()
def keyframe_selection_overlap(camera,
                               cur_frame,
                               keyframes_graph,
                               k,
                               N_samples=16,
                               pixs_per_image=100,
                               use_ray_sample=True,
                               device='cuda:0'):
    H, W, fx, fy, cx, cy = (camera.height, camera.width, camera.fx, camera.fy,
                            camera.cx, camera.cy)
    if use_ray_sample:
        rays_o, rays_d, gt_depth, _ = get_samples(camera=camera,
                                                  n=pixs_per_image,
                                                  c2w=cur_frame.get_pose(),
                                                  depth=cur_frame.depth,
                                                  color=cur_frame.rgb,
                                                  device=device,
                                                  depth_filter=True)
        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, N_samples)
        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
        near = gt_depth * 0.8
        far = gt_depth + 0.5
        z_vals = near * (1. - t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[
            ..., :, None]  # [N_rays, N_samples, 3]
    else:
        # Radomly Sample Pixel Indices from valid depth pixels
        pixs_per_image = pixs_per_image * N_samples
        gt_depth = torch.tensor(cur_frame.depth).to(device)
        valid_depth_indices = torch.where(gt_depth > 0)
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        indices = torch.randint(valid_depth_indices.shape[0],
                                (pixs_per_image, ))
        sampled_indices = valid_depth_indices[indices]
        # Back Project the selected pixels to 3D Pointcloud
        pts = get_pointcloud(
            depth=gt_depth,  # [H,W]
            camera=camera,
            c2w=cur_frame.get_pose().to(device),
            sampled_indices=sampled_indices)
    vertices = pts.reshape(-1, 3).detach().cpu().numpy()
    list_keyframe = []
    for keyframe in keyframes_graph:
        c2w = keyframe.get_pose().detach().cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate([vertices, ones],
                                       axis=1).reshape(-1, 4, 1)  # (N, 4)
        cam_cord_homo = w2c @ homo_vertices  # (N, 4, 1)=(4,4)*(N, 4, 1)
        cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
        K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
        if use_ray_sample:
            # flip the x-axis such that the pixel space is u from the left
            # to right, v top to bottom. without the flipping of the x-axis,
            # the image is assumed to be flipped horizontally.
            cam_cord[:, 0] *= -1
        uv = K @ cam_cord
        z = uv[:, -1:] + 1e-5
        uv = uv[:, :2] / z
        uv = uv.astype(np.float32)
        edge = 20
        mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
            (uv[:, 1] < H-edge)*(uv[:, 1] > edge)
        if use_ray_sample:
            mask = mask & (z[:, :, 0] < 0)
        else:
            mask = mask & (z[:, :, 0] > 0)
        mask = mask.reshape(-1)
        percent_inside = mask.sum() / uv.shape[0]
        list_keyframe.append({
            'keyframe': keyframe,
            'percent_inside': percent_inside
        })
    list_keyframe = sorted(list_keyframe,
                           key=lambda i: i['percent_inside'],
                           reverse=True)
    selected_keyframe_list = [
        dic['keyframe'] for dic in list_keyframe
        if dic['percent_inside'] > 0.00
    ]
    selected_keyframe_list = list(
        np.random.permutation(np.array(selected_keyframe_list))[:k])
    return selected_keyframe_list


def save_render_imgs(idx, gt_color_np, gt_depth_np, color_np, depth_np,
                     img_save_dir):
    result_2d = None

    gt_color_np = np.clip(gt_color_np, 0, 1)
    if color_np is not None:
        color_np = np.clip(color_np, 0, 1)
        color_residual = np.abs(gt_color_np - color_np)
        color_residual[gt_depth_np == 0.0] = 0.0
        color_residual = np.clip(color_residual, 0, 1)

    if depth_np is not None:
        depth_residual = np.abs(gt_depth_np - depth_np)
        depth_residual[gt_depth_np == 0.0] = 0.0
        max_depth = np.max(gt_depth_np)

        gt_color = torch.tensor(gt_color_np)
        rcolor = torch.tensor(color_np)
    elif color_np is not None:
        depth_mask = (torch.from_numpy(gt_depth_np > 0).unsqueeze(-1)).float()
        gt_color = torch.tensor(gt_color_np) * depth_mask
        rcolor = torch.tensor(color_np) * depth_mask

    # 2d metrics
    # depth
    if depth_np is not None:
        gt_depth = torch.tensor(gt_depth_np)
        rdepth = torch.tensor(depth_np)
        depth_l1_render = torch.abs(gt_depth[gt_depth_np > 0] - rdepth[
            gt_depth_np > 0]).mean().item() * 100
    else:
        depth_l1_render = 0.0
    # rgb
    if color_np is not None:
        mse_loss = torch.nn.functional.mse_loss(gt_color, rcolor)
        psnr = -10. * torch.log10(mse_loss)
        ssim = ms_ssim(gt_color.transpose(0, 2).unsqueeze(0).float(),
                       rcolor.transpose(0, 2).unsqueeze(0).float(),
                       data_range=1.0,
                       size_average=True)
        cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex',
                                                          normalize=True)
        lpips = cal_lpips((gt_color).unsqueeze(0).permute(0, 3, 1, 2).float(),
                          (rcolor).unsqueeze(0).permute(0, 3, 1,
                                                        2).float()).item()
        text = (f'PSNR[dB]^: {psnr.item():.2f}, '
                f'SSIM^: {ssim:.2f}, '
                f'LPIPS: {lpips:.2f}, '
                f'Depth_L1[cm]: {depth_l1_render:.2f}')

        result_2d = psnr, ssim, lpips, depth_l1_render

    if depth_np is not None:
        fig, axs = plt.subplots(2, 3)
        fig.tight_layout()
        axs[0, 0].imshow(gt_depth_np, cmap='plasma', vmin=0, vmax=max_depth)
        axs[0, 0].set_title('Input Depth')
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].imshow(depth_np, cmap='plasma', vmin=0, vmax=max_depth)
        axs[0, 1].set_title('Generated Depth')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 2].imshow(depth_residual, cmap='plasma', vmin=0, vmax=max_depth)
        axs[0, 2].set_title('Depth Residual')
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])
        axs[1, 0].imshow(gt_color_np, cmap='plasma')
        axs[1, 0].set_title('Input RGB')
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(color_np, cmap='plasma')
        axs[1, 1].set_title('Generated RGB')
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        axs[1, 2].imshow(color_residual, cmap='plasma')
        axs[1, 2].set_title('RGB Residual')
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])
        fig.text(0.02,
                 0.02,
                 text,
                 ha='left',
                 va='bottom',
                 fontsize=12,
                 color='red')
    else:
        if gt_depth_np is None:
            fig, axs = plt.subplots(1, 1)
            axs.imshow(gt_color_np, cmap='plasma')
            axs.set_title('Input RGB')
            axs.set_xticks([])
            axs.set_yticks([])
        else:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(gt_color_np, cmap='plasma')
            axs[0].set_title('Input RGB')
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            max_depth = np.max(gt_depth_np)
            axs[1].imshow(gt_depth_np, cmap='plasma', vmin=0, vmax=max_depth)
            axs[1].set_title('Input Depth')
            axs[1].set_xticks([])
            axs[1].set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)
    if color_np is not None:
        plt.savefig(f'{img_save_dir}/{idx:05d}.jpg',
                    bbox_inches='tight',
                    pad_inches=0.2)
    plt.clf()
    plt.close()

    return result_2d


def rgbd2pcd(color_np, depth_np, c2w_np, camera, render_mode, device):
    """This function is modified from splaTAM, licensed under the BSD 3-Clause
    License."""
    color_np = np.clip(color_np, 0, 1.0)
    color = torch.from_numpy(color_np).to(device)  # render image [H, W, C]
    depth = torch.from_numpy(depth_np).to(device)  #
    c2w = torch.from_numpy(c2w_np).to(device)
    w2c = torch.inverse(c2w)

    width, height = camera.width, camera.height
    CX = camera.cx
    CY = camera.cy
    FX = camera.fx
    FY = camera.fy

    # Compute indices
    xx = torch.tile(torch.arange(width).cuda(), (height, ))
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z_depth = depth.reshape(-1)
    # Initialize point cloud
    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    w2c = w2c.clone().detach().cuda().float()
    c2w = torch.inverse(w2c)
    pts = (c2w @ pts4.T).T[:, :3]
    # Colorize point cloud
    if render_mode == 'depth':
        cols = z_depth
        bg_mask = (cols < 15).float()
        cols = cols * bg_mask
        colormap = plt.get_cmap('jet')
        cNorm = plt.Normalize(vmin=0, vmax=torch.max(cols))
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=colormap)
        cols = scalarMap.to_rgba(cols.contiguous().cpu().numpy())[:, :3]
        bg_mask = bg_mask.cpu().numpy()
        cols = cols * bg_mask[:, None] + (1 - bg_mask[:, None]) * np.array(
            [1.0, 1.0, 1.0])
        cols = torch.from_numpy(cols)
    else:
        cols = color.reshape(-1, 3)
    return pts.contiguous().double().cpu().numpy(), cols.contiguous().double(
    ).cpu().numpy()


def setup_camera(camera, w2c, near=0.01, far=100):
    """This function is modified from splaTAM, licensed under the BSD 3-Clause
    License."""
    w, h, fx, fy, cx, cy = (camera.width, camera.height, camera.fx, camera.fy,
                            camera.cx, camera.cy)
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor(
        [[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
         [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
         [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
         [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = GaussianCamera(image_height=h,
                         image_width=w,
                         tanfovx=w / (2 * fx),
                         tanfovy=h / (2 * fy),
                         bg=torch.tensor([0, 0, 0],
                                         dtype=torch.float32,
                                         device='cuda'),
                         scale_modifier=1.0,
                         viewmatrix=w2c,
                         projmatrix=full_proj,
                         sh_degree=0,
                         campos=cam_center,
                         prefiltered=False)
    return cam


def get_mesh_from_RGBD(camera, keyframe_graph, scale=1):
    """Modified from point-slam, licensed under the Apache License, Version
    2.0."""
    H, W, fx, fy, cx, cy = (camera.height, camera.width, camera.fx, camera.fy,
                            camera.cx, camera.cy)
    if version.parse(o3d.__version__) >= version.parse('0.13.0'):
        # for new version as provided in environment.yaml
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 * scale / 512.0,
            sdf_trunc=0.04 * scale,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    else:
        # for lower version
        volume = o3d.integration.ScalableTSDFVolume(
            voxel_length=4.0 * scale / 512.0,
            sdf_trunc=0.04 * scale,
            color_type=o3d.integration.TSDFVolumeColorType.RGB8)

    # address the misalignment in open3d marching cubes
    compensate_vector = (-0.0 * scale / 512.0, 2.5 * scale / 512.0,
                         -2.5 * scale / 512.0)
    for keyframe in keyframe_graph:
        c2w = keyframe.get_pose().cpu().numpy()
        # convert to open3d camera pose
        c2w[:3, 1] *= -1.0
        c2w[:3, 2] *= -1.0
        w2c = np.linalg.inv(c2w)
        depth = keyframe.depth
        color = keyframe.rgb
        depth = o3d.geometry.Image(depth.astype(np.float32))
        color = o3d.geometry.Image(np.array((color * 255).astype(np.uint8)))
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1,
            depth_trunc=1000,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, w2c)
    o3d_mesh = volume.extract_triangle_mesh()
    o3d_mesh = o3d_mesh.translate(compensate_vector)
    return o3d_mesh


def clean_mesh(mesh):
    """Modified from point-slam, licensed under the Apache License, Version
    2.0."""
    mesh_tri = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                               faces=np.asarray(mesh.triangles),
                               vertex_colors=np.asarray(mesh.vertex_colors))
    components = trimesh.graph.connected_components(
        edges=mesh_tri.edges_sorted)

    min_len = 100
    components_to_keep = [c for c in components if len(c) >= min_len]

    new_vertices = []
    new_faces = []
    new_colors = []
    vertex_count = 0
    for component in components_to_keep:
        vertices = mesh_tri.vertices[component]
        colors = mesh_tri.visual.vertex_colors[component]

        # Create a mapping from old vertex indices to new vertex indices
        index_mapping = {
            old_idx: vertex_count + new_idx
            for new_idx, old_idx in enumerate(component)
        }
        vertex_count += len(vertices)

        # Select faces that are part of the current connected component
        # and update vertex indices

        faces_in_component = mesh_tri.faces[np.any(np.isin(
            mesh_tri.faces, component),
                                                   axis=1)]
        reindexed_faces = np.vectorize(index_mapping.get)(faces_in_component)

        new_vertices.extend(vertices)
        new_faces.extend(reindexed_faces)
        new_colors.extend(colors)

    cleaned_mesh_tri = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    cleaned_mesh_tri.visual.vertex_colors = np.array(new_colors)

    cleaned_mesh_tri.remove_degenerate_faces()
    cleaned_mesh_tri.remove_duplicate_faces()

    return cleaned_mesh_tri


def cull_mesh(dataset,
              mesh,
              estimate_c2w_list=None,
              eval_rec=False,
              truncation=0.06,
              device='cuda:0'):
    """Modified from co-slam, licensed under the Apache License, Version
    2.0."""

    if estimate_c2w_list is not None:
        n_imgs = len(estimate_c2w_list)
    else:
        n_imgs = len(dataset)

    pc = mesh.vertices
    whole_mask = np.ones(pc.shape[0]).astype('bool')

    camera = dataset.get_camera()
    H, W, fx, fy, cx, cy = (camera.height, camera.width, camera.fx, camera.fy,
                            camera.cx, camera.cy)

    for i in range(0, n_imgs, 1):
        _, _, depth, c2w = dataset[i]
        depth, c2w = depth.to(device), c2w.to(device)

        if estimate_c2w_list is not None:
            c2w = estimate_c2w_list[i].to(device)

        points = pc.copy()
        points = torch.from_numpy(points).to(device)

        w2c = torch.inverse(c2w)
        K = torch.from_numpy(
            np.array([[fx, .0, cx], [.0, fy, cy],
                      [.0, .0, 1.0]]).reshape(3, 3)).to(device)
        ones = torch.ones_like(points[:, 0]).reshape(-1, 1).to(device)
        homo_points = torch.cat([points, ones],
                                dim=1).reshape(-1, 4, 1).to(device).float()
        cam_cord_homo = w2c @ homo_points
        cam_cord = cam_cord_homo[:, :3]

        cam_cord[:, 0] *= -1
        uv = K.float() @ cam_cord.float()
        z = uv[:, -1:] + 1e-5
        uv = uv[:, :2] / z
        uv = uv.squeeze(-1)

        grid = uv[None, None].clone()
        grid[..., 0] = grid[..., 0] / W
        grid[..., 1] = grid[..., 1] / H
        grid = 2 * grid - 1
        depth_samples = F.grid_sample(depth[None, None],
                                      grid,
                                      padding_mode='zeros',
                                      align_corners=True).squeeze()

        edge = 0
        if eval_rec:
            mask = (depth_samples + truncation >= -z[:, 0, 0]) & (
                0 <= -z[:, 0, 0]) & (uv[:, 0] < W - edge) & (
                    uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] >
                                                                edge)
        else:
            mask = (0 <= -z[:, 0, 0]) & (uv[:, 0] < W - edge) & (
                uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)

        mask = mask.cpu().numpy()

        whole_mask &= ~mask

    face_mask = whole_mask[mesh.faces].all(axis=1)
    mesh.update_faces(~face_mask)
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=False)

    return mesh
