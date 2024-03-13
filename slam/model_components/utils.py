import cv2
import numpy as np
import torch
import torch.nn.functional as F


def mse2psnr(x):
    """MSE to PSNR."""
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x)


def coordinates(voxel_dim, device: torch.device, flatten=True):
    '''
    Params: voxel_dim: int or tuple of int
    Return: coordinates of the voxel grid
    '''
    if type(voxel_dim) is int:
        nx = ny = nz = voxel_dim
    else:
        nx, ny, nz = voxel_dim[0], voxel_dim[1], voxel_dim[2]
    x = torch.arange(0, nx, dtype=torch.long, device=device)
    y = torch.arange(0, ny, dtype=torch.long, device=device)
    z = torch.arange(0, nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z, indexing='ij')

    if not flatten:
        return torch.stack([x, y, z], dim=-1)

    return torch.stack((x.flatten(), y.flatten(), z.flatten()))


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    '''
    Params:
        bins: torch.Tensor, (Bs, N_samples)
        weights: torch.Tensor, (Bs, N_samples)
        N_importance: int
    Return:
        samples: torch.Tensor, (Bs, N_importance)
    '''
    # device = weights.get_device()
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # Bs, N_samples-2
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1], device=device), cdf],
                    -1)  # Bs, N_samples-1
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / N_importance,
                           1. - 0.5 / N_importance,
                           steps=N_importance,
                           device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=device),
                        denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def batchify(fn, chunk=1024 * 64):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs, inputs_dir=None):
        if inputs_dir is not None:
            return torch.cat([
                fn(inputs[i:i + chunk], inputs_dir[i:i + chunk])
                for i in range(0, inputs.shape[0], chunk)
            ], 0)
        return torch.cat([
            fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)
        ], 0)

    return ret


def get_masks(z_vals, target_d, truncation):
    '''
    Params:
        z_vals: torch.Tensor, (Bs, N_samples)
        target_d: torch.Tensor, (Bs,)
        truncation: float
    Return:
        front_mask: torch.Tensor, (Bs, N_samples)
        sdf_mask: torch.Tensor, (Bs, N_samples)
        fs_weight: float
        sdf_weight: float
    '''

    # before truncation
    front_mask = torch.where(z_vals < (target_d - truncation),
                             torch.ones_like(z_vals), torch.zeros_like(z_vals))
    # after truncation
    back_mask = torch.where(z_vals > (target_d + truncation),
                            torch.ones_like(z_vals), torch.zeros_like(z_vals))
    # valid mask
    depth_mask = torch.where(target_d > 0.0, torch.ones_like(target_d),
                             torch.zeros_like(target_d))
    # Valid sdf regionn
    sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

    num_fs_samples = torch.count_nonzero(front_mask)
    num_sdf_samples = torch.count_nonzero(sdf_mask)
    num_samples = num_sdf_samples + num_fs_samples
    fs_weight = 1.0 - num_fs_samples / num_samples
    sdf_weight = 1.0 - num_sdf_samples / num_samples

    return front_mask, sdf_mask, fs_weight, sdf_weight


def compute_loss(prediction, target, loss_type='l2'):
    '''
    Params:
        prediction: torch.Tensor, (Bs, N_samples)
        target: torch.Tensor, (Bs, N_samples)
        loss_type: str
    Return:
        loss: torch.Tensor, (1,)
    '''

    if loss_type == 'l2':
        return F.mse_loss(prediction, target)
    elif loss_type == 'l1':
        return F.l1_loss(prediction, target)

    raise Exception('Unsupported loss type')


def get_sdf_loss(z_vals,
                 target_d,
                 predicted_sdf,
                 truncation,
                 loss_type=None,
                 grad=None):
    '''
    Params:
        z_vals: torch.Tensor, (Bs, N_samples)
        target_d: torch.Tensor, (Bs,)
        predicted_sdf: torch.Tensor, (Bs, N_samples)
        truncation: float
    Return:
        fs_loss: torch.Tensor, (1,)
        sdf_loss: torch.Tensor, (1,)
        eikonal_loss: torch.Tensor, (1,)
    '''
    front_mask, sdf_mask, fs_weight, sdf_weight = get_masks(
        z_vals, target_d, truncation)

    fs_loss = compute_loss(predicted_sdf * front_mask,
                           torch.ones_like(predicted_sdf) * front_mask,
                           loss_type) * fs_weight
    sdf_loss = compute_loss((z_vals + predicted_sdf * truncation) * sdf_mask,
                            target_d * sdf_mask, loss_type) * sdf_weight

    if grad is not None:
        eikonal_loss = (((grad.norm(2, dim=-1) - 1)**2) * sdf_mask /
                        sdf_mask.sum()).sum()
        return fs_loss, sdf_loss, eikonal_loss

    return fs_loss, sdf_loss


def raw2outputs_nerf_color(raw,
                           z_vals,
                           rays_d,
                           occupancy=False,
                           device='cuda:0',
                           coef=10.0):
    """Transforms model's predictions to semantically meaningful values.

    Args:
        raw (tensor, N_rays*N_samples*4): prediction from model.
        z_vals (tensor, N_rays*N_samples): integration time.
        rays_d (tensor, N_rays*3): direction of each ray.
        occupancy (bool, optional): occupancy or volume density.
        device (str, optional): device. Defaults to 'cuda:0'.
        coef (float, optional): to multiply the input of sigmoid
        function when calculating occupancy. Defaults to 10.

    Returns:
        depth_map (tensor, N_rays): estimated distance to object.
        depth_var (tensor, N_rays): depth variance/uncertainty.
        rgb_map (tensor, N_rays*3): estimated RGB color of a ray.
        weights (tensor, N_rays*N_samples): weights assigned to
        each sampled color.
    """
    def raw2alpha(raw, dists, act_fn=F.relu):
        return 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = dists.float()
    dists = torch.cat([
        dists,
        torch.Tensor([1e10]).float().to(device).expand(dists[..., :1].shape)
    ], -1)  # [N_rays, N_samples]

    # different ray angle corresponds to different unit length
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = raw[..., :-1]
    if occupancy:
        raw[..., 3] = torch.sigmoid(coef * raw[..., -1])
        alpha = raw[..., -1]
    else:
        # original nerf, volume density
        alpha = raw2alpha(raw[..., -1], dists)  # (N_rays, N_samples)

    weights = alpha.float() * torch.cumprod(
        torch.cat([
            torch.ones((alpha.shape[0], 1)).to(device).float(),
            (1. - alpha + 1e-10).float()
        ], -1).float(), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # (N_rays, 3)
    depth_map = torch.sum(weights * z_vals, -1)  # (N_rays)
    tmp = (z_vals - depth_map.unsqueeze(-1))  # (N_rays, N_samples)
    depth_var = torch.sum(weights * tmp * tmp, dim=1)  # (N_rays)
    return depth_map, depth_var, rgb_map, weights


def raw2outputs_nerf_color2(raw, z_vals, rays_d, device='cuda:0', coef=0.1):
    """Transforms model's predictions to semantically meaningful values.

    Args:
        raw (tensor, (N_rays,N_samples,4) ): prediction from model. i.e.
          (R,G,B) and density
        z_vals (tensor, (N_rays,N_samples) ): integration time. i.e.
          the sampled locations on this ray
        rays_d (tensor, (N_rays,3) ): direction of each ray.
        device (str, optional): device. Defaults to 'cuda:0'.
        coef (float, optional): to multiply the input of sigmoid
          function when calculating occupancy. Defaults to 0.1.

    Returns:
        depth_map (tensor, N_rays): estimated distance to object.
        depth_var (tensor, N_rays): depth variance/uncertainty along
          the ray, see eq(7) in paper.
        rgb_map (tensor, (N_rays,3)): estimated RGB color of a ray.
        weights (tensor, (N_rays,N_samples) ): weights assigned to each
          sampled color.
    """

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = dists.float()
    dists = torch.cat([
        dists,
        torch.Tensor([1e10]).float().to(device).expand(dists[..., :1].shape)
    ], -1)

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = raw[..., :-1]

    raw[..., -1] = torch.sigmoid(coef * raw[..., -1])
    alpha = raw[..., -1]

    weights = alpha.float() * torch.cumprod(torch.cat([
        torch.ones((alpha.shape[0], 1)).to(device).float(),
        (1. - alpha + 1e-10).float()
    ], -1).float(),
                                            dim=-1)[:, :-1]
    weights_sum = torch.sum(weights, dim=-1).unsqueeze(-1) + 1e-10
    rgb_map = torch.sum(weights[..., None] * rgb, -2) / weights_sum
    depth_map = torch.sum(weights * z_vals, -1) / weights_sum.squeeze(-1)

    tmp = (z_vals - depth_map.unsqueeze(-1))
    depth_var = torch.sum(weights * tmp * tmp, dim=1)
    return depth_map, depth_var, rgb_map, weights


def get_mask_from_c2w(camera, bound, c2w, key, val_shape, depth_np):
    """Frustum feature selection based on current camera pose and depth image.

    Args:
        c2w (tensor): camera pose of current frame.
        key (str): name of this feature grid.
        val_shape (tensor): shape of the grid.
        depth_np (numpy.array): depth image of current frame.

    Returns:
        mask (tensor): mask for selected optimizable feature.
        points (tensor): corresponding point coordinates.
    """
    H, W, fx, fy, cx, cy, = (camera.height, camera.width, camera.fx, camera.fy,
                             camera.cx, camera.cy)
    X, Y, Z = torch.meshgrid(torch.linspace(bound[0][0], bound[0][1],
                                            val_shape[2]),
                             torch.linspace(bound[1][0], bound[1][1],
                                            val_shape[1]),
                             torch.linspace(bound[2][0], bound[2][1],
                                            val_shape[0]),
                             indexing='ij')

    points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
    if key == 'grid_coarse':
        mask = np.ones(val_shape[::-1]).astype(bool)
        return mask
    points_bak = points.clone()
    c2w = c2w.detach().cpu().numpy()
    w2c = np.linalg.inv(c2w)
    ones = np.ones_like(points[:, 0]).reshape(-1, 1)
    homo_vertices = np.concatenate([points, ones], axis=1).reshape(-1, 4, 1)
    cam_cord_homo = w2c @ homo_vertices
    cam_cord = cam_cord_homo[:, :3]
    K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
    cam_cord[:, 0] *= -1
    uv = K @ cam_cord
    z = uv[:, -1:] + 1e-5
    uv = uv[:, :2] / z
    uv = uv.astype(np.float32)

    remap_chunk = int(3e4)
    depths = []
    for i in range(0, uv.shape[0], remap_chunk):
        depths += [
            cv2.remap(depth_np,
                      uv[i:i + remap_chunk, 0],
                      uv[i:i + remap_chunk, 1],
                      interpolation=cv2.INTER_LINEAR)[:, 0].reshape(-1, 1)
        ]
    depths = np.concatenate(depths, axis=0)

    edge = 0
    mask = (uv[:, 0] < W - edge) * (uv[:, 0] > edge) * (
        uv[:, 1] < H - edge) * (uv[:, 1] > edge)

    # For ray with depth==0, fill it with maximum depth
    zero_mask = (depths == 0)
    depths[zero_mask] = np.max(depths)

    # depth test
    mask = mask & (0 <= -z[:, :, 0]) & (-z[:, :, 0] <= depths + 0.5)
    mask = mask.reshape(-1)

    # add feature grid near cam center
    ray_o = c2w[:3, 3]
    ray_o = torch.from_numpy(ray_o).unsqueeze(0)

    dist = points_bak - ray_o
    dist = torch.sum(dist * dist, axis=1)
    mask2 = dist < 0.5 * 0.5
    mask2 = mask2.cpu().numpy()
    mask = mask | mask2

    points = points[mask]
    mask = mask.reshape(val_shape[2], val_shape[1], val_shape[0])
    return mask
