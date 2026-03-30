import torch
import torch.nn as nn
import torch.nn.functional as F
import ot
from einops import rearrange

def Inverse_Norm_Scale(to_inv, max_val=4981.8994140625):
    """Rescales normalized model output back to physical units."""
    return to_inv * max_val

def loss_dirichlet_energy(ori, reconstruction):
    """Computes the difference in Dirichlet energy (gradient smoothness) between volumes."""
    def compute_energy(voxel_grid):
        # Differences along Depth, Height, and Width axes
        diff_d = voxel_grid[..., 1:, :, :] - voxel_grid[..., :-1, :, :]
        diff_h = voxel_grid[..., :, 1:, :] - voxel_grid[..., :, :-1, :]
        diff_w = voxel_grid[..., :, :, 1:] - voxel_grid[..., :, :, :-1]

        energy_d = (diff_d ** 2).sum(dim=(-3, -2, -1))
        energy_h = (diff_h ** 2).sum(dim=(-3, -2, -1))
        energy_w = (diff_w ** 2).sum(dim=(-3, -2, -1))

        return 0.5 * (energy_d + energy_h + energy_w)

    energy_gt = compute_energy(ori)
    energy_pred = compute_energy(reconstruction)
    
    # Square root difference to stabilize the energy loss magnitude
    diff = torch.sqrt(torch.abs(energy_pred) + 1e-8) - torch.sqrt(torch.abs(energy_gt) + 1e-8)
    loss = torch.mean(diff ** 2)

    return torch.clamp(loss, min=1e-6, max=10000.0)

def loss_topological_proxy(ori, reconstruction, num_slices=32, percentile_threshold=0.25):
    """Uses 1D persistence-like peaks along axes and diagonals to enforce structural topology."""
    def get_1d_persistence(vol, dim_axis, num_samples):
        B, C, D, H, W = vol.shape
        
        if dim_axis == 2:
            flat = vol.permute(0, 1, 3, 4, 2).reshape(-1, D)
        elif dim_axis == 3:
            flat = vol.permute(0, 1, 2, 4, 3).reshape(-1, H)
        else:
            flat = vol.reshape(-1, W)
        
        indices = torch.randperm(flat.size(0), device=flat.device)[:num_samples]
        rays = flat[indices]
        
        threshold = torch.quantile(rays, percentile_threshold, dim=1, keepdim=True)
        dilated = F.max_pool1d(rays.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        is_peak = (rays == dilated) & (rays > threshold)
        
        masked_rays = torch.where(is_peak, rays, -2 * torch.ones_like(rays))
        sorted_peaks, _ = torch.sort(masked_rays, dim=1, descending=True)
        
        num_real_peaks = int(is_peak.sum(dim=1).max().item())
        k = min(max(int(num_real_peaks * 1.2), 5), 50)
        return sorted_peaks[:, :k].clone()

    def get_diagonal_persistence(vol, diagonal_type, num_samples):
        B, C, D, H, W = vol.shape
        if diagonal_type == 'xy':
            min_size = min(D, H)
            diag_vol = torch.stack([vol[:, :, i, i, :] for i in range(min_size)], dim=2)
            flat = diag_vol.reshape(-1, min_size)
        elif diagonal_type == 'xz':
            min_size = min(D, W)
            diag_vol = torch.stack([vol[:, :, i, :, i] for i in range(min_size)], dim=2)
            flat = diag_vol.reshape(-1, min_size)
        else: # yz
            min_size = min(H, W)
            diag_vol = torch.stack([vol[:, :, :, i, i] for i in range(min_size)], dim=2)
            flat = diag_vol.reshape(-1, min_size)
        
        indices = torch.randperm(flat.size(0), device=flat.device)[:num_samples]
        rays = flat[indices]
        threshold = torch.quantile(rays, percentile_threshold, dim=1, keepdim=True)
        dilated = F.max_pool1d(rays.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        is_peak = (rays == dilated) & (rays > threshold)
        
        masked_rays = torch.where(is_peak, rays, -2 * torch.ones_like(rays))
        sorted_peaks, _ = torch.sort(masked_rays, dim=1, descending=True)
        
        num_real_peaks = int(is_peak.sum(dim=1).max().item())
        k = min(max(int(num_real_peaks * 1.2), 5), 50)
        return sorted_peaks[:, :k].clone()

    loss_accum = 0.0
    for axis in [2, 3, 4]:
        p_gt = get_1d_persistence(ori, axis, num_slices // 2)
        p_rec = get_1d_persistence(reconstruction, axis, num_slices // 2)
        max_k = max(p_gt.shape[1], p_rec.shape[1])
        loss_accum += F.mse_loss(F.pad(p_rec, (0, max_k - p_rec.shape[1])), F.pad(p_gt, (0, max_k - p_gt.shape[1])))
    
    for diag in ['xy', 'xz', 'yz']:
        p_gt = get_diagonal_persistence(ori, diag, num_slices // 3)
        p_rec = get_diagonal_persistence(reconstruction, diag, num_slices // 3)
        max_k = max(p_gt.shape[1], p_rec.shape[1])
        loss_accum += F.mse_loss(F.pad(p_rec, (0, max_k - p_rec.shape[1])), F.pad(p_gt, (0, max_k - p_gt.shape[1])))

    return loss_accum / 6.0

def loss_spectral(ori, rec):
    """L1 loss on the log-amplitude of the Fourier Transform."""
    fft_ori = torch.abs(torch.fft.rfftn(ori, dim=(-3, -2, -1)))
    fft_rec = torch.abs(torch.fft.rfftn(rec, dim=(-3, -2, -1)))
    return F.l1_loss(torch.log(fft_rec + 1e-8), torch.log(fft_ori + 1e-8))

def wa_loss(ori, reconstruction, denorm=True, split_posneg=True, max_val=4981.8994140625):
    with torch.amp.autocast('cuda', enabled=False):
        dev = ori.device
        scale_factor = max_val if denorm else 1.0
        def compute_axis_loss(t_ori, t_rec, pattern, projvec):
            flat_ori = rearrange(t_ori, pattern)
            flat_rec = rearrange(t_rec, pattern)
            
            dist = ot.sliced.max_sliced_wasserstein_distance(
                flat_ori, flat_rec, projections=projvec
            )
            return dist
        accumulated_loss = 0.0
        if split_posneg:
            with torch.no_grad():
                ori_pos_mass = torch.relu(ori).sum()
                ori_neg_mass = torch.relu(-ori).sum()
                eps = 1e-6
                total_mass = ori_pos_mass + ori_neg_mass + eps
                
                w_pos = total_mass / (2 * (ori_pos_mass + eps))
                w_neg = total_mass / (2 * (ori_neg_mass + eps))
                w_pos = torch.clamp(w_pos, min=0.1, max=10.0)
                w_neg = torch.clamp(w_neg, min=0.1, max=10.0)
                total_weight = w_pos + w_neg
                w_pos = w_pos / total_weight * 2.0
                w_neg = w_neg / total_weight * 2.0
            
            phases = [
                ("pos", lambda x: torch.relu(x), w_pos),
                ("neg", lambda x: torch.relu(-x), w_neg)
            ]
        else:
            phases = [("all", lambda x: x, 1.0)]
        
        for phase_name, op, weight in phases:
            curr_rec = op(reconstruction.float())
            with torch.no_grad():
                curr_ori = op(ori.float())
            
            dist = 0.0
            projvec = torch.tensor(
                ot.sliced.get_random_projections(1089, 1089)
            ).float().to(dev)
            dist += compute_axis_loss(curr_ori, curr_rec, 'b s x y z -> (b z) (s x y)', projvec)
            dist += compute_axis_loss(curr_ori, curr_rec, 'b s x y z -> (b y) (s x z)', projvec)        
            dist += compute_axis_loss(curr_ori, curr_rec, 'b s x y z -> (b x) (s y z)', projvec)
            del projvec
            diagvec = torch.tensor(
                ot.sliced.get_random_projections(33, 50)
            ).float().to(dev)
            dist += compute_axis_loss(curr_ori, curr_rec, 'b s x y z -> (b x y) (s z)', diagvec)
            dist += compute_axis_loss(curr_ori, curr_rec, 'b s x y z -> (b x z) (s y)', diagvec)
            dist += compute_axis_loss(curr_ori, curr_rec, 'b s x y z -> (b y z) (s x)', diagvec) 
            del diagvec
            accumulated_loss += dist * weight
            
            del curr_rec, curr_ori
        
        return accumulated_loss * scale_factor

def mi_loss(z, tmp=0.01):
    """Mutual Information loss (InfoNCE) for latent vectors."""
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / tmp
    lbls = torch.arange(z.size(0), device=z.device)
    return F.cross_entropy(sim, lbls)

def merge_mask(rec, dat, mask=None):
    """Combined MSE and Wasserstein loss, with optional spatial masking."""
    if mask is not None:
        mask = mask.unsqueeze(1)
        dat = dat * mask
        # We don't hard-zero the reconstruction here to allow the model to learn 
        # the boundary, but the loss is computed on the masked target.
    
    mse = F.mse_loss(Inverse_Norm_Scale(rec.float()), Inverse_Norm_Scale(dat.float()))
    wa = wa_loss(rec.float(), dat.float(), denorm=False)
    return mse + wa
