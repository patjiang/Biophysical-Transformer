import torch
import torch.nn.functional as F
from einops import rearrange
import ot

def Inverse_Norm_Scale(to_inv, max_val=4981.8994140625):
    return to_inv * max_val

def loss_dirichlet_energy(ori, reconstruction):
    def compute_energy(voxel_grid):
        diff_d = voxel_grid[..., 1:, :, :] - voxel_grid[..., :-1, :, :]
        diff_h = voxel_grid[..., :, 1:, :] - voxel_grid[..., :, :-1, :]
        diff_w = voxel_grid[..., :, :, 1:] - voxel_grid[..., :, :, :-1]

        energy_d = (diff_d ** 2).sum(dim=(-3, -2, -1))
        energy_h = (diff_h ** 2).sum(dim=(-3, -2, -1))
        energy_w = (diff_w ** 2).sum(dim=(-3, -2, -1))

        return 0.5 * (energy_d + energy_h + energy_w)

    energy_gt = compute_energy(ori)
    energy_pred = compute_energy(reconstruction)
    diff = torch.sqrt(torch.abs(energy_pred) + 1e-8) - torch.sqrt(torch.abs(energy_gt) + 1e-8)
    loss = torch.mean(diff ** 2)

    return torch.clamp(loss, min=1e-6, max=10000.0)

def loss_topological_proxy(ori, reconstruction, num_slices=32, percentile_threshold=0.25):
    def get_1d_persistence(vol, dim_axis, num_samples):
        B, C, D, H, W = vol.shape
        
        if dim_axis == 2:
            flat = vol.permute(0, 1, 3, 4, 2).reshape(B * C * H * W, D)
        elif dim_axis == 3:
            flat = vol.permute(0, 1, 2, 4, 3).reshape(B * C * D * W, H)
        else:
            flat = vol.permute(0, 1, 2, 3, 4).reshape(B * C * D * H, W)
        
        indices = torch.randperm(flat.size(0), device=flat.device)[:num_samples]
        rays = flat[indices]
        del flat, indices
        
        threshold = torch.quantile(rays, percentile_threshold, dim=1, keepdim=True)
        dilated = F.max_pool1d(rays.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        is_peak = (rays == dilated) & (rays > threshold)
        del dilated
        
        masked_rays = torch.where(is_peak, rays, -2 * torch.ones_like(rays))
        del rays
        
        sorted_peaks, _ = torch.sort(masked_rays, dim=1, descending=True)
        del masked_rays
        
        num_real_peaks = (is_peak.sum(dim=1)).max().item()
        del is_peak
        k = min(max(int(num_real_peaks * 1.2), 5), 50)
        
        result = sorted_peaks[:, :k].clone()
        del sorted_peaks
        return result
    
    def get_diagonal_persistence(vol, diagonal_type, num_samples):
        B, C, D, H, W = vol.shape
        
        if diagonal_type == 'xy':
            min_size = min(D, H)
            diag_vol = torch.stack([vol[:, :, i, i, :] for i in range(min_size)], dim=2)
            flat = diag_vol.reshape(B * C * W, min_size)
        elif diagonal_type == 'xz':
            min_size = min(D, W)
            diag_vol = torch.stack([vol[:, :, i, :, i] for i in range(min_size)], dim=2)
            flat = diag_vol.reshape(B * C * H, min_size)
        else:
            min_size = min(H, W)
            diag_vol = torch.stack([vol[:, :, :, i, i] for i in range(min_size)], dim=2)
            flat = diag_vol.reshape(B * C * D, min_size)
        
        del diag_vol
        indices = torch.randperm(flat.size(0), device=flat.device)[:num_samples]
        rays = flat[indices]
        del flat, indices
        
        threshold = torch.quantile(rays, percentile_threshold, dim=1, keepdim=True)
        dilated = F.max_pool1d(rays.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        is_peak = (rays == dilated) & (rays > threshold)
        del dilated
        
        masked_rays = torch.where(is_peak, rays, -2 * torch.ones_like(rays))
        del rays
        
        sorted_peaks, _ = torch.sort(masked_rays, dim=1, descending=True)
        del masked_rays
        
        num_real_peaks = (is_peak.sum(dim=1)).max().item()
        del is_peak
        k = min(max(int(num_real_peaks * 1.2), 5), 50)
        
        result = sorted_peaks[:, :k].clone()
        del sorted_peaks
        return result

    loss_accum = 0.0
    num_directions = 0
    
    for axis in [2, 3, 4]:
        peaks_gt = get_1d_persistence(ori, axis, num_slices // 2)
        peaks_rec = get_1d_persistence(reconstruction, axis, num_slices // 2)
        
        max_k = max(peaks_gt.shape[1], peaks_rec.shape[1])
        if peaks_gt.shape[1] < max_k:
            peaks_gt = F.pad(peaks_gt, (0, max_k - peaks_gt.shape[1]), value=0.0)
        if peaks_rec.shape[1] < max_k:
            peaks_rec = F.pad(peaks_rec, (0, max_k - peaks_rec.shape[1]), value=0.0)
        
        loss_accum += F.mse_loss(peaks_rec, peaks_gt)
        del peaks_gt, peaks_rec
        num_directions += 1
    
    for diag in ['xy', 'xz', 'yz']:
        peaks_gt = get_diagonal_persistence(ori, diag, num_slices // 3)
        peaks_rec = get_diagonal_persistence(reconstruction, diag, num_slices // 3)
        
        max_k = max(peaks_gt.shape[1], peaks_rec.shape[1])
        if peaks_gt.shape[1] < max_k:
            peaks_gt = F.pad(peaks_gt, (0, max_k - peaks_gt.shape[1]), value=0.0)
        if peaks_rec.shape[1] < max_k:
            peaks_rec = F.pad(peaks_rec, (0, max_k - peaks_rec.shape[1]), value=0.0)
        
        loss_accum += F.mse_loss(peaks_rec, peaks_gt)
        del peaks_gt, peaks_rec
        num_directions += 1

    return loss_accum / num_directions

def loss_spectral(ori, rec):
    fft_ori = torch.fft.rfftn(ori, dim=(-3, -2, -1))
    fft_rec = torch.fft.rfftn(rec, dim=(-3, -2, -1))
    
    amp_ori = torch.abs(fft_ori)
    amp_rec = torch.abs(fft_rec)
    loss = F.l1_loss(torch.log(amp_rec + 1e-8), torch.log(amp_ori + 1e-8))    
    return loss

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

def seqs_to_tens(seqs):
    seqdict = {
        'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'E': 6, 'Q': 7, 'G': 8,
        'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16,
        'T': 17, 'W': 18, 'Y': 19, 'V': 20
    }
    olist = []
    for s in seqs:
        a = s.split('_')[0]
        if(len(a) != 3):
            olist.append(9000)
            continue
        uuid = seqdict.get(a[0], 40)*20**2 + seqdict.get(a[1], 40)*20 + seqdict.get(a[2], 40)
        olist.append(uuid)
    return torch.FloatTensor(olist).type(torch.float16)

def contrastive_loss(z, labels, margin=5.0):
    labels.extend(labels)
    labels = seqs_to_tens(labels).to(z.device)

    n = len(z)
    if n < 2:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    idx = torch.triu_indices(n, n, offset=1, device=z.device)

    z1 = z[idx[0]]
    z2 = z[idx[1]]
    lbls = (labels[idx[0]] == labels[idx[1]]).to(dtype=torch.float16, device=z.device)

    dist = ot.sliced.sliced_wasserstein_distance(z1, z2, p=2)
    pos_loss = lbls * dist.pow(2)
    neg_loss = (1 - lbls) * F.relu(margin - dist).pow(2)

    return (pos_loss + neg_loss).sum()

def mi_loss(z, tmp = 0.01):
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / tmp
    bs = z.size(0)
    lbls = torch.arange(bs).to(z.device)    
    return F.cross_entropy(sim, lbls, reduction='mean')

def merge_mask(rec, dat, projvec, mask=None):
    if mask is not None:
        mask = mask.unsqueeze(1)
        dat = dat * mask
        dat[mask == 0] = 0
    return F.mse_loss(Inverse_Norm_Scale(rec.float()), Inverse_Norm_Scale(dat.float()), reduction = 'mean') + wa_loss(rec.float(), dat, projvec)
