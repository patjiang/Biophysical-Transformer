#import torch_interpolations as ptinterp
import copy, os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import numpy as np
import random
from itertools import chain

PAD_TOKEN_ID = 8000
MASK_TOKEN_ID = 8001

def Norm_Scale(to_norm, ma_x=4981.8994140625, mi_x=-4981.8994140625):
    return to_norm.div_(ma_x)

def gen_rot():
    typedict = {0: 'xyx',
                1: 'xyz',
                2: 'xzx',
                3: 'xzy',
                4: 'yxy',
                5: 'yxz',
                6: 'yzx',
                7: 'yzy',
                8: 'zxy',
                9: 'zxz',
                10:'zyx',
                11:'zyz'}
    
    def exp_ix(x):
        eix = np.exp(1j * x)
        return eix.imag, eix.real
    def rots(thet, t):
        s, c = exp_ix(thet)
        if(t == 'x'):
            return torch.tensor([[1, 0, 0],
                                 [0, c, -s],
                                 [0, s, c]])
        elif(t == 'y'):
            return torch.tensor([[c, 0, -s],
                                 [0, 1, 0],
                                 [s, 0, c]])
        elif(t == 'z'):
            return torch.tensor([[c, -s, 0],
                                 [s, c, 0],
                                 [0, 0, 1]])
        else:
            return None
    
    angles = np.random.uniform(0, 360, 3)
    rot_type = typedict[random.randint(0, 11)]
    transfrm = [rots(angles[idx], rot_type[idx]) for idx in range(len(angles))]
    return torch.matmul(torch.matmul(transfrm[0], transfrm[1]), transfrm[2]).float()

def rotate_voxel(voxel, transfrms):
    device = voxel.device
    trans_mat_inv = torch.linalg.inv(transfrms).to(device)
    
    Nz, Ny, Nx = voxel.shape
    x, x_center = torch.linspace(0, Nx - 1, Nx, device=device), (Nx//2)+1
    y, y_center = torch.linspace(0, Ny - 1, Ny, device=device), (Ny//2)+1
    z, z_center = torch.linspace(0, Nz - 1, Nz, device=device), (Nz//2)+1
    
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    coor = torch.stack([xx - x_center, yy - y_center, zz - z_center])
    
    coor_prime = torch.tensordot(trans_mat_inv, coor, dims=([1], [0]))

    xx_prime = coor_prime[0] + x_center
    yy_prime = coor_prime[1] + y_center
    zz_prime = coor_prime[2] + z_center

    x_valid1 = xx_prime >= 0
    x_valid2 = xx_prime <= Nx - 1
    y_valid1 = yy_prime >= 0
    y_valid2 = yy_prime <= Ny - 1
    z_valid1 = zz_prime >= 0
    z_valid2 = zz_prime <= Nz - 1
    valid_voxel = x_valid1 * x_valid2 * y_valid1 * y_valid2 * z_valid1 * z_valid2

    z_valid_idx, y_valid_idx, x_valid_idx = torch.where(valid_voxel > 0)

    transvox = torch.zeros((Nz, Ny, Nx), device=device)

    data_w_coor = ptinterp.RegularGridInterpolator(((z).to(device), (y).to(device), (x).to(device)), voxel)
    
    interp_points = torch.stack([zz_prime[z_valid_idx, y_valid_idx, x_valid_idx],
                                 yy_prime[z_valid_idx, y_valid_idx, x_valid_idx],
                                 xx_prime[z_valid_idx, y_valid_idx, x_valid_idx]])
    
    interp_result = data_w_coor(interp_points)

    transvox[z_valid_idx, y_valid_idx, x_valid_idx] = interp_result.float().detach()

    return transvox

def translate_voxel(voxel, amt):
    return torch.roll(voxel, shifts=amt, dims=(0, 1, 2))

def truncate_outliers(tensor, threshold=200):
    tensor = tensor.contiguous()
    median = torch.median(tensor)
    mask = (tensor >= median-threshold) & (tensor <= median+threshold)
    tensor[~mask] = 0
    return tensor

def iqr_norm(data, threshold=200, glob_min = -4981.8994140625):
    median = torch.median(data.view(-1))
    trunc = torch.clamp(data, median-threshold, median+threshold)
    trunc[(trunc==(median-threshold)) | (trunc==(median+threshold))] = 0
    return trunc, median+threshold, median-threshold

class BiophysicalRepresentation(Dataset):
    """Load data as pytorch dataset."""
    def __init__(self, in_dir, device, subs = None ,env_dir = None):
        self.dir = in_dir
        self.device = device
        self.rototranslation = False
        self.env = False
        if(env_dir != None):
            pathlist = os.listdir(env_dir)
            self.env_dir = env_dir
            self.env = True
        else:
            pathlist = os.listdir(in_dir)
        if subs != None:
            self.distinct_proteins = pathlist[subs[0]:subs[1]]
        else:
            self.distinct_proteins = pathlist
        
    def __len__(self):
        return len(self.distinct_proteins)
        
    def rot(self):
        self.rototranslation = not self.rototranslation
    
    def shuffle(self):
        random.shuffle(self.distinct_proteins)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        i = self.distinct_proteins[idx]
        if ('.pt' not in i):
            return []
        
        if(self.env):
            initdict = {}
            tmp = torch.load(f"{self.dir}/{i}", map_location = 'cpu')
            env_tups = torch.load(f"{self.env_dir}/{i}", map_location = 'cpu')
            b_ind = {value[0]: key for key, value in tmp.items()}
            lisss = {index: b_ind[index] for index, _ in env_tups.values() if index in b_ind}
            pdbid = i.split('.pt')[0]
            potlist, kmers, index_list, chain, msk = [], [], [], [], []
            rote = gen_rot()
            p_max, p_min = -99999999999, 9999999999
            for n in env_tups.keys():
                idx, tens = env_tups[n]
                seqkey = lisss[idx]
                index_list.append(tmp[seqkey][0])
                mask = tmp[seqkey][1].abs().sub(torch.median(tmp[seqkey][1]))
                pots = torch.where(mask.abs() < 200, tmp[seqkey][1], torch.zeros_like(tmp[seqkey][1]))
                chain.append(tmp[seqkey][2])
                epots, ma, mi = iqr_norm(tens)#truncate_outliers(tens)
                if(ma > p_max):
                    p_max = ma
                if(mi < p_min):
                    p_min = mi

                if(self.rototranslation):
                    msk_gen = rotate_voxel(pots.to(self.device), rote.to(self.device))
                    toappen = rotate_voxel(epots.to(self.device), rote.to(self.device))
                else:
                    msk_gen = pots#.to(self.device)
                    toappen = epots#.to(self.device)
                
                msk1 = (torch.where(msk_gen.abs() < 5, 0, 1))
                msk2 = (torch.where((toappen.abs() - torch.median(toappen)) < 5, 0, 1))
                potlist.append((msk_gen, toappen))
                msk.append((msk1, msk2))
                kmers.append(seqkey)
            p_max, p_min = 4981.8994140625, -4981.8994140625
            for p in range(len(potlist)):
                tmp1 = Norm_Scale(potlist[p][0], 203.0406, -202.9563)
                tmp2 = Norm_Scale(potlist[p][1], p_max, p_min)
                
                potlist[p] = (tmp1, tmp2)
                
            initdict['ord'] = index_list
            initdict['seq'] = kmers
            initdict['atompot'] = potlist
            initdict['ori'] = [pdbid] * len(kmers)
            initdict['chain'] = chain
            initdict['mask'] = msk
            initdict['minmax'] = (p_min, p_max)
            del tmp, env_tups
            
        else:
            initdict = {}
            tmp = torch.load(f"{self.dir}/{i}", map_location = self.device)
            pdbid = i.split('.pt')[0]
            kmers = list(tmp.keys())
            if(len(kmers) == 0):
                print(i)
            kmers.sort()
            potlist = []
            index_list = []
            chain = []
            #etc = []
            msk = []
            rote = gen_rot()
            #tra = (np.random.randint(0, 4), np.random.randint(0, 7), np.random.randint(0, 4))
            for i in kmers:
                index_list.append(tmp[i][0])
                #potlist.append(tmp[i][1])
                mask = tmp[i][1].abs().sub(torch.median(tmp[i][1]))
                chain.append(tmp[i][2][1])
                #pots_cons = torch.where(mask.abs() > 5, tmp[i][1], torch.zeros_like(tmp[i][1]))
                #pots = torch.where(mask.abs() < 200, pots_cons, torch.zeros_like(pots_cons))
                pots = torch.where(mask.abs() < 200, tmp[i][1], torch.zeros_like(tmp[i][1]))
                if(self.rototranslation):
                    toapp = rotate_voxel(pots, rote.to(self.device))
                else:
                    toapp = pots
                msk.append(torch.where(toapp.abs() < 5, 0, 1))
                potlist.append(toapp)
                
            initdict['ord'] = index_list
            initdict['seq'] = kmers
            initdict['atompot'] = potlist
            initdict['ori'] = [pdbid] * len(kmers)
            initdict['chain'] = chain
            initdict['mask'] = msk
            del tmp
            
        return initdict

def order_model_nonorm(data):
    ords = copy.deepcopy(data['ord'])
    ords.sort()
    get_indexes = [data['ord'].index(x) for x in ords]
    tdat = torch.stack([data['atompot'][ind][0] for ind in get_indexes])
    cdat = torch.stack([data['atompot'][ind][1] for ind in get_indexes])
    seqs = [data['seq'][ind] for ind in get_indexes]
    tmsk = torch.stack([data['mask'][ind][0] for ind in get_indexes])
    cmsk = torch.stack([data['mask'][ind][1] for ind in get_indexes])
    return tdat, cdat, seqs, tmsk, cmsk

def order_model_nonorm_batch(data, bs=16):
    ords = copy.deepcopy(data['ord'])
    ords.sort()
    get_indexes = [data['ord'].index(x) for x in ords]
    tdat = torch.stack([data['atompot'][ind][0] for ind in get_indexes]).unsqueeze(1)
    cdat = torch.stack([data['atompot'][ind][1] for ind in get_indexes]).unsqueeze(1)
    seqs = [data['seq'][ind] for ind in get_indexes]
    tmsk = torch.stack([data['mask'][ind][0] for ind in get_indexes])
    cmsk = torch.stack([data['mask'][ind][1] for ind in get_indexes])
    
    generator = torch.Generator()
    generator.manual_seed(42)
    
    seqsl= []
    tdatl = DataLoader(tdat, batch_size = bs, shuffle = False, generator = generator)
    cdatl = DataLoader(cdat, batch_size = bs, shuffle = False, generator = generator)
    tmskl = DataLoader(tmsk, batch_size = bs, shuffle = False, generator = generator)
    cmskl = DataLoader(cmsk, batch_size = bs, shuffle = False, generator = generator)
    for i in range((len(tdat) // bs) + 1):
        
        seqsl.append([seqs[(i*bs):((i + 1)*bs)]])    
    
    return zip(tdatl, cdatl, seqsl, tmskl, cmskl)
    
def order_model_nonch(data, bs, shuff = False, ma = 203.0406, mi = -202.9563):
    baseindexes, ch = data['ord'], data['chain']
    outdict = defaultdict(list)
    for i, (chi, ordi) in enumerate(zip(ch, baseindexes)):
        outdict[chi].append((ordi, i))
    ol, sqs, msks = [], [], []
    generator = torch.Generator()
    seed  = random.randint(0, 100000)
    generator.manual_seed(seed)
    for k, v in outdict.items():
        v.sort(key=lambda x: x[0])
        tmp, tmp2, tmp3 = [], [], []
        for _, idx in v:
            tmp.append(data['atompot'][idx])
            tmp2.append(data['seq'][idx])
            tmp3.append(data['mask'][idx])
        if(bs == -1):
            ol.append(torch.stack(tmp))
            msks.append(torch.stack(tmp3))
        else:
            ol.append(DataLoader(tmp, batch_size = bs, shuffle = shuff, generator = generator))
            msks.append(DataLoader(tmp3, batch_size = bs, shuffle = shuff, generator = generator))
        sqs.append(tmp2)
        
    return ol, sqs, msks

def order_model(data, ma_x = 203.0406, mi_x = -202.9563):
    ords = copy.deepcopy(data['ord'])
    ords.sort()
    get_indexes = [data['ord'].index(x) for x in ords]
    return torch.stack([Norm_Scale(data['atompot'][ind], ma_x, mi_x) for ind in get_indexes]), [data['seq'][ind] for ind in get_indexes], torch.stack([data['mask'][ind] for ind in get_indexes])

def order_model_ch(data, bs, shuff = False, ma = 203.0406, mi = -202.9563):
    baseindexes, ch = data['ord'], data['chain']
    outdict = defaultdict(list)
    for i, (chi, ordi) in enumerate(zip(ch, baseindexes)):
        outdict[chi].append((ordi, i))
    ol, sqs, msks = [], [], []
    generator = torch.Generator()
    seed  = random.randint(0, 100000)
    generator.manual_seed(seed)
    for k, v in outdict.items():
        v.sort(key=lambda x: x[0])
        tmp, tmp2, tmp3 = [], [], []
        for _, idx in v:
            tmp.append(Norm_Scale(data['atompot'][idx], ma, mi))
            tmp2.append(data['seq'][idx])
            tmp3.append(data['mask'][idx])
        if(bs == -1):
            ol.append(torch.stack(tmp))
            msks.append(torch.stack(tmp3))
        else:
            ol.append(DataLoader(tmp, batch_size = bs, shuffle = shuff, generator = generator))
            msks.append(DataLoader(tmp3, batch_size = bs, shuffle = shuff, generator = generator))
        sqs.append(tmp2)
        
    return ol, sqs, msks

def ret_uniq_index(seqlist):
    cleaned_seqs = [''.join([se.split('_')[0] for se in seq]) for seq in seqlist]
    uniq = list(set(cleaned_seqs))
    nidx = [cleaned_seqs.index(uni) for uni in uniq]
    random.shuffle(nidx)
    return nidx


def shuffle_model(data, ma, mi):
    seq = copy.deepcopy(data['seq'])
    random.shuffle(seq)
    get_indexes = [data['seq'].index(x) for x in seq]
    return [Norm_Scale(data['atompot'][ind], ma, mi) for ind in get_indexes], seq

def seqs_to_tens(seqs):
    seqdict = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7,
        'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15,
        'T': 16, 'W': 17, 'Y': 18, 'V': 19
    }
    olist = []
    for s in seqs:
        a = s.split('_')[0]
        if(len(a) != 3):
            olist.append(PAD_TOKEN_ID)
            continue
        uuid = seqdict.get(a[0], 0)*400 + seqdict.get(a[1], 0)*20 + seqdict.get(a[2], 0)
        olist.append(uuid)
    return torch.LongTensor(olist)

def apply_mlm_masking(seqs, device='cpu'):
    masked_seqs = seqs.clone()
    labels = seqs.clone()

    padding_mask = (seqs == PAD_TOKEN_ID)

    prob = torch.full(labels.shape, 0.15, device=device)
    masked_indices = torch.bernoulli(prob).bool()
    masked_indices[padding_mask] = False

    labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.85, device=device)).bool() & masked_indices
    masked_seqs[indices_replaced] = MASK_TOKEN_ID

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(0, PAD_TOKEN_ID, labels.shape, dtype=torch.long, device=device)
    
    masked_seqs[indices_random] = random_words[indices_random]
    
    return masked_seqs, labels

class env_collate:
    def __init__(self, mask=True):
        self.mask = mask

    def __call__(self, batch):
        batch = batch[0]
        if isinstance(batch, list):
            return [None]
        dat, eat, seqs, tmask, cmask = order_model_nonorm(batch)
        max_len = 0
        seq_len = eat.shape[0]

        if seq_len < max_len:
            pad_len = max_len - seq_len
            eat_pad = torch.zeros(
                (pad_len, *eat.shape[1:]),
                dtype=eat.dtype,
                device=eat.device
            )
            cmask_pad = torch.zeros(
                (pad_len, *cmask.shape[1:]),
                dtype=cmask.dtype,
                device=cmask.device
            )
            eat = torch.cat([eat, eat_pad], dim=0)
            cmask = torch.cat([cmask, cmask_pad], dim=0)
            seqs = seqs + ['XXX'] * pad_len
            
        seq_tensor = seqs_to_tens(seqs).unsqueeze(0)
        
        if self.mask:
            masked_seqs, mlm_labels = apply_mlm_masking(seq_tensor, device=eat.device)
            return [
                (
                    eat.to(torch.float16),
                    masked_seqs,
                    mlm_labels,
                    cmask.to(torch.float16),
                    seq_tensor
                )
            ]

        else:
            return [
                (
                    eat.to(torch.float16),
                    seq_tensor,
                    cmask.to(torch.float16),
                    
                )
            ]


        

class ChainedDataLoader:
    def __init__(self, dataset, **dataloader_kwargs):
        self.dataset = dataset
        self.dataloader_kwargs = dataloader_kwargs
    
    def __iter__(self):
        return chain.from_iterable(DataLoader(self.dataset, **self.dataloader_kwargs))
