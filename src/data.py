"""Data loading and collation for BioT training."""

import os
from itertools import chain

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from .span_masking import span_mask_tokens

PAD_TOKEN_ID = 8000
MASK_TOKEN_ID = 8001


def seqs_to_tens(seqs):
    """Convert 3-mer string tokens to integer ids."""
    seqdict = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7,
        'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15,
        'T': 16, 'W': 17, 'Y': 18, 'V': 19
    }
    olist = []
    for s in seqs:
        a = s.split('_')[0]
        if len(a) != 3:
            olist.append(PAD_TOKEN_ID)
            continue
        uuid = seqdict.get(a[0], 0) * 400 + seqdict.get(a[1], 0) * 20 + seqdict.get(a[2], 0)
        olist.append(uuid)
    return torch.LongTensor(olist)


class MLMCollate:
    """Collate function that tokenizes sequences and optionally applies span masking."""

    def __init__(self, mask=True, device='cpu'):
        self.mask = mask
        self.device = device

    def __call__(self, batch):
        samples = []
        for item in batch:
            eat = item['cdat']
            cmsk = item['cmsk']
            seqs = item['seqs']

            seq_tensor = seqs_to_tens(seqs).unsqueeze(0).to(self.device)
            eat = eat.to(self.device)
            cmsk = cmsk.to(self.device)

            if self.mask:
                masked_seqs, mlm_labels = span_mask_tokens(
                    seq_tensor, PAD_TOKEN_ID, MASK_TOKEN_ID,
                    mask_rate=0.15, span_residues=1)
                samples.append((eat, masked_seqs, mlm_labels, cmsk, seq_tensor))
            else:
                samples.append((eat, seq_tensor, cmsk))

        return samples


class ChainedDataLoader:
    """Wraps a dataset so each item yields one or more batches, chained into
    a single iterator.  Used because each protein is its own "batch"."""

    def __init__(self, dataset, **dataloader_kwargs):
        self.dataset = dataset
        self.dataloader_kwargs = dataloader_kwargs

    def __len__(self):
        total_batches = len(self.dataset)
        if dist.is_available() and dist.is_initialized():
            return total_batches // dist.get_world_size()
        return total_batches

    def __iter__(self):
        return chain.from_iterable(DataLoader(self.dataset, **self.dataloader_kwargs))


class PreprocessedBiophysicalDataset(Dataset):
    """Loads pre-computed biophysical tensors directly from disk."""

    def __init__(self, preprocessed_dir):
        self.preprocessed_dir = preprocessed_dir
        self.files = [f for f in os.listdir(preprocessed_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.preprocessed_dir, self.files[idx])
        return torch.load(file_path, weights_only=False)
