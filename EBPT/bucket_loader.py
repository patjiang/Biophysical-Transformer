import os
import sys
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
import torch
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
else:
    print("CUDA not found. Check your environment.")
    sys.exit(1)

import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from collections import defaultdict
from data import PreprocessedBiophysicalDataset, MLMCollate

TRAIN_DIR = "../../kstruct_pt_preprocessed/train"
NUM_WORKERS = 4 

class DistributedBucketSampler(Sampler):
    def __init__(self, dataset_lengths, num_replicas, rank, bin_size=10, seed=42):
        self.dataset_lengths = dataset_lengths
        self.num_replicas = num_replicas
        self.rank = rank
        self.bin_size = bin_size
        self.seed = seed
        self.epoch = 0
        self.length_to_indices = defaultdict(list)
        for idx, length in enumerate(dataset_lengths):
            bucket_id = length // self.bin_size
            self.length_to_indices[bucket_id].append(idx)
            
        self.macrobatches = self._form_macrobatches()
        
    def _form_macrobatches(self):
        macrobatches = []
        global_remainders = []

        for bucket_id, indices in self.length_to_indices.items():
            indices.sort(key=lambda idx: self.dataset_lengths[idx])
            
            num_full = (len(indices) // self.num_replicas) * self.num_replicas
            full_indices = indices[:num_full]
            remainders = indices[num_full:]
            for i in range(0, len(full_indices), self.num_replicas):
                macrobatches.append(full_indices[i : i + self.num_replicas])
            
            global_remainders.extend(remainders)
        
        global_remainders.sort(key=lambda idx: self.dataset_lengths[idx])
        
        for i in range(0, len(global_remainders), self.num_replicas):
            chunk = global_remainders[i : i + self.num_replicas]
            if len(chunk) == self.num_replicas:
                macrobatches.append(chunk)
                
        return macrobatches

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        shuffled_indices = torch.randperm(len(self.macrobatches), generator=g).tolist()
        
        for idx in shuffled_indices:
            macrobatch = self.macrobatches[idx]
            yield macrobatch[self.rank]

    def __len__(self):
        return len(self.macrobatches)

def main():
    dist.init_process_group(backend="nccl")    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    train_dataset = PreprocessedBiophysicalDataset(TRAIN_DIR)
    if hasattr(train_dataset, 'lengths'):
        dataset_lengths = train_dataset.lengths
    else:
        if rank == 0:
            print("Calculating true dataset lengths...")
        dataset_lengths = [len(item['seqs']) for item in train_dataset] 
    
    true_total_tokens = sum(dataset_lengths)

    train_sampler = DistributedBucketSampler(
        dataset_lengths=dataset_lengths,
        num_replicas=world_size,
        rank=rank,
        bin_size=10
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=1, 
        num_workers=NUM_WORKERS, 
        collate_fn=MLMCollate(mask=True, device='cpu'), 
        sampler=train_sampler,
        pin_memory=True
    )

    dist.barrier()
    if rank == 0:
        print(f"\n[Ready] Processing {world_size} GPUs with {len(train_sampler)} macrobatches\n")

    total_tokens_processed = 0

    for batch in train_loader:
        masked_seqs = batch[1] 
        seq_length = masked_seqs.shape[1] 
        total_tokens_processed += seq_length

    dist.barrier()
    tokens_tensor = torch.tensor([total_tokens_processed], dtype=torch.long, device=torch.cuda.current_device())
    gathered_tokens = [torch.zeros_like(tokens_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tokens, tokens_tensor)

    if rank == 0:
        gpu_totals = [t.item() for t in gathered_tokens]
        sampled_total = sum(gpu_totals)
        diff = true_total_tokens - sampled_total

        print("="*60)
        print("DDP BUCKET SAMPLER REPORT")
        print("="*60)
        for i, count in enumerate(gpu_totals):
            print(f"GPU {i} Processed:\t{count:,} tokens")
        print("-" * 60)
        print(f"Total processed:\t{sampled_total:,}")
        print(f"Dataset total:\t\t{true_total_tokens:,}")
        print(f"Dropped:\t\t{diff:,} ({(diff/true_total_tokens)*100:.4f}%)")
        print("="*60)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
