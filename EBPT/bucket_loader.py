import os
import torch
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
        for bucket_id, indices in self.length_to_indices.items():
            indices.sort(key=lambda idx: self.dataset_lengths[idx])
            for i in range(0, len(indices), self.num_replicas):
                chunk = indices[i : i + self.num_replicas]
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
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    train_dataset = PreprocessedBiophysicalDataset(TRAIN_DIR)
    if hasattr(train_dataset, 'lengths'):
        dataset_lengths = train_dataset.lengths
    else:
        if rank == 0:
            print("Calculating true dataset lengths... (this may take a moment)")
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
        print(f"\n[Node Ready] Iterating over full dataset with {world_size} GPUs...\n")

    total_tokens_processed = 0

    for batch in train_loader:
        masked_seqs = batch[0][1] 
        seq_length = masked_seqs.shape[1] 
        
        total_tokens_processed += seq_length

    dist.barrier()
    tokens_tensor = torch.tensor([total_tokens_processed], dtype=torch.long, device=f'cuda:{local_rank}')
    gathered_tokens = [torch.zeros_like(tokens_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tokens, tokens_tensor)

    if rank == 0:
        gpu_totals = [t.item() for t in gathered_tokens]
        sampled_total = sum(gpu_totals)

        print("="*60)
        print("DDP BUCKET SAMPLER REPORT")
        print("="*60)
        for i, count in enumerate(gpu_totals):
            print(f"GPU {i} Processed:\t{count:,} sequence tokens")
        print("-" * 60)
        print(f"Total tokens processed (All GPUs):\t{sampled_total:,}")
        print(f"True total tokens in dataset:\t\t{true_total_tokens:,}")
        
        diff = true_total_tokens - sampled_total
        if diff == 0:
            print("\nRESULT: Perfect Match. 100% of data processed.")
        else:
            print(f"\nRESULT: {diff:,} total tokens were dropped.")
            print(f"Percentage of dataset dropped: {(diff/true_total_tokens)*100:.4f}%")
            print("\n*Note: This difference represents the remainder sequences at the edges "
                  "of your buckets that couldn't cleanly form a size-6 macrobatch. This drop "
                  "is expected and mathematically necessary to prevent the node from hanging.*")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
