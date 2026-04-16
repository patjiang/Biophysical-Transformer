import os
import torch
import copy
from torch.utils.data import DataLoader
from data import BiophysicalRepresentation, pre_process_and_save

def run_preprocessing():
    device = 'cpu'
    in_dir = '../../data/kstruct_pt'
    env_dir = '../../data/kstruct_env_pt'
    
    parent_dir = os.path.dirname(os.path.normpath(in_dir))
    output_dir = os.path.join(parent_dir, 'preproced')
    
    raw_dataset = BiophysicalRepresentation(
        in_dir=in_dir, 
        device=device, 
        env_dir=env_dir
    )
    
    pre_process_and_save(raw_dataset, output_dir)

if __name__ == "__main__":
    run_preprocessing()
