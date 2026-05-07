# BiophysicalTransformer (WIP)

This repository implements a Biophysical Transformer capturing joint sequence-electrostatics latent representations using a Student-Teacher schema with sinkorn-knopp clustering for self-supervised learning. The output trunk is a Voxel generator trained GAN-style.

Follow the steps in [setup.md](EBPT/setup.md) to configure your environment, or run setup.sh:
```{bash}
chmod +x setup.sh
./setup.sh
```


for me:
```
git clone https://patjiang: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@github.com/patjiang/BiophysicalTransformer.git
```

```
python train_bt_elec.py --data_base ../full_3structs_pt \
                --data_env ../3struct_env_pt \
                --data_split 0.85 \
                --max_epochs 100
```


```
(EBPT) [asinghar@wheat-r5-ai35a EBPT]$ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 bucket_loader.py
Calculating true dataset lengths... (this may take a moment)
[rank4]:[W507 11:20:59.602822346 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 4]  using GPU 4 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify device_ids in barrier() to force use of a particular device, or call init_process_group() with a device_id.
[rank3]:[W507 11:20:59.602829906 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 3]  using GPU 3 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify device_ids in barrier() to force use of a particular device, or call init_process_group() with a device_id.
[rank5]:[W507 11:20:59.602829835 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 5]  using GPU 5 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify device_ids in barrier() to force use of a particular device, or call init_process_group() with a device_id.
[rank2]:[W507 11:20:59.602829775 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 2]  using GPU 2 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify device_ids in barrier() to force use of a particular device, or call init_process_group() with a device_id.
[rank1]:[W507 11:20:59.602834020 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 1]  using GPU 1 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify device_ids in barrier() to force use of a particular device, or call init_process_group() with a device_id.
[rank0]:[W507 11:21:00.841666639 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 0]  using GPU 0 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify device_ids in barrier() to force use of a particular device, or call init_process_group() with a device_id.

[Node Ready] Iterating over full dataset with 6 GPUs...

============================================================
DDP BUCKET SAMPLER REPORT
============================================================
GPU 0 Processed:	882,814 sequence tokens
GPU 1 Processed:	882,959 sequence tokens
GPU 2 Processed:	883,122 sequence tokens
GPU 3 Processed:	883,284 sequence tokens
GPU 4 Processed:	883,424 sequence tokens
GPU 5 Processed:	883,552 sequence tokens
------------------------------------------------------------
Total tokens processed (All GPUs):	5,299,155
True total tokens in dataset:		5,525,892

RESULT: 226,737 total tokens were dropped.
Percentage of dataset dropped: 4.1032%

*Note: This difference represents the remainder sequences at the edges of your buckets that couldn't cleanly form a size-6 macrobatch. This drop is expected and mathematically necessary to prevent the node from hanging.*

```
