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
python train_bt_elec.py --data_base ../../data/kstruct_pt \
                --data_env ../../data/kstruct_env_pt \
                --data_split 0.85 \
                --max_epochs 100
```
