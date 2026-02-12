import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchinfo import summary
from EBPT.data import *
from EBPT.EBPT import *
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

def main():
    parser = argparse.ArgumentParser(description="EBPT Training Script")
    
    parser.add_argument("--data_base", type=str, required=True)
    parser.add_argument("--data_env", type=str, required=True)
    parser.add_argument("--data_split", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=32)
    
    args = parser.parse_args()

    TOTAL_SAMPLES = 6250 
    split_idx = int(TOTAL_SAMPLES * args.data_split)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_dataset = BiophysicalRepresentation(
        args.data_base, device, [0, split_idx], args.data_env
    )
    vali_dataset = BiophysicalRepresentation(
        args.data_base, device, [split_idx, -1], args.data_env
    )

    train_loader = ChainedDataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        collate_fn=env_collate(mask=True), 
        shuffle=True
    )
    vali_loader = ChainedDataLoader(
        vali_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        collate_fn=env_collate(mask=False)
    )

    SAVE_DIR = "chkpts"
    torch.set_float32_matmul_precision('medium')

    checkpoint_cb = ModelCheckpoint(
        dirpath=SAVE_DIR,
        filename="best_{epoch:02d}-{val_loss:.4f}",
        save_top_k=10,
        monitor="val_loss",
        mode="min",
    )

    checkpoint_last = ModelCheckpoint(
        dirpath=SAVE_DIR,
        filename="last_{epoch:02d}",
        save_top_k=1,
        monitor="epoch",
        mode="max",
    )

    lr_monitor_cb = LearningRateMonitor(logging_interval="step")

    model = SE3PLMLightning(
        learning_rate=5e-5,
        weight_decay=1e-7,
        optimizer_type="adamw",
        cyclic_lr=True,
        max_lr_factor=40.0,
        step_size_up=10,
        lambda_wa=50.0,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loggers = [
        TensorBoardLogger(save_dir=SAVE_DIR, name="tb_logs", version=timestamp),
        CSVLogger(save_dir=SAVE_DIR, name="csv_logs", version=timestamp)
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        default_root_dir=SAVE_DIR,
        callbacks=[checkpoint_cb, checkpoint_last, lr_monitor_cb],
        precision='16-mixed',
        log_every_n_steps=1,
        logger=loggers,
        strategy='ddp_find_unused_parameters_true',
        profiler="simple"
    )

    print(summary(model))
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=vali_loader)

if __name__ == "__main__":
    main()
