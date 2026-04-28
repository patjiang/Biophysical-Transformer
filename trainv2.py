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

torch.autograd.set_detect_anomaly(True)
NUM_WORKERS = 24
device = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_DIR = '../kstruct_pt_preprocessed/train' 
VAL_DIR = '../kstruct_pt_preprocessed/val'

train_dataset = PreprocessedBiophysicalDataset(TRAIN_DIR)
vali_dataset = PreprocessedBiophysicalDataset(VAL_DIR)

train_loader = DataLoader(
    train_dataset, 
    batch_size=1, 
    num_workers=NUM_WORKERS, 
    collate_fn=MLMCollate(mask=True, device='cpu'), 
    shuffle=True,
    pin_memory=True
)

vali_loader = DataLoader(
    vali_dataset, 
    batch_size=1, 
    num_workers=NUM_WORKERS // 2, 
    collate_fn=MLMCollate(mask=False, device='cpu'),
    pin_memory=True
)

SAVE_DIR = "chkpts"
MAX_EPOCHS = 200
BASE_LR = 1e-5
MAX_LR_FACTOR = 10.0
STEP_SIZE_UP = 5000
LAMBDA_WA = 5.0

torch.set_float32_matmul_precision('medium')

checkpoint_cb = ModelCheckpoint(
    dirpath=SAVE_DIR,
    filename="less_mask_{epoch:02d}-{val_loss:.4f}",
    save_top_k=10,
    monitor="val_loss",
    mode="min",
)

checkpoint_last = ModelCheckpoint(
    dirpath=SAVE_DIR,
    filename="gan_{epoch:02d}",
    save_top_k=1,
    monitor="epoch",
    mode="max",
)

lr_monitor_cb = LearningRateMonitor(logging_interval="step")

model = SE3PLMLightning(
    learning_rate=BASE_LR,
    weight_decay=1e-6,
    optimizer_type="adam",
    cyclic_lr=True,
    max_lr_factor=MAX_LR_FACTOR,
    step_size_up=STEP_SIZE_UP,
    lambda_wa=LAMBDA_WA,
)

def setup_loggers(save_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_logger = TensorBoardLogger(save_dir=save_dir, name="tb_logs", version=timestamp)
    csv_logger = CSVLogger(save_dir=save_dir, name="csv_logs", version=timestamp)
    return [tb_logger, csv_logger]

loggers = setup_loggers(SAVE_DIR)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
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
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=vali_loader)
