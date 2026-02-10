import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn
from torchinfo import summary
from data import *
#from model_interact import *
from model_dino import *
from itertools import chain
from torch.utils.data import DataLoader, TensorDataset, Dataset
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

NUM_WORKERS = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_dataset = BiophysicalRepresentation('../../data/kstruct_pt', device, [0,5000], '../../data/kstruct_env_pt')
vali_dataset = BiophysicalRepresentation('../../data/kstruct_pt', device, [5000,-1], '../../data/kstruct_env_pt')
train_loader = ChainedDataLoader(train_dataset, batch_size=1, num_workers=NUM_WORKERS, collate_fn=env_collate(mask=True), shuffle=True)
vali_loader = ChainedDataLoader(vali_dataset, batch_size=1, num_workers=NUM_WORKERS, collate_fn=env_collate(mask=False))

SAVE_DIR = "chkpts"
MAX_EPOCHS = 200
BASE_LR = 5e-5
MAX_LR_FACTOR = 40.0
STEP_SIZE_UP = 10
LAMBDA_WA = 50.0#5000.0

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
    learning_rate=BASE_LR,
    weight_decay=1e-7,
    optimizer_type="adamw",
    cyclic_lr=True,
    max_lr_factor=MAX_LR_FACTOR,
    step_size_up=STEP_SIZE_UP,
    lambda_wa=LAMBDA_WA,
)

#swa_callback = StochasticWeightAveraging(swa_lrs=BASE_LR * 0.1, swa_epoch_start=0.75)

def setup_loggers(save_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_logger = TensorBoardLogger(save_dir=save_dir, name="tb_logs", version=timestamp)
    csv_logger = CSVLogger(save_dir=save_dir, name="csv_logs", version=timestamp)
    return [tb_logger, csv_logger]

loggers = setup_loggers('chkpts')

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    devices="auto",
    default_root_dir=SAVE_DIR,
    callbacks=[checkpoint_cb, checkpoint_last, lr_monitor_cb],# swa_callback],
    precision='16-mixed',
    log_every_n_steps=1,
    #gradient_clip_val=2.0,
    logger=loggers,
    #accumulate_grad_batches=32,
    strategy='ddp_find_unused_parameters_true',
    profiler="simple"
)

print(summary(model))
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=vali_loader)
