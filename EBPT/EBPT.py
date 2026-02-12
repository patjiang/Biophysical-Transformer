import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import copy

from einops import rearrange, repeat

from .modules import (
    TransformerStackEnc,
    InteractionTokenBlock,
    VoxelizeTrunk,
    PatchDiscriminator3D,
)
from .losses import wa_loss

class EBPTLightning(pl.LightningModule):
    def __init__(self, 
                 learning_rate: float = 1e-3,
                 d_learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 optimizer_type: str = "adam",
                 cyclic_lr: bool = True,
                 max_lr_factor: float = 100.0,
                 step_size_up: int = 1000,
                 lambda_wa: float = 5000.0,
                 lambda_adv: float = 0.001,
                 lambda_dino: float = 0.1,
                 num_prototypes: int = 128,
                 teacher_momentum: float = 0.992,
                 dino_warmup_steps: int = 10000,
                 **kwargs):
        super().__init__()
        self.automatic_optimization = False
        
        self.save_hyperparameters()
        self.lambda_wa = lambda_wa
        self.lambda_adv = lambda_adv
        self.lambda_dino = lambda_dino
        self.teacher_momentum = teacher_momentum
        self.dino_warmup_steps = dino_warmup_steps
        
        self.projvec = None
        
        self.student_encoder = TransformerStackEnc(input_dim=128, output_dim=64, num_layers=5, vocab_size=8002)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.teacher_encoder.requires_grad_(False)
        
        self.mlm_head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 8002))
        
        self.student_dino_head = nn.Linear(64, num_prototypes)
        self.teacher_dino_head = copy.deepcopy(self.student_dino_head)
        self.teacher_dino_head.requires_grad_(False)
        
        self.i = InteractionTokenBlock()
        self.b = VoxelizeTrunk()
        nn.init.constant_(self.b.convout[0].bias, 0.5)
        
        self.discriminator = PatchDiscriminator3D(in_channels=1)

    def forward(self, seq, pred=False):
        z = self.student_encoder(seq)
        if pred:
            return z
        bos = z[:, 0, :]
        zr = rearrange(z[:, 1:, :], 'b l c -> (b l) 1 c')
        bos_expanded = repeat(bos, 'b c -> (b l) 1 c', l=z.shape[1] - 1)
        zr = self.i(zr)
        recon = self.b(zr, bos_expanded)
        return z, recon

    @torch.no_grad()
    def update_teacher_ema(self):
        for student_p, teacher_p in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            teacher_p.data.mul_(self.teacher_momentum).add_(student_p.data, alpha=1 - self.teacher_momentum)
        for student_p, teacher_p in zip(self.student_dino_head.parameters(), self.teacher_dino_head.parameters()):
            teacher_p.data.mul_(self.teacher_momentum).add_(student_p.data, alpha=1 - self.teacher_momentum)

    def sinkhorn_knopp(self, teacher_output, n_iter=3, epsilon=0.1):
        teacher_output = teacher_output - teacher_output.mean(dim=-1, keepdim=True)
        
        Q = torch.exp(teacher_output / epsilon)
        
        if torch.isnan(Q).any() or torch.isinf(Q).any():
            return F.softmax(teacher_output / epsilon, dim=-1)
        
        for _ in range(n_iter):
            Q /= (Q.sum(dim=0, keepdim=True) + 1e-8)
            Q /= (Q.sum(dim=1, keepdim=True) + 1e-8)
        
        Q = Q / (Q.sum(dim=-1, keepdim=True) + 1e-8)
        return Q

    def get_dino_weight(self, current_step):
        if current_step < self.dino_warmup_steps:
            return self.lambda_dino * (current_step / self.dino_warmup_steps)
        return self.lambda_dino

    def compute_dino_loss(self, student_z, full_seqs):
        student_bos = student_z[:, 0, :]
        student_bos = F.normalize(student_bos, dim=-1, p=2)
        
        teacher_temp = 0.04
        student_temp = 0.1
        
        student_prototypes = self.student_dino_head(student_bos) / student_temp
        student_probs = F.log_softmax(student_prototypes, dim=-1)
        
        with torch.no_grad():
            teacher_z = self.teacher_encoder(full_seqs)
            teacher_bos = teacher_z[:, 0, :]
            teacher_bos = F.normalize(teacher_bos, dim=-1, p=2)
            
            teacher_prototypes = self.teacher_dino_head(teacher_bos) / teacher_temp
            teacher_probs = self.sinkhorn_knopp(teacher_prototypes)
        
        dino_loss = -torch.sum(teacher_probs * student_probs, dim=-1).mean()
        return dino_loss

    def compute_generator_losses(self, rec, tdat, tmsk=None):
        with torch.cuda.amp.autocast(enabled=False):
            rec_masked = rec.float()
            tdat_masked = tdat.unsqueeze(1).float() if tdat.dim() == 4 else tdat.float()
           
            if tmsk is not None:
                mask = tmsk.unsqueeze(1)
                rec_masked = rec_masked * mask
                tdat_masked = tdat_masked * mask
           
            mse_loss = F.mse_loss(rec_masked, tdat_masked, reduction='mean')
            wa_loss_val = self.lambda_wa * wa_loss(rec_masked, tdat_masked, self.projvec, denorm=False)
            spatial_loss = mse_loss + wa_loss_val
            
            self.discriminator.eval()
            d_fake = self.discriminator(rec_masked)
            self.discriminator.train()
            adv_loss = self.lambda_adv * F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
            
            recon_loss = spatial_loss + adv_loss
           
            return {
                'mse_loss': mse_loss,
                'wa_loss': wa_loss_val,
                'adv_loss': adv_loss,
                'spatial_loss': spatial_loss,
                'recon_loss': recon_loss
            }
    
    def compute_discriminator_losses(self, rec, tdat, tmsk=None):
        with torch.cuda.amp.autocast(enabled=False):
            rec_masked = rec.float()
            tdat_masked = tdat.unsqueeze(1).float() if tdat.dim() == 4 else tdat.float()
           
            if tmsk is not None:
                mask = tmsk.unsqueeze(1)
                rec_masked = rec_masked * mask
                tdat_masked = tdat_masked * mask
            
            d_real = self.discriminator(tdat_masked)
            d_fake = self.discriminator(rec_masked.detach())
            
            d_loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real) * 0.9)
            d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            d_loss = d_loss_real + d_loss_fake
            
            return {'d_loss': d_loss, 'd_loss_real': d_loss_real, 'd_loss_fake': d_loss_fake}

    def preprocess_sequence(self, seqs):
        bos_padding = torch.zeros((seqs.shape[0], 1), device=seqs.device, dtype=seqs.dtype)
        return torch.cat([bos_padding, seqs], dim=1)

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        N_ACC = 16
        opt_g, opt_d = self.optimizers()

        if batch_idx % N_ACC == 0:
            opt_g.zero_grad()
        if batch_idx % (2 * N_ACC) == 0:
            opt_d.zero_grad()
        
        tdat, masked_seqs, labels, _, full_seqs = batch
        labels = labels.view(-1)
        labels = torch.cat((torch.tensor([0], dtype=torch.int64, device=labels.device), labels))

        masked_seqs = self.preprocess_sequence(masked_seqs)
        full_seqs = self.preprocess_sequence(full_seqs)
        
        z, rec = self.forward(masked_seqs)
        
        d_losses = self.compute_discriminator_losses(rec, tdat, None)
        self.manual_backward(d_losses['d_loss'] / (2 * N_ACC))
        
        if (batch_idx + 1) % (2 * N_ACC) == 0:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=2.0)
            opt_d.step()
        
        mlm_logits = self.mlm_head(z)
        
        assert mlm_logits.shape[0] * mlm_logits.shape[1] == labels.shape[0], \
            f"MLM shape mismatch: {mlm_logits.shape} vs {labels.shape}"
        
        if (labels != -100).any():
            raw_mlm_loss = F.cross_entropy(mlm_logits.view(-1, 8002), labels, reduction='mean')
        else:
            raw_mlm_loss = 0 * torch.mean(mlm_logits)

        dino_loss = self.compute_dino_loss(z, full_seqs)
        dino_weight = self.get_dino_weight(self.global_step)

        g_losses = self.compute_generator_losses(rec, tdat, None)
        raw_recon_loss = g_losses['recon_loss']

        total_g_loss = raw_mlm_loss + dino_weight * dino_loss + raw_recon_loss
        self.manual_backward(total_g_loss / N_ACC)
        
        if (batch_idx + 1) % N_ACC == 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
            opt_g.step()
            self.update_teacher_ema()

        with torch.no_grad():
            student_bos = z[:, 0, :]
            student_bos = F.normalize(student_bos, dim=-1, p=2)
            
            student_temp = 0.1
            student_prototypes = self.student_dino_head(student_bos) / student_temp
            
            student_probs = F.softmax(student_prototypes, dim=-1)
            prototype_entropy = -(student_probs * torch.log(student_probs + 1e-8)).sum(dim=-1).mean()

        self.log('train_raw_mlm_loss', raw_mlm_loss, prog_bar=True, sync_dist=True, batch_size=1)
        self.log('train_dino_loss', dino_loss, prog_bar=True, sync_dist=True, batch_size=1)
        self.log('train_dino_weight', dino_weight, prog_bar=False, sync_dist=True, batch_size=1)
        self.log('prototype_entropy', prototype_entropy, prog_bar=True, sync_dist=True, batch_size=1)
        self.log('train_mse_loss', g_losses['mse_loss'].detach(), prog_bar=True, sync_dist=True, batch_size=1)
        self.log('train_wa_loss', g_losses['wa_loss'].detach(), prog_bar=True, sync_dist=True, batch_size=1)
        self.log('train_adv_loss', g_losses['adv_loss'].detach(), prog_bar=True, sync_dist=True, batch_size=1)
        self.log('train_d_loss', d_losses['d_loss'].detach(), prog_bar=True, sync_dist=True, batch_size=1)
        self.log('train_spatial_loss', g_losses['spatial_loss'].detach(), prog_bar=True, sync_dist=True, batch_size=1)
        self.log('train_loss', total_g_loss, sync_dist=True, batch_size=1)
        self.log('learning_rate', opt_g.param_groups[0]['lr'], prog_bar=True, sync_dist=True, batch_size=1)

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None

        tdat, seqs, _ = batch
        seqs = self.preprocess_sequence(seqs)
        z, rec = self.forward(seqs)

        g_losses = self.compute_generator_losses(rec, tdat, None)
        raw_recon_loss = g_losses['recon_loss']

        self.log('val_mse_loss', g_losses['mse_loss'].detach(), prog_bar=False, sync_dist=True)
        self.log('val_wa_loss', g_losses['wa_loss'].detach(), prog_bar=False, sync_dist=True)
        self.log('val_adv_loss', g_losses['adv_loss'].detach(), prog_bar=False, sync_dist=True)
        self.log('val_spatial_loss', g_losses['spatial_loss'].detach(), prog_bar=False, sync_dist=True)
        self.log('val_recon_loss', raw_recon_loss.detach(), prog_bar=False, sync_dist=True)
        self.log('val_loss', raw_recon_loss.detach(), prog_bar=True, sync_dist=True)

        return raw_recon_loss
        
    def test_step(self, batch, batch_idx):
        tdat, seqs, tmsk = batch
        seqs = self.preprocess_sequence(seqs)
        z, rec = self(seqs)
        return {'embeddings': z.detach(), 'sequences': seqs, 'reconstruction': rec.detach()}

    def predict_step(self, batch, batch_idx):
        batch = self.preprocess_sequence(batch)
        return self(batch, True)

    def configure_optimizers(self):
        g_params = [p for n, p in self.named_parameters() if 'discriminator' not in n and 'teacher' not in n]
        d_params = self.discriminator.parameters()
        
        if self.hparams.optimizer_type.lower() == "adam":
            opt_g = torch.optim.Adam(g_params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
            opt_d = torch.optim.Adam(d_params, lr=self.hparams.d_learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer_type.lower() == "adamw":
            opt_g = torch.optim.AdamW(g_params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
            opt_d = torch.optim.AdamW(d_params, lr=self.hparams.d_learning_rate, weight_decay=self.hparams.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer_type}")
        
        if self.hparams.cyclic_lr:
            sched_g = torch.optim.lr_scheduler.CyclicLR(opt_g, step_size_up=self.hparams.step_size_up, base_lr=self.hparams.learning_rate, max_lr=self.hparams.learning_rate * self.hparams.max_lr_factor, mode='exp_range', gamma=0.9999, cycle_momentum=False)
            return [opt_g, opt_d], [{"scheduler": sched_g, "interval": "step", "frequency": 2}]
        else:
            sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=50, eta_min=1e-8)
            return [opt_g, opt_d], [{"scheduler": sched_g, "interval": "epoch"}]
