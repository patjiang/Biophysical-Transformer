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
    PatchDiscriminator3D_sm,
)
from .losses import wa_loss

def check_nan(tensor, name):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"NaN/Inf in {name}: max={tensor.max()}, min={tensor.min()}")
        
def compute_and_initialize_biases(
    train_loader,
    mlm_head,
    convout_module,
    vocab_size=8003,
    device="cuda"
):
    print("Computing unigram + reconstruction biases in single pass...")
    
    counts = torch.zeros(vocab_size, dtype=torch.long)
    total_tokens = 0
    
    running_mean = 0.0
    total_voxels = 0
    
    for batch in tqdm(train_loader, desc="Unified Bias Init"):
        if batch is None:
            continue

        tdat = batch[0]
        if not torch.isfinite(tdat).all():
            nan_count = (~torch.isfinite(tdat)).sum().item()
            print(f"Warning: Found {nan_count} non-finite voxels. Skipping them.")
            print(f"to remove: {tdat[-1]}")
        batch_count = tdat.numel()
        batch_mean = tdat.mean().item()
        new_total = total_voxels + batch_count
        running_mean += (batch_mean - running_mean) * (batch_count / new_total)
        total_voxels = new_total
        if len(batch) >= 5:
            full_seqs = batch[4]
            tokens = full_seqs.view(-1)
            valid_mask = tokens < (vocab_size - 3)
            valid_tokens = tokens[valid_mask]
            if valid_tokens.numel() > 0:
                counts += torch.bincount(
                    valid_tokens.cpu(), minlength=vocab_size
                )
                total_tokens += valid_tokens.numel()

    probs = counts.float() / (total_tokens + 1e-8)
    log_probs = torch.log(probs + 1e-8)

    with torch.no_grad():
        mlm_head.bias.data.copy_(log_probs.to(device))

    entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
    #print(f"Total potential: {total_potential:.4f}")


    with torch.no_grad():
        for layer in convout_module:
            if isinstance(layer, nn.Conv3d) and layer.bias is not None:
                layer.bias.fill_(running_mean)

    print(f"MLM baseline entropy: {entropy:.4f}")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Unique tokens with counts > 0: {(counts > 0).sum().item()}")
    print(f"Reconstruction mean potential: {running_mean:.4f}")
    print(f"Total voxels processed: {total_voxels}")
    #print("Throw error", err)

    return {
        "mlm_entropy": entropy,
        "recon_mean_bias": running_mean,
        "total_tokens": total_tokens,
        "total_voxels": total_voxels,
    }


class SE3PLMLightning(pl.LightningModule):
    def __init__(self, 
                 learning_rate: float = 1e-3,
                 d_learning_rate: float = 5e-6,
                 weight_decay: float = 1e-5,
                 optimizer_type: str = "adam",
                 cyclic_lr: bool = True,
                 max_lr_factor: float = 100.0,
                 step_size_up: int = 1000,
                 lambda_wa: float = 5.0,
                 lambda_adv: float = 2.0,
                 lambda_dino: float = 0.1,
                 num_prototypes: int = 128,
                 teacher_momentum: float = 0.999,
                 dino_warmup_steps: int = 60000,
                 vocab_size: int = 8003,
                 queue_length=128,
                 **kwargs):
        super().__init__()
        self.automatic_optimization = False
        
        self.save_hyperparameters()
        self.lambda_wa = lambda_wa
        self.lambda_adv = lambda_adv
        self.lambda_dino = lambda_dino
        self.teacher_momentum = teacher_momentum
        self.dino_warmup_steps = dino_warmup_steps
        
        self.student_encoder = TransformerStackEnc(input_dim=64, output_dim=64, num_layers=5, vocab_size=vocab_size)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.teacher_encoder.requires_grad_(False)
        """
        self.mlm_head = self.mlm_head = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, vocab_size)
        )
        """
        #Weight tying MLM head
        self.mlm_ln = nn.LayerNorm(64)
        self.mlm_out = nn.Linear(64, 8003, bias=True)
        del self.mlm_out.weight
        self.mlm_out.weight = self.student_encoder.embedding.weight
        self.mlm_head = nn.Sequential(
            self.mlm_ln,
            self.mlm_out
        )
        self.student_dino_head = nn.Linear(64, num_prototypes)
        self.teacher_dino_head = copy.deepcopy(self.student_dino_head)
        self.teacher_dino_head.requires_grad_(False)
        self.num_prototypes = num_prototypes
        self.i = InteractionTokenBlock()
        self.b = VoxelizeTrunk()
        self.step = 0#50001
        self.discriminator = PatchDiscriminator3D(in_channels=1)
        #self.register_buffer("prototype_center", torch.zeros(self.num_prototypes))
        self.register_buffer("queue", torch.zeros(queue_length, num_prototypes))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_length = queue_length
        
        self.center_momentum = 0.999
        self.student_temp = 0.15
        self.teacher_temp = 0.07
        self.loss_log_vars = nn.Parameter(torch.tensor([2.2, 0, 0]))
        
    def on_train_start(self):
        if not hasattr(self, '_biases_initialized'):
            self._biases_initialized = False
        if self._biases_initialized:
            print("\n" + "="*60)
            print("INITIALIZING BIASES FROM TRAINING DATA")
            print("="*60 + "\n")
            train_loader = self.trainer.train_dataloader
            compute_and_initialize_biases(
                train_loader=train_loader,
                mlm_head=self.mlm_head,
                convout_module = self.b.convout,
                vocab_size=8003,
                device=self.device
            )
            print("\n" + "="*60)
            print("BIAS INITIALIZATION COMPLETE")
            print("="*60 + "\n")
            self._biases_initialized = True
            
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
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.queue_length:
            ptr = 0
            
        self.queue[ptr:ptr + batch_size] = keys
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_length
        
    @torch.no_grad()
    def update_teacher_ema(self):
        for student_p, teacher_p in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            teacher_p.data.mul_(self.teacher_momentum).add_(student_p.data, alpha=1 - self.teacher_momentum)
        for student_p, teacher_p in zip(self.student_dino_head.parameters(), self.teacher_dino_head.parameters()):
            teacher_p.data.mul_(self.teacher_momentum).add_(student_p.data, alpha=1 - self.teacher_momentum)

    def compute_dino_loss(self, student_z, full_seqs):
        student_bos = F.normalize(student_z[:, 0, :], dim=-1, p=2)
        student_logits = self.student_dino_head(student_bos) / self.student_temp
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        
        with torch.no_grad():
            teacher_z = self.teacher_encoder(full_seqs)
            teacher_bos = F.normalize(teacher_z[:, 0, :], dim=-1, p=2)
            teacher_weight = F.normalize(self.teacher_dino_head.weight, dim=1)
            teacher_logits = F.linear(teacher_bos, teacher_weight)
            ptr = int(self.queue_ptr)
            if ptr > 0:
                combined = torch.cat([teacher_logits, self.queue[:ptr]], dim=0)
            else:
                combined = teacher_logits
            teacher_probs_all = self.sinkhorn_knopp(combined)
            teacher_probs = teacher_probs_all[:teacher_logits.shape[0]]
            teacher_probs = teacher_probs.clamp(min=1e-7)
            teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)
            self._dequeue_and_enqueue(teacher_logits.detach())
        #print("student_logits range:", student_logits.min().item(), student_logits.max().item())
        #print("teacher_probs range:", teacher_probs.min().item(), teacher_probs.max().item())
        #print("student_log_probs range:", student_log_probs.min().item(), student_log_probs.max().item())
        loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
        if not torch.isfinite(loss):
            return torch.tensor(0.0, device=loss.device, requires_grad=True)
        return loss

    def sinkhorn_knopp(self, teacher_output, n_iter=3):
        teacher_output = teacher_output.float()
        logits = teacher_output / self.teacher_temp
        
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        Q = torch.exp(logits.clamp(min=-20.0, max=20.0))
        B, K = Q.shape
        Q = Q / (Q.sum() + 1e-8)
        
        for _ in range(n_iter):
            Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)
            Q = Q / K
            Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-8)
            Q = Q / B
            
        Q = Q * B
        return Q

    def compute_structure_loss(self, rec, tdat, tmsk=None):
        def soft_dice_loss(pred, target, eps=1e-6):
            p = pred.reshape(pred.size(0), -1)
            t = target.reshape(target.size(0), -1)
            
            intersection = (p * t).sum(dim=1)
            union = p.sum(dim=1) + t.sum(dim=1)
            
            dice = (2.0 * intersection + eps) / (union + eps)
            return 1.0 - dice.mean()

        with torch.cuda.amp.autocast(enabled=False):
            rec_f = rec.float()
            tdat_f = tdat.unsqueeze(1).float() if tdat.dim() == 4 else tdat.float()
            
            if tmsk is not None:
                mask = tmsk.unsqueeze(1).float()
                rec_f = rec_f * mask
                tdat_f = tdat_f * mask
                
            pred_pos, target_pos = torch.relu(rec_f), torch.relu(tdat_f)
            pred_neg, target_neg = torch.relu(-rec_f), torch.relu(-tdat_f)

            pos_loss = soft_dice_loss(pred_pos, target_pos)
            neg_loss = soft_dice_loss(pred_neg, target_neg)
            
            raw_total_dice = (pos_loss + neg_loss) / 2.0

            dice_weight = 0.01

            weighted_str_loss = raw_total_dice * dice_weight

            return {
                'str_loss': weighted_str_loss,
                'raw_str_loss': raw_total_dice,
                'pos_dice': pos_loss, 
                'neg_dice': neg_loss,
                'dice_weight': dice_weight
            }

    def compute_adaptive_loss(self, raw_loss, index):
        log_var = self.loss_log_vars[index]
        precision = torch.exp(-log_var.clamp(min=-4.0, max=4.0))
        weighted_loss = 0.5 * (precision * raw_loss + log_var)
        return weighted_loss, precision

    def compute_generator_losses(self, rec, tdat, tmsk=None):
        with torch.cuda.amp.autocast(enabled=False):
            rec_masked = rec.float()
            tdat_masked = tdat.unsqueeze(1).float() if tdat.dim() == 4 else tdat.float()
            
            if tmsk is not None:
                mask = tmsk.unsqueeze(1)
                rec_masked = rec_masked * mask
                tdat_masked = tdat_masked * mask
            mae_loss = F.l1_loss(rec_masked, tdat_masked, reduction='mean')
            #mse_loss = F.mse_loss(rec_masked, tdat_masked, reduction='mean')
            wa_loss_val = self.lambda_wa * wa_loss(rec_masked, tdat_masked, denorm=False)
            spatial_loss = mae_loss + wa_loss_val
            
            if(self.step > 50000):
                self.discriminator.eval()
                d_fake = self.discriminator(rec_masked)
                self.discriminator.train()
                adv_loss = self.lambda_adv * (-d_fake).mean()
            else:
                adv_loss = 0.0
            
            recon_loss = spatial_loss# + adv_loss
            
            return {
                'mae_loss': mae_loss,
                'wa_loss': wa_loss_val,
                'adv_loss': adv_loss,
                'spatial_loss': spatial_loss,
                'recon_loss': recon_loss
            }
            
    def compute_discriminator_losses(self, rec, tdat, tmsk=None, epsilon_drift = 0.001):
        with torch.cuda.amp.autocast(enabled=False):
            rec_masked = rec.float()
            tdat_masked = tdat.unsqueeze(1).float() if tdat.dim() == 4 else tdat.float()
            noise_std = max(0.001, 0.1 - (self.step / 200000) * 0.1)
            if tmsk is not None:
                mask = tmsk.unsqueeze(1)
                rec_masked = rec_masked * mask
                tdat_masked = tdat_masked * mask
            
            tdat_masked.requires_grad_(True)
            if noise_std > 0:
                real_noise = torch.randn_like(tdat_masked) * noise_std
                fake_noise = torch.randn_like(rec_masked) * noise_std
                
                d_real = self.discriminator(tdat_masked + real_noise)
                d_fake = self.discriminator(rec_masked.detach() + fake_noise)
            else:
                d_real = self.discriminator(tdat_masked)
                d_fake = self.discriminator(rec_masked.detach())
                
            d_real = torch.clamp(d_real, min=-1000.0, max=1000.0)
            d_fake = torch.clamp(d_fake, min=-1000.0, max=1000.0)
        
            if torch.isnan(d_real).any() or torch.isinf(d_real).any():
                print(f"\nFATAL: d_real contains NaN/Inf! Max: {d_real.max()}, Min: {d_real.min()}")
                for name, p in self.discriminator.named_parameters():
                    if not torch.isfinite(p).all():
                        print('NaN in param: ', name)
                torch.save({'rec': rec, 'tdat': tdat}, 'nan_dump_inputs.pt')
                raise RuntimeError("d_real exploded.")
                
            if torch.isnan(d_fake).any() or torch.isinf(d_fake).any():
                print(f"\nFATAL: d_fake contains NaN/Inf! Max: {d_fake.max()}, Min: {d_fake.min()}")
                for name, p in self.discriminator.named_parameters():
                    if not torch.isfinite(p).all():
                        print('NaN in param: ', name)
                torch.save({'rec': rec, 'tdat': tdat}, 'nan_dump_inputs.pt')
                raise RuntimeError("d_fake exploded.")
            
            grad_real = torch.autograd.grad(
                outputs=d_real.sum(), 
                inputs=tdat_masked, 
                create_graph=True, 
                retain_graph=True
            )[0]
            
            r1_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
            
            gamma = 10.0
            d_loss_real = F.softplus(1.0-d_real).mean()
            d_loss_fake = F.softplus(1.0+d_fake).mean()
            
            d_loss = d_loss_real + d_loss_fake + (gamma / 2) * r1_penalty
            
            return {
                'd_loss': d_loss, 
                'd_loss_real': d_loss_real, 
                'd_loss_fake': d_loss_fake,
                'r1_penalty': r1_penalty
            }
    
            
    def preprocess_sequence(self, seqs):
        bos_padding = torch.full((seqs.shape[0], 1), fill_value=8002, device=seqs.device, dtype=seqs.dtype)
        return torch.cat([bos_padding, seqs], dim=1)
     
    def get_weight(self, current_step, max_steps=10000):
        base_weight = 0.001
        scale = min(current_step / max_steps, 1.0)
        return base_weight + (0.999 * scale)

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        N_ACC = 16
        opts = self.optimizers()
        opt_g = opts[0]
        opt_d = opts[1]
        if batch_idx % N_ACC == 0:
            opt_g.zero_grad()
        if batch_idx % (N_ACC // 2) == 0:
            opt_d.zero_grad()

        tdat, masked_seqs, labels, _, full_seqs = batch[0]
        labels = labels.view(-1)
        labels = torch.cat((torch.tensor([-100], dtype=torch.int64, device=labels.device), labels))

        masked_seqs = self.preprocess_sequence(masked_seqs)
        full_seqs = self.preprocess_sequence(full_seqs)
        z, rec = self.forward(masked_seqs)
        
        if(self.step > 30000):
            d_losses = self.compute_discriminator_losses(rec, tdat, None)
            #if torch.isnan(val).any() or torch.isinf(val).any():
            #        print(f"\nFATAL: Discriminator '{key}' became NaN/Inf!")
            #        print(f"Step: {self.step}, Batch Index: {batch_idx}")
            #        torch.save(d_losses, 'nan_dump_losses.pt')
            #        raise RuntimeError(f"Training halted due to NaN in {key}")
            self.manual_backward(d_losses['d_loss'] / (N_ACC // 2))

            if (batch_idx + 1) % (N_ACC // 2) == 0:
                self.clip_gradients(opt_d, gradient_clip_val=2.0, gradient_clip_algorithm="norm")
                opt_d.step()
            self.log('train_full_d_loss', d_losses['d_loss'].detach(), sync_dist=True)
            self.log('train_real_d_loss', d_losses['d_loss_real'].detach(), sync_dist=True)
            self.log('train_fake_d_loss', d_losses['d_loss_fake'].detach(), sync_dist=True)
            self.log('r1_penalty', d_losses['r1_penalty'].detach(), sync_dist=True)
        else:
            self.log('train_full_d_loss', 0.0, sync_dist=True, batch_size=1)
            self.log('train_real_d_loss', 0.0, sync_dist=True, batch_size=1)
            self.log('train_fake_d_loss', 0.0, sync_dist=True, batch_size=1)
            self.log('r1_penalty', 0.0, sync_dist=True)
                
        mlm_logits = self.mlm_head(z)
        if (labels[1:] != -100).any():
            raw_mlm_loss = F.cross_entropy(mlm_logits.view(-1, 8003).float(), labels, reduction='mean')
        else:
            raw_mlm_loss = 0 * torch.mean(mlm_logits)

        dino_loss = self.compute_dino_loss(z, full_seqs)

        g_losses = self.compute_generator_losses(rec, tdat, None)
        if(self.step > 50000):
            self.log('train_adv_loss', g_losses['adv_loss'].detach(), sync_dist=True, batch_size=1)
        else:
            self.log('train_adv_loss', g_losses['adv_loss'], sync_dist=True, batch_size=1)
        raw_recon_loss = g_losses['recon_loss']
        
        str_loss = self.compute_structure_loss(rec, tdat, None)
        structure_loss = str_loss['str_loss']
        
        final_mlm, prec0 = self.compute_adaptive_loss(raw_mlm_loss, 0)
        final_dino, prec1 = dino_loss, 1.0
        final_recon, prec2 = raw_recon_loss, 1.0
        

        total_g_loss = (5 * final_mlm) + final_dino + final_recon + structure_loss + g_losses['adv_loss']

        check_nan(z, "encoder output z")
        check_nan(rec, "reconstruction rec")
        check_nan(raw_mlm_loss, "mlm_loss")
        check_nan(dino_loss, "dino_loss")
        check_nan(g_losses['mae_loss'], "mae_loss")
        check_nan(g_losses['wa_loss'], "wa_loss")
        check_nan(structure_loss, "structure_loss")
        check_nan(total_g_loss, "total_g_loss")

        self.manual_backward(total_g_loss / N_ACC)
        if (batch_idx + 1) % N_ACC == 0:
            mlm_params = list(self.mlm_head.parameters()) + list(self.student_encoder.parameters())
            b_parm = list(self.b.parameters())
            i_parm = list(self.i.parameters())
            torch.nn.utils.clip_grad_norm_([self.loss_log_vars], max_norm=20.0)
            torch.nn.utils.clip_grad_norm_(mlm_params, max_norm=20.0)
            torch.nn.utils.clip_grad_norm_(b_parm, max_norm=4.0)
            torch.nn.utils.clip_grad_norm_(i_parm, max_norm=12.0)

            opt_g.step()
            self.update_teacher_ema()
            self.loss_log_vars.data.clamp_(-2.0, 4.0)
            sched = self.lr_schedulers()
            if sched is not None:
                sched.step()

        with torch.no_grad():
            student_bos = F.normalize(z[:, 0, :], dim=-1, p=2)
            student_logits = self.student_dino_head(student_bos)
            student_probs = F.softmax(student_logits, dim=-1)
            prototype_entropy = -(student_probs * torch.log(student_probs + 1e-8)).sum(dim=-1).mean()
        self.log('train_loss', total_g_loss, sync_dist=True, batch_size=tdat.shape[0])
        self.log('train_raw_mlm_loss', raw_mlm_loss.detach(), sync_dist=True, batch_size=1)
        self.log('train_dino_loss', dino_loss.detach(), sync_dist=True, batch_size=1)
        self.log('train_mae_loss', g_losses['mae_loss'].detach(), sync_dist=True, batch_size=1)
        self.log('train_wa_loss', g_losses['wa_loss'].detach(), sync_dist=True, batch_size=1)
        self.log('train_spatial_loss', g_losses['spatial_loss'].detach(), sync_dist=True, batch_size=1)
        self.log('prototype_entropy', prototype_entropy, sync_dist=True, batch_size=1)
        self.log('prec_mlm', prec0.detach(), sync_dist=True, batch_size=1)
        self.log('structure_loss', str_loss['raw_str_loss'].detach(), sync_dist=True, batch_size=1)
        self.log('p_dice', str_loss['pos_dice'].detach(), sync_dist=True, batch_size=1)
        self.log('b_dice', str_loss['neg_dice'].detach(), sync_dist=True, batch_size=1)
        self.step += 1


    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None

        tdat, seqs, _ = batch[0]
        seqs = self.preprocess_sequence(seqs)
        z, rec = self.forward(seqs)

        g_losses = self.compute_generator_losses(rec, tdat, None)
        raw_recon_loss = g_losses['recon_loss']
        
        weighted_recon_loss, _ = self.compute_adaptive_loss(raw_recon_loss, 2)
        if(self.step > 50000):
            self.log('val_adv_loss', g_losses['adv_loss'].detach(), sync_dist=True)
        else:
            self.log('val_adv_loss', g_losses['adv_loss'], prog_bar=False, sync_dist=True)
        self.log('val_mae_loss', g_losses['mae_loss'].detach(), prog_bar=False, sync_dist=True)
        self.log('val_wa_loss', g_losses['wa_loss'].detach(), prog_bar=False, sync_dist=True)
        self.log('val_spatial_loss', g_losses['spatial_loss'].detach(), prog_bar=False, sync_dist=True)
        self.log('val_recon_loss', raw_recon_loss.detach(), prog_bar=False, sync_dist=True)
        self.log('val_loss', weighted_recon_loss.detach(), prog_bar=True, sync_dist=True, batch_size=tdat.shape[0])

        return weighted_recon_loss
     
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
            opt_g = torch.optim.Adam(g_params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, betas=(0.0, 0.99))
            opt_d = torch.optim.Adam(d_params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, betas=(0.0, 0.99), eps=1e-4)
        elif self.hparams.optimizer_type.lower() == "adamw":
            opt_g = torch.optim.AdamW(g_params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, betas=(0.0, 0.99))
            opt_d = torch.optim.AdamW(d_params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, betas=(0.0, 0.99), eps=1e-4)
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer_type}")
        
        if self.hparams.cyclic_lr:
            sched_g = torch.optim.lr_scheduler.CyclicLR(
                opt_g, 
                step_size_up=self.hparams.step_size_up, 
                base_lr=self.hparams.learning_rate, 
                max_lr=self.hparams.learning_rate * self.hparams.max_lr_factor, 
                mode='exp_range', 
                gamma=0.9999, 
                cycle_momentum=False
            )
            return [opt_g, opt_d], [{"scheduler": sched_g, "interval": "step"}]
        else:
            sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=50, eta_min=1e-8)
            return [opt_g, opt_d], [{"scheduler": sched_g, "interval": "epoch"}]
