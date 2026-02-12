import torch

print("Running EBPT test.py")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Testing SE3CNN...")

from se3cnn.image.convolution import SE3Convolution

size = 16
scalar_field = torch.randn(1, 1, size, size, size).to(device)

conv = SE3Convolution(
    Rs_in=[(1, 0)],
    Rs_out=[(1, 0)],
    size=5,
    padding=2
).to(device)

out = conv(scalar_field)
assert out.shape == scalar_field.shape
print("SE3CNN OK")

print("Testing module imports...")

from EBPT.modules import (
    TransformerStackEnc,
    InteractionTokenBlock,
    VoxelizeTrunk,
    PatchDiscriminator3D,
)

from EBPT.losses import wa_loss

print("Module imports OK")

print("Testing TransformerStackEnc...")

B, L = 2, 32
seqs = torch.randint(0, 8002, (B, L), device=device)

encoder = TransformerStackEnc(
    input_dim=128,
    output_dim=64,
    num_layers=3,
    vocab_size=8002
).to(device)

with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    z = encoder(seqs)
assert z.shape == (B, L, 64)
print("Encoder forward OK")

print("Testing VoxelizeTrunk...")

interaction = InteractionTokenBlock().to(device)
trunk = VoxelizeTrunk().to(device)

z_flat = z[:, 1:, :].reshape(-1, 1, 64)
z_flat = interaction(z_flat)

bos = z[:, 0, :].repeat_interleave(z.shape[1] - 1, dim=0).unsqueeze(1)

vox = trunk(z_flat, bos)
assert vox.ndim == 5 and vox.shape[1] == 1
print("Voxel trunk OK")

print("Testing PatchDiscriminator3D...")

disc = PatchDiscriminator3D().to(device)
d_out = disc(vox)
assert d_out.numel() > 0
print("Discriminator OK")

print("Testing Wasserstein loss...")

projvec = torch.randn(64, 64, device=device)
loss_val = wa_loss(vox, vox.detach(), projvec, denorm=False)
assert torch.isfinite(loss_val)
print("WA loss OK")


print("Testing Lightning module...")

from Biotransformer.EBPT import EBPTLightning

model = EBTLightning().to(device)
model.eval()

seqs = torch.randint(0, 8002, (2, 32), device=device)
z, rec = model(seqs)

assert z.shape[0] == 2
assert rec.ndim == 5
print("Lightning forward OK")

print("\nAll tests passed successfully!")
