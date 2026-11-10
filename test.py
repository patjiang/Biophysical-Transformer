import torch

print("Running BioT test.py")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
print(f"Using device: {device} with {dtype}")

# ── SE3CNN ──
print("Testing SE3CNN...")
from se3cnn.image.convolution import SE3Convolution

size = 16
scalar_field = torch.randn(1, 1, size, size, size, device=device, dtype=dtype)
conv = SE3Convolution(
    Rs_in=[(1, 0)], Rs_out=[(1, 0)], size=5, padding=2
).to(device).to(dtype)
out = conv(scalar_field)
assert out.shape == scalar_field.shape
print("SE3CNN OK")

# ── Lightning module ──
print("Testing BioTLightning...")
from src.model import BioTLightning

model = BioTLightning().to(device).to(dtype)
model.eval()

B, L = 2, 32
seqs = torch.randint(0, 8002, (B, L), device=device)

# encoder forward
z, rec = model(seqs)
assert z.shape[0] == B
assert z.shape[2] == model.student_encoder.output_dim
assert rec.ndim == 5
print(f"  forward OK  z={tuple(z.shape)}  rec={tuple(rec.shape)}")

# interaction-block forward (what the eval scripts use)
zr = model.forward_int(seqs, pred=True)
assert zr.shape[0] == B
assert zr.shape[1] == L - 1  # BOS stripped
print(f"  forward_int OK  zr={tuple(zr.shape)}")

print("\nAll tests passed!")
