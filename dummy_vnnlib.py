import numpy as np
from pathlib import Path

# Paths you choose
out_dir = Path("tmp_sdp_test")
out_dir.mkdir(parents=True, exist_ok=True)

vnnlib_path = out_dir / "sample_0.vnnlib"
meta_path = out_dir / "sample_0.npz"  # same stem, .npz suffix

# 1) Create a dummy vnnlib file (SDP-CROWN won't parse it itself)
vnnlib_path.write_text("; dummy VNNLIB for SDP-CROWN manual test\n", encoding="utf-8")

# 2) Create the NPZ sidecar as Verona would
# Replace these with a real image/label from your dataset.
image = np.random.rand(3, 32, 32).astype(np.float32)  # or flattened, SDP-CROWN reshapes
image_class = 1  # true label
epsilon = 0.0    # or your desired radius

np.savez_compressed(
    meta_path,
    image=image,
    image_class=image_class,
    epsilon=epsilon,
)

print("Created:")
print(f"  vnnlib: {vnnlib_path}")
print(f"  sidecar: {meta_path}")