import torch
import matplotlib.pyplot as plt
from model.E34REIN import REIN 
import cv2
import numpy as np 
import torch.nn.functional as F


def feature_map_generator(img, feat):
    feat = feat.squeeze(0)
    mean_feature_map = torch.sum(feat, 0) / feat.shape[0]  # Compute mean across channels
    feature_map = mean_feature_map.data.cpu().numpy() 
    print("\n Processed feature maps shape")


    # ----- Prepare original image -----
    orig_img = img[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    orig_img = np.clip(orig_img, 0, 1)

    # ----- Prepare feature map -----
    feature_map_norm = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map) + 1e-8)

    # Resize to match image size if needed
    if feature_map_norm.shape[:2] != orig_img.shape[:2]:
        feature_map_norm = cv2.resize(feature_map_norm, (orig_img.shape[1], orig_img.shape[0]))
    return orig_img, feature_map_norm

def load_img(path, slice_angle=0, device=None):
    """
    Load a grayscale image, resize to (200,200), convert to 3-channel tensor,
    and rotate using PyTorch affine grid (like REIN).
    
    Args:
        path (str): Path to the image.
        slice_angle (float): Rotation angle in degrees (counterclockwise).
        device (torch.device): Torch device (CPU or CUDA).
    Returns:
        torch.Tensor: Rotated image tensor of shape (1, 3, H, W)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load and normalize -----
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (200, 200))
    img = img.astype(np.float32) / 256.0

    # Convert to 3-channel
    img = np.repeat(img[np.newaxis, :, :], 3, axis=0)  # shape: (3, H, W)
    img = np.expand_dims(img, axis=0)  # shape: (1, 3, H, W)

    # Convert to torch tensor
    img = torch.from_numpy(img).to(device)

    # ----- Apply rotation via affine_grid + grid_sample -----
    if slice_angle != 0:
        angle = torch.tensor([slice_angle * np.pi / 180], device=device)  # deg → rad
        batch_size = img.size(0)

        aff = torch.zeros(batch_size, 2, 3, device=device)
        aff[:, 0, 0] = torch.cos(angle)
        aff[:, 0, 1] = torch.sin(angle)
        aff[:, 1, 0] = -torch.sin(angle)
        aff[:, 1, 1] = torch.cos(angle)

        grid = F.affine_grid(aff, torch.Size(img.size()), align_corners=True)
        # torch.rot90(img, k=1, dims=[2,3])
        img = F.grid_sample(img, grid, mode='bicubic', align_corners=True)

    return img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
slice_angle = 90              
# ----- Load images -----
img1 = load_img('../../oord_data/cartesian/Bellmouth_1_resize/1637842717347873.png')
img2 = load_img('../../oord_data/cartesian/Bellmouth_1_resize/1637842717347873.png', slice_angle)
angles = -torch.arange(0,359.00001,360.0/8)/180*torch.pi 
print(img1.shape)
# ----- Load model -----
model = REIN().to(device)
# checkpoint = torch.load(
#     'runs/rerein/model_best.pt',
#     map_location=lambda storage, loc: storage,
#     weights_only=False
# )
# model.load_state_dict(checkpoint['state_dict'])
# model.load_state_dict(checkpoint, strict=False)
model.eval()

# ----- Forward pass -----
with torch.no_grad():
    out1, local_feat1, global_desc1 = model(img1)
    out2, local_feat2, global_desc2 = model(img2)

orig_img1, feature_map_norm1 = feature_map_generator(img1, local_feat1)
orig_img2, feature_map_norm2 = feature_map_generator(img2, local_feat2)
# Convert feature maps to tensors
feature_map_norm1_t = torch.tensor(feature_map_norm1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # shape: (1,1,H,W)
feature_map_norm2_t = torch.tensor(feature_map_norm2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# ----- Rotate feature map 2 back (inverse rotation) -----
batch_size = 1
angle = torch.tensor([slice_angle * np.pi / 180], device=device)  # convert degrees → radians

# Create affine matrix for inverse rotation (negate angle)
aff = torch.zeros(batch_size, 2, 3, device=device)
aff[:, 0, 0] = torch.cos(-angle)
aff[:, 0, 1] = torch.sin(-angle)
aff[:, 1, 0] = -torch.sin(-angle)
aff[:, 1, 1] = torch.cos(-angle)

# Create grid for sampling
grid = F.affine_grid(aff, feature_map_norm2_t.size(), align_corners=True)
# Apply grid sampling to rotate back
feature_map_norm2_rot = F.grid_sample(feature_map_norm2_t, grid, mode='bicubic', align_corners=False)
# feature_map_norm2_rot = torch.rot90(feature_map_norm2_t, k=-1, dims=[2,3])

# Convert back to numpy for visualization
feature_map_norm1 = feature_map_norm1_t.squeeze().cpu().numpy()
feature_map_norm2_rot = feature_map_norm2_rot.squeeze().cpu().numpy()

# ----- Ensure same shape -----
# min_h = min(feature_map_norm1.shape[0], feature_map_norm2_rot.shape[0])
# min_w = min(feature_map_norm1.shape[1], feature_map_norm2_rot.shape[1])
# feature_map_norm1 = feature_map_norm1[:min_h, :min_w]
# feature_map_norm2_rot = feature_map_norm2_rot[:min_h, :min_w]

# ----- Compute difference -----
feature_diff = np.abs(feature_map_norm1 - feature_map_norm2_rot)

# ----- Visualization -----
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

axes[0].imshow(orig_img1)
axes[0].set_title("Original Image 1", fontsize=16)
axes[0].axis("off")

axes[1].imshow(orig_img2)
axes[1].set_title(f"Original Image 2 (rotated {slice_angle}°)", fontsize=16)
axes[1].axis("off")

axes[2].imshow(feature_diff, cmap='magma')
axes[2].set_title("Difference (Feature Map 1 - Rotated Back 2)", fontsize=16)
axes[2].axis("off")

axes[3].imshow(feature_map_norm1, cmap='viridis')
axes[3].set_title("Feature Map 1", fontsize=16)
axes[3].axis("off")

axes[4].imshow(feature_map_norm2_rot, cmap='viridis')
axes[4].set_title("Feature Map 2 (Rotated Back, via grid_sample)", fontsize=16)
axes[4].axis("off")

axes[5].axis("off")

plt.tight_layout()
plt.show()