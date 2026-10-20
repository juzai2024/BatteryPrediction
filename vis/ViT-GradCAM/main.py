import timm  # PyTorch image models library
import torch  # PyTorch deep learning library
import sys
sys.path.append("/path/to/mae")
from models_vit import vit_batter_patch16 as vit
import scipy.io as sio
from torchvision.models import vgg19, resnet18  # Pretrained models from torchvision
from gradcam import GradCam  # Grad-CAM implementation
import numpy as np  # NumPy for numerical operations
import cv2  # OpenCV for image processing
from matplotlib import pyplot as plt
import os
# List all Vision Transformer models available in the timm library
# timm.list_models('vit_*')

# Function to prepare input image for the model
def prepare_input(image, mean=None, std=None):
    image = image.copy()  # Copy the image to avoid modifying the original
    if mean is not None and std is not None:
        # Normalize using provided mean and std
        image -= mean
        image /= std
    else:
        # Normalize the image using the mean and standard deviation
        means = image.mean()
        stds = image.std()
        image -= means
        image /= stds

    # Transpose the image to match the model's expected input format (C, H, W)
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    image = image[np.newaxis, ...]  # Add batch dimension

    return torch.tensor(image, requires_grad=True).float().cuda()  # Convert to PyTorch tensor

# 加载预训练ViT
def prepapre_model(ckpt):
    checkpoint = torch.load(ckpt, map_location='cuda')
    vit_model = vit(img_size=128, in_chans=6)
    vit_model.load_state_dict(checkpoint['model'], strict=False)
    return vit_model.cuda() if torch.cuda.is_available() else vit_model.cpu()

# Function to generate a Grad-CAM heatmap
def gen_cam(image, mask):
    # Create a heatmap from the Grad-CAM mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    image = cv2.applyColorMap(np.uint8(255 * image), cv2.COLORMAP_JET)
    image = np.float32(image) / 255

    # Superimpose the heatmap on the original image
    cam = (1 - 0.5) * heatmap + 0.5 * image
    cam = cam / np.max(cam)  # Normalize the result
    return np.uint8(255 * cam)  # Convert to 8-bit image

if __name__ == '__main__':
    # Load and preprocess the input image
    type_b = "charge"
    # data_path = "/path/to/mae/data/fixed_data_final"
    # data_files = os.listdir(data_path)
    # nmc_files = [i for i in data_files if f'NMC#2_{type_b}' in i]
    # nmc_files = [os.path.join(data_path, i) for i in nmc_files]
    nmc_files = ["/path/to/mae/data/history/Battery_selection_origin/NMC#2_charge_wo_5.mat"]
    global_mean = 0
    global_std = 0
    all_mat = []
    for file_p in nmc_files:
        mat = sio.loadmat(file_p)
        img = mat['data'][:,:,1:]

        all_mat.append(img)
    all_mat = np.array(all_mat)
    global_mean = all_mat.mean()
    global_std = all_mat.std()
    # print(global_mean, global_std)
    
    model = prepapre_model("/path/to/mae/mae_weight_out/mae_train_final/checkpoint-final.pth")

    # Candidate target layers to compare
    target_layers = {
        'norm1_last':  model.blocks[-1].norm1,   # original (baseline)
        'norm2_last':  model.blocks[-1].norm2,
        'block_last':  model.blocks[-1],
        'block_prev':  model.blocks[-2],
        'final_norm':  model.norm,
    }

    # Diagnostic: what does the model output look like?
    with torch.no_grad():
        dummy = prepare_input(all_mat[0])
        out = model(dummy)
        print(f"model output shape: {tuple(out.shape)}; values: {out.cpu().numpy().ravel()[:10]}")

    norm = plt.Normalize(vmin=-3.14, vmax=3.14)
    crop_y, crop_x = slice(38, 108), slice(32, 90)
    for file_p in nmc_files:
        mat = sio.loadmat(file_p)
        img = mat['data'][:,:,1:]
        ah = mat['ah']
        bg = img[crop_y, crop_x, 3]

        for tl_name, tl in target_layers.items():
            grad_cam = GradCam(model, tl)
            inputs = prepare_input(img)
            raw_cam = grad_cam(inputs)  # signed, (128, 128)
            print(f"[{tl_name}] ah={ah[0][0]} raw cam range "
                  f"[{raw_cam.min():.4e}, {raw_cam.max():.4e}] "
                  f"(spread={raw_cam.max()-raw_cam.min():.4e})")

            def _save(mask_crop, cmap, vmin, vmax, kind):
                fig, ax = plt.subplots()
                ax.imshow(bg, cmap='gray', norm=norm)
                im = ax.imshow(mask_crop, alpha=0.5, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.savefig(f'vit_gradcam_{tl_name}_{kind}_{type_b}_{ah[0][0]}.png',
                            bbox_inches='tight', dpi=300)
                plt.close(fig)

            # Normalized ReLU CAM (standard display)
            relu_full = np.maximum(raw_cam, 0)
            mA = (relu_full - relu_full.min()) / (relu_full.max() - relu_full.min() + 1e-8)
            _save(mA[crop_y, crop_x], 'jet', 0, 1, 'relu')

            # Signed CAM (diagnostic — shows sign + absolute magnitude)
            signed = raw_cam[crop_y, crop_x]
            v = float(np.abs(signed).max() + 1e-8)
            _save(signed, 'bwr', -v, v, 'signed')

