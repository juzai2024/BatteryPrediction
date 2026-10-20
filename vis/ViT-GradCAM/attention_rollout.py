import sys
sys.path.append("/path/to/mae")

import cv2
import numpy as np
import scipy.io as sio
import torch
from matplotlib import pyplot as plt
from models_vit import vit_batter_patch16 as vit


def prepare_input(image):
    image = image.copy().astype(np.float32)
    image -= image.mean()
    image /= image.std()
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))[np.newaxis, ...]
    return torch.tensor(image).float().cuda()


def prepare_model(ckpt):
    checkpoint = torch.load(ckpt, map_location='cuda')
    m = vit(img_size=128, in_chans=6)
    m.load_state_dict(checkpoint['model'], strict=False)
    return m.cuda().eval()


class AttentionRollout:
    """Hook every block's attn_drop to capture softmaxed attention, then roll out.

    Follows Abnar & Zuidema (2020): for each layer l take A_l (fused across heads),
    form M_l = (A_l + I)/2 renormalized (residual + attention), then
    R = M_L @ M_{L-1} @ ... @ M_1.  Row 0, columns 1: gives CLS -> patches.
    """

    def __init__(self, model):
        self.model = model
        self.attns = []
        self.handles = []
        for blk in model.blocks:
            self.handles.append(
                blk.attn.attn_drop.register_forward_hook(self._hook)
            )

    def _hook(self, module, inp, out):
        self.attns.append(out.detach().cpu())

    @torch.no_grad()
    def __call__(self, x, head_fusion='mean', discard_ratio=0.0):
        self.attns = []
        self.model(x)
        if not self.attns:
            raise RuntimeError(
                "No attention matrices captured. timm may be using fused SDPA; "
                "check blk.attn.fused_attn or monkey-patch attn to store the matrix."
            )
        return self._rollout(head_fusion, discard_ratio)

    def _rollout(self, head_fusion, discard_ratio):
        N = self.attns[0].size(-1)
        result = torch.eye(N)
        for a in self.attns:
            a = a[0]  # (heads, N, N), drop batch
            if head_fusion == 'mean':
                fused = a.mean(dim=0)
            elif head_fusion == 'max':
                fused = a.max(dim=0).values
            elif head_fusion == 'min':
                fused = a.min(dim=0).values
            else:
                raise ValueError(head_fusion)

            if discard_ratio > 0:
                flat = fused.flatten()
                k = int(flat.numel() * discard_ratio)
                if k > 0:
                    thresh = flat.kthvalue(k).values
                    fused = torch.where(fused <= thresh, torch.zeros_like(fused), fused)

            M = (fused + torch.eye(N)) / 2
            M = M / M.sum(dim=-1, keepdim=True)
            result = M @ result

        cls_to_patches = result[0, 1:].numpy()
        side = int(np.sqrt(cls_to_patches.size))
        return cls_to_patches.reshape(side, side)

    def close(self):
        for h in self.handles:
            h.remove()


if __name__ == '__main__':
    import glob
    import os

    data_dir = "/path/to/mae/data/history/Battery_selection_origin"
    model = prepare_model(
        "/path/to/mae/mae_weight_out/mae_train_final/checkpoint-final.pth"
    )
    assert getattr(model.blocks[0].attn, 'fused_attn', None) in (None, False), \
        "timm fused attention bypasses the attn_drop hook; need to monkey-patch."

    rollout = AttentionRollout(model)
    bg_norm = plt.Normalize(vmin=-3.14, vmax=3.14)
    crop_y, crop_x = slice(38, 108), slice(32, 90)
    out_dir = "rollout"
    os.makedirs(out_dir, exist_ok=True)

    for type_b in ("charge", "discharge"):
        pattern = os.path.join(data_dir, f"NMC#2_{type_b}_wo_*.mat")
        files = sorted(glob.glob(pattern), key=lambda p: int(p.rsplit('_', 1)[-1].split('.')[0]))
        for file_p in files:
            mat = sio.loadmat(file_p)
            img = mat['data'][:, :, 1:]
            ah = float(mat['ah'][0][0])
            x = prepare_input(img)
            bg = img[crop_y, crop_x, 3]

            mask8 = rollout(x, head_fusion='mean')
            mask128 = cv2.resize(mask8, (128, 128), interpolation=cv2.INTER_CUBIC)
            mask128 = mask128 / (mask128.max() + 1e-12)
            print(f"[{type_b} ah={ah:.4f}] 8x8 range "
                  f"[{mask8.min():.4e}, {mask8.max():.4e}] "
                  f"argmax={np.unravel_index(mask8.argmax(), mask8.shape)}")

            def _plot(with_cbar):
                fig, ax = plt.subplots()
                ax.imshow(bg, cmap='gray', norm=bg_norm)
                im = ax.imshow(mask128[crop_y, crop_x], alpha=0.5, cmap='jet',
                               vmin=0, vmax=1)
                ax.axis('off')
                if with_cbar:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    fig.savefig(os.path.join(out_dir, 'vit_rollout_colorbar_legend.png'),
                                bbox_inches='tight', dpi=300)
                else:
                    fig.savefig(os.path.join(out_dir, f'vit_rollout_{type_b}_{ah}.png'),
                                bbox_inches='tight', pad_inches=0, dpi=300)
                plt.close(fig)

            _plot(with_cbar=False)
            # One shared colorbar reference (all maps use the same vmin=0/vmax=1 scale)
            if type_b == "charge" and abs(ah - 0.583) < 1e-3:
                _plot(with_cbar=True)

    rollout.close()
