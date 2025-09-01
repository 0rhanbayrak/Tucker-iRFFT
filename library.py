import os
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker

# 1) Load data
def load_images(folder, size=(128,128)):
    transform = T.Compose([T.Resize(size), T.ToTensor()])
    data_list = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert("RGB")
            data_list.append(transform(img))
    X = torch.stack(data_list)
    X_formatted = X.permute(2,3,1,0)
    return X_formatted

# 2) 3D RFFT and visualization
def rfft_visualize(X_formatted, img_index=0):
    X_rfft = torch.fft.rfftn(X_formatted, dim=(0,1,2))
    X_rfft_real = X_rfft.real
    fft_image_real = X_rfft_real[:, :, :, img_index]
    fft_magnitude = torch.abs(fft_image_real).sum(dim=2)
    plt.imshow(fft_magnitude, cmap='gray')
    plt.title(f"3D RFFT (Real Part) - Image #{img_index+1}")
    plt.axis('off')
    plt.show()
    #return fft_magnitude

# 3) Spatial Tucker
def tucker_spatial(X_formatted, H_rank=60, W_rank=60, sample_rank=None):
    tl.set_backend('numpy')
    H, W, C, N = X_formatted.shape
    X_np = X_formatted.cpu().numpy()
    if sample_rank is None: sample_rank = min(N, 200)
    ranks = [min(H_rank,H), min(W_rank,W), C, sample_rank]
    core, fac = tucker(X_np, rank=ranks)
    X_rec = tl.tucker_to_tensor((core, fac))
    X_rec = np.clip(X_rec, 0.0, 1.0)
    return X_np, X_rec

# 4) 3D FFT → Tucker → iFFT
def tucker_fft_reconstruct(X_formatted, H_rank=60, W_rank=60, sample_rank=None):
    tl.set_backend("numpy")
    H, W, C, N = X_formatted.shape
    X_fft3 = torch.fft.rfftn(X_formatted, dim=(0,1,2), norm='ortho')
    H3, W3, Cf, N3 = X_fft3.shape
    R = X_fft3.real.cpu().numpy()
    I = X_fft3.imag.cpu().numpy()
    Z = np.stack([R, I], axis=3)
    if sample_rank is None: sample_rank = min(N, 200)
    ranks_fft = [min(H_rank,H3), min(W_rank,W3), Cf, 2, sample_rank]
    core_f, fac_f = tucker(Z, rank=ranks_fft)
    Z_hat = tl.tucker_to_tensor((core_f, fac_f))
    R_hat = Z_hat[:,:,:,0,:]
    I_hat = Z_hat[:,:,:,1,:]
    X_fft3_hat = torch.complex(torch.from_numpy(R_hat), torch.from_numpy(I_hat))
    X_restored = torch.fft.irfftn(X_fft3_hat, s=(H,W,C), dim=(0,1,2), norm='ortho').real
    X_restored_np = np.clip(X_restored.cpu().numpy(), 0.0, 1.0)
    return X_restored_np

# 5) Metrics
def metrics(orig, rec):
    diff = orig - rec
    mse = float(np.mean(diff**2))
    maxd = float(np.max(np.abs(diff)))
    mind = float(np.min(np.abs(diff)))
    return mse, maxd, mind

# 6) Grid visualization
def plot_tucker_fft_grid(X_np, X_rec_spatial, X_restored_np, img_index,
                         width=8, height=5.5, dpi=140, font=8, interp='bilinear'):
    X_orig = np.clip(X_np, 0.0, 1.0)
    X_spat = np.clip(X_rec_spatial, 0.0, 1.0)
    X_freq = np.clip(X_restored_np, 0.0, 1.0)
    orig = X_orig[:, :, :, img_index]
    rec_s = X_spat[:, :, :, img_index]
    rec_f = X_freq[:, :, :, img_index]

    rel_err_spatial = 1-(np.linalg.norm(orig - rec_s) / np.linalg.norm(orig))**2
    rel_err_fft = 1-(np.linalg.norm(orig - rec_f) / np.linalg.norm(orig))**2

    err_s = np.abs(orig - rec_s).mean(axis=2)
    err_f = np.abs(orig - rec_f).mean(axis=2)
    vmax_err = float(max(err_s.max(), err_f.max()))

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    gs = fig.add_gridspec(2,3,width_ratios=[1,1,1], wspace=0.08, hspace=0.12)

    ax_orig = fig.add_subplot(gs[:,0])
    ax_orig.imshow(orig, vmin=0,vmax=1, interpolation=interp)
    ax_orig.set_title("Original", fontsize=font)
    ax_orig.axis('off')

    ax_tucker = fig.add_subplot(gs[0,1])
    ax_tucker.imshow(rec_s, vmin=0,vmax=1, interpolation=interp)
    ax_tucker.set_title(f"Tucker,RelErr: {rel_err_spatial:.4f}", fontsize=font)
    ax_tucker.axis('off')

    ax_fft = fig.add_subplot(gs[1,1])
    ax_fft.imshow(rec_f, vmin=0,vmax=1, interpolation=interp)
    ax_fft.set_title(f"FFT Tucker,RelErr: {rel_err_fft:.4f}", fontsize=font)
    ax_fft.axis('off')

    ax_err_s = fig.add_subplot(gs[0,2])
    im_err_s = ax_err_s.imshow(err_s, cmap='magma', vmin=0.0, vmax=vmax_err, interpolation='nearest')
    ax_err_s.set_title("Error (Tucker)", fontsize=font)
    ax_err_s.axis('off')

    ax_err_f = fig.add_subplot(gs[1,2])
    im_err_f = ax_err_f.imshow(err_f, cmap='magma', vmin=0.0, vmax=vmax_err, interpolation='nearest')
    ax_err_f.set_title("Error (FFT Tucker)", fontsize=font)
    ax_err_f.axis('off')

    cbar = fig.colorbar(im_err_f, ax=[ax_err_s, ax_err_f], fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=max(font-2,6))
    cbar.set_label('abs error (mean over channels)', fontsize=font-1)
    plt.tight_layout(pad=0.6)
    plt.show()

