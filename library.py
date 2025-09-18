import os
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from typing import Tuple, List

# 1) Load data
def load_images(folder, size=(128,128)):
    transform = T.Compose([T.Resize(size), T.ToTensor()])
    data_list = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert("RGB")
            data_list.append(transform(img))
    X = torch.stack(data_list).to(torch.float64)
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

# 3) Spatial Tucker
def tucker_spatial(X_formatted, H_rank=20, W_rank=20, sample_rank=200):
    tl.set_backend('pytorch')
    print(X_formatted.numel() * X_formatted.element_size() / 1024)
    H, W, C, N = X_formatted.shape
    #X_np = X_formatted.copy()
    if sample_rank is None: sample_rank = min(N, sample_rank)
    ranks = [min(H_rank,H), min(W_rank,W), C, sample_rank]
    core, fac = tucker(X_formatted, rank=ranks,tol=1e-12)
    X_rec = tl.tucker_to_tensor((core, fac))
    X_rec = np.clip(X_rec, 0.0, 1.0)
    return X_formatted, X_rec

def tucker_fft_reconstruct(X_formatted, H_rank=20, W_rank=20, sample_rank=200):
    H,W,C,N = X_formatted.shape
    X_fft3 = torch.fft.rfftn(X_formatted, dim=(0,1), norm='ortho').to(torch.complex64)
    print(X_fft3.numel() * X_fft3.element_size() / 1024)
    print(X_fft3.shape)
    Us, G, X_hat, errs = tucker_hooi_4d(X_fft3, ranks=[H_rank,W_rank,3,sample_rank])
    print(sum(t.element_size() * t.numel() for t in Us) / 1024 + G.element_size()*G.numel() / 1024)

    X_restored = torch.fft.irfftn(X_hat, dim=(0,1), norm='ortho').real
    X_restored_np = np.clip(X_restored.cpu().numpy(), 0.0, 1.0)
    
    return X_restored_np
    
# 4) Metrics
def metrics(orig, rec):
    diff = orig - rec
    mse = float(np.mean(diff**2))
    maxd = float(np.max(np.abs(diff)))
    mind = float(np.min(np.abs(diff)))
    return mse, maxd, mind

# 5) Grid visualization
def plot_tucker_fft_grid(X_np, X_rec_spatial, X_restored_np, img_index,
                         width=8, height=5.5, dpi=140, font=8, interp='bilinear'):
    X_orig = np.clip(X_np, 0.0, 1.0)
    X_spat = np.clip(X_rec_spatial, 0.0, 1.0)
    X_freq = np.clip(X_restored_np, 0.0, 1.0)
    orig = X_orig[:, :, :, img_index]
    rec_s = X_spat[:, :, :, img_index]
    rec_f = X_freq[:, :, :, img_index]

    rel_err_spatial = 1-(np.linalg.norm(orig - rec_s)**2 / np.linalg.norm(orig)**2)
    rel_err_fft = 1-(np.linalg.norm(orig - rec_f)**2 / np.linalg.norm(orig)**2)

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
    return rel_err_spatial, rel_err_fft


def unfold(X: torch.Tensor, mode: int) -> torch.Tensor:
    """Mode-n matricization: (I_n) x (prod_{m!=n} I_m)."""
    N = X.ndim
    perm = (mode,) + tuple(i for i in range(N) if i != mode)
    Xp = X.permute(perm)
    return Xp.reshape(X.shape[mode], -1)

def n_mode_product(X: torch.Tensor, U: torch.Tensor, mode: int, left_adj: bool=False) -> torch.Tensor:
    """
    Compute Y = X ×_mode U     (if left_adj=False), where U shape is (J, I_mode)
            or Y = X ×_mode Uᴴ (if left_adj=True),  where U shape is (I_mode, J)
    This routine uses Hermitian when left_adj=True via .mH.
    """
    N = X.ndim
    perm = (mode,) + tuple(i for i in range(N) if i != mode)
    Xp = X.permute(perm).reshape(X.shape[mode], -1)  # (I_mode, rest)
    if left_adj:
        # Left-multiply by Uᴴ:   (J, I_mode) <- Uᴴ has shape (J, I_mode)
        Ymat = (U.mH) @ Xp  # (J, rest)
        new_dim0 = U.shape[1]  # J when U is (I_mode, J)
    else:
        # Left-multiply by U:    (J, I_mode) <- U has shape (J, I_mode)
        Ymat = U @ Xp          # (J, rest)
        new_dim0 = U.shape[0]
    new_shape = (new_dim0,) + tuple(X.shape[i] for i in range(N) if i != mode)
    Yp = Ymat.reshape(new_shape)
    # invert the permutation
    inv = [perm.index(i) for i in range(N)]
    return Yp.permute(inv)

def multi_mode_left_adj(X: torch.Tensor, factors: List[torch.Tensor], skip: int) -> torch.Tensor:
    """
    Y = X ×_{m≠skip} U_mᴴ    (i.e., left-adjoint by all factors except 'skip').
    After this, mode 'skip' keeps its original size; all other modes become R_m.
    """
    Y = X
    for m, U in enumerate(factors):
        if m == skip:
            continue
        Y = n_mode_product(Y, U, mode=m, left_adj=True)
    return Y

def tucker_hooi_4d(
    X: torch.Tensor,
    ranks: Tuple[int, int, int, int],
    n_iter_max: int = 50,
    tol: float = 1e-12,
    verbose: bool = True,
):
    """
    HOOI (ALS) Tucker decomposition for a 4th-order tensor X (I1 x I2 x I3 x I4).
    Supports complex dtypes and uses Hermitian adjoints correctly.

    Returns: (factors, core, recon, errors)
      - factors: [U1, U2, U3, U4] with shapes (I_n x R_n)
      - core: G of shape (R1 x R2 x R3 x R4)
      - recon: reconstruction of X
      - errors: list of relative reconstruction errors per iteration
    """
    assert X.ndim == 4, "This implementation expects a 4th-order tensor."
    device, dtype = X.device, X.dtype
    I1, I2, I3, I4 = X.shape
    R1, R2, R3, R4 = ranks
    R1, R2, R3, R4 = min(R1, I1), min(R2, I2), min(R3, I3), min(R4, I4)
    ranks = (R1, R2, R3, R4)

    # --- HOSVD init: leading left singular vectors per unfolding ---
    Us: List[torch.Tensor] = []
    for n, (In, Rn) in enumerate(zip(X.shape, ranks)):
        Xn = unfold(X, n)  # (In x prod others)
        # SVD works for real or complex dtypes
        Un, _, _ = torch.linalg.svd(Xn, full_matrices=False)
        Us.append(Un[:, :Rn].to(device=device, dtype=dtype))

    # --- ALS (HOOI) ---
    normX = torch.linalg.norm(X)
    prev_err = float('inf')
    errors: List[float] = []

    for it in range(n_iter_max):
        for n in range(4):
            # Project X by U_mᴴ for all m ≠ n, then SVD on mode-n unfolding
            Yn = multi_mode_left_adj(X, Us, skip=n)
            Yn_mat = unfold(Yn, n)  # shape: (I_n x prod_{m≠n} R_m)
            Un, _, _ = torch.linalg.svd(Yn_mat, full_matrices=False)
            Us[n] = Un[:, :ranks[n]].to(device=device, dtype=dtype)

        # Compute core with Hermitian adjoints of all factors
        G = X
        for n in range(4):
            G = n_mode_product(G, Us[n], mode=n, left_adj=True)

        # Reconstruct
        Xhat = G
        for n in range(4):
            Xhat = n_mode_product(Xhat, Us[n], mode=n, left_adj=False)

        err = (torch.linalg.norm(X - Xhat) / (normX + 1e-12)).item()
        errors.append(err)
        if verbose:
            print(f"Iter {it+1:02d} | rel. error = {err:.6e}")
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    return Us, G, Xhat, errors