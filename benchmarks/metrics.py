from __future__ import annotations

import torch


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return g.unsqueeze(0)


@torch.no_grad()
def ssim(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    assert img1.shape == img2.shape and img1.dim() == 4
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ch = img1.shape[1]
    g1d = _gaussian_window(11, 1.5, img1.device, img1.dtype).squeeze(0)
    win2d = (g1d.unsqueeze(1) @ g1d.unsqueeze(0)).expand(ch, 1, 11, 11)
    pad = 11 // 2

    def f(x):
        return torch.nn.functional.conv2d(x, win2d, padding=pad, groups=ch)

    mu1, mu2 = f(img1), f(img2)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = f(img1 * img1) - mu1_sq
    s2 = f(img2 * img2) - mu2_sq
    s12 = f(img1 * img2) - mu12
    cs = (2 * s12 + C2) / (s1 + s2 + C2)
    l = (2 * mu12 + C1) / (mu1_sq + mu2_sq + C1)
    return (cs * l).mean(dim=(1, 2, 3))


@torch.no_grad()
def psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = (img1 - img2).square().mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(torch.tensor(data_range ** 2, device=img1.device) / mse.clamp_min(1e-12))


@torch.no_grad()
def mae(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    return (img1 - img2).abs().mean(dim=(1, 2, 3))


def u8_to_tensor(arr, alpha_weight: bool = False) -> torch.Tensor:
    import numpy as np
    a = np.asarray(arr, dtype=np.float32)
    if a.shape[-1] == 4:
        a = a[:, :, :3]
    t = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0) / 255.0
    return t


@torch.no_grad()
def compare_u8(ref_u8, soft_u8) -> dict:
    a = u8_to_tensor(ref_u8)
    b = u8_to_tensor(soft_u8)
    return {
        "ssim": float(ssim(a, b)),
        "psnr": float(psnr(a, b)),
        "mae": float(mae(a, b)),
    }
