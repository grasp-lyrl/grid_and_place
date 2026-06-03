import numpy as np
import torch
import torch.nn.functional as F


def lag_sum(
    base: torch.Tensor,
    lagged: torch.Tensor,
) -> torch.Tensor:
    """
    base: (N, H, W), base map
    lagged: (N, H, W), lagged response map
    """
    if base.ndim != 3 or lagged.ndim != 3:
        raise ValueError(f"Expected (N, H, W), got {base.shape} and {lagged.shape}")
    if base.shape != lagged.shape:
        raise ValueError(
            "base and lagged must have same shape, ", 
            f"got {base.shape} and {lagged.shape}"
        )

    _, H, W = base.shape

    padding = (0, W - 1, 0, H - 1)
    base_padded = F.pad(base, padding)
    lagged_padded = F.pad(lagged, padding)

    base_fft = torch.fft.fft2(base_padded)
    lagged_fft = torch.fft.fft2(lagged_padded)

    lag_sums = torch.fft.ifft2(base_fft * torch.conj(lagged_fft)).real
    lag_sums = torch.fft.fftshift(lag_sums, dim=(-2, -1))

    center_y = lag_sums.shape[-2] // 2
    center_x = lag_sums.shape[-1] // 2

    return lag_sums[
        :,
        center_y - (H - 1): center_y + H,
        center_x - (W - 1): center_x + W,
    ]


def compute_sac(
    ratemaps: torch.Tensor, 
    eps: float = 1e-12,
    min_overlap: int = 3,
):
    """
    ratemaps: (N, H, W). NaN means invalid / unvisited bin
    eps: numerical stability constant
    min_overlap: minimum valid overlapping bins required per lag
    """
    if ratemaps.ndim != 3:
        raise ValueError(f"Expected (N, H, W), got {tuple(ratemaps.shape)}")
    
    x = ratemaps.to(torch.float32)
    N, H, W = x.shape

    valid = torch.isfinite(x)
    mask = valid.to(x.dtype)

    # fill invalid bins with zero
    x = torch.where(valid, x, torch.zeros_like(x))
    x_squared = x * x

    # pre-compute lag-sums
    n = torch.round(lag_sum(mask, mask))

    sum_x = lag_sum(x, mask)
    sum_y = lag_sum(mask, x)

    sum_x_squared = lag_sum(x_squared, mask)
    sum_y_squared = lag_sum(mask, x_squared)

    sum_xy = lag_sum(x, x)

    # corr(x, y) =
    #   (n * sum_xy - sum_x * sum_y)
    #   / sqrt((n * sum_x2 - sum_x^2) * (n * sum_y2 - sum_y^2))
    numerator = n * sum_xy - sum_x * sum_y

    var_x = torch.clamp(n * sum_x_squared - sum_x * sum_x, min=0.0)
    var_y = torch.clamp(n * sum_y_squared - sum_y * sum_y, min=0.0)

    denominator = torch.sqrt(var_x * var_y)

    supported = (n >= float(min_overlap)) & (denominator > eps)

    sac = torch.where(
        supported,
        numerator / denominator.clamp(min=eps),
        torch.zeros_like(numerator),
    )

    center_y, center_x = H - 1, W - 1
    center_supported = supported[:, center_y, center_x]

    sac[:, center_y, center_x] = torch.where(
        center_supported,
        torch.ones_like(sac[:, center_y, center_x]),
        torch.zeros_like(sac[:, center_y, center_x]),
    )

    sac = torch.nan_to_num(sac, nan=0.0, posinf=0.0, neginf=0.0)
    sac = torch.clamp(sac, min=-1.0, max=1.0)

    return sac


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    ratemap_path = args.path
    if not ratemap_path.name.startswith("ratemap"):
        raise ValueError(f"Expected filename to start with 'ratemap': {ratemap_path}")

    autocorr_path = ratemap_path.with_name(
        "autocorr" + ratemap_path.name[len("ratemap"):]
    )

    ratemap = np.load(ratemap_path)["ratemap"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocorr = compute_sac(torch.from_numpy(ratemap).to(device)).cpu().numpy()

    np.savez_compressed(autocorr_path, autocorr=autocorr)
    print(f"saved {autocorr_path}")


if __name__ == "__main__":
    main()
