"""FreSca core: frequency-domain guidance scaling.

Paper : arXiv:2504.02154  "FreSca: Scaling in Frequency Space Enhances
        Diffusion Models"  (CVPR 2025 GMCV Workshop)
Origin: WikiChao/FreSca  (MIT License)

Algorithm (Eq. 5 in the paper)
───────────────────────────────
Given the guidance delta in denoised space:

    Δ = cond_denoised − uncond_denoised        (≡ −σ · Δε_t, same freq content)

Compute its 2-D DFT, shift DC to the centre, build a circular mask that
splits the spectrum into low-frequency (|freq| ≤ freq_cutoff) and
high-frequency (|freq| > freq_cutoff) components, apply independent
scaling factors l and h respectively, then reconstruct:

    Δ̂ = F⁻¹( l · M_l ⊙ U + h · M_h ⊙ U )

Final output:

    denoised_out = uncond_d + cfg_scale · Δ̂

When l = h = 1 the result is bit-identical to standard CFG.
"""

from __future__ import annotations

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _freq_scale_2d(
    x: torch.Tensor,
    scale_low: float,
    scale_high: float,
    freq_cutoff: int,
) -> torch.Tensor:
    """Apply independent low/high-frequency scaling to a spatial tensor.

    Args:
        x           : (B, C, H, W) float tensor to be scaled.
        scale_low   : Multiplier *l* for |freq| ≤ freq_cutoff (low band).
        scale_high  : Multiplier *h* for |freq| >  freq_cutoff (high band).
        freq_cutoff : Radius in FFT-index units that separates the two bands.
                      Expressed as distance from the DC component after
                      fftshift, matching the ComfyUI node convention
                      (default 20 ≈ inner ~30 % at 64 × 64 latent size).

    Returns:
        Frequency-scaled tensor, same shape and dtype as *x*.
    """
    orig_dtype = x.dtype
    xf = x.float()                          # work in fp32 for FFT stability
    H, W = xf.shape[-2:]

    # ── Forward FFT — shift DC to spatial centre ──────────────────────────
    F = torch.fft.fftshift(torch.fft.fft2(xf))   # (B, C, H, W)  complex64

    # ── Circular frequency mask  (computed once, shared across B and C) ───
    cy, cx = H // 2, W // 2
    gy, gx = torch.meshgrid(
        torch.arange(H, device=xf.device, dtype=torch.float32),
        torch.arange(W, device=xf.device, dtype=torch.float32),
        indexing="ij",
    )
    dist      = torch.sqrt((gy - cy) ** 2 + (gx - cx) ** 2)  # (H, W)
    low_mask  = (dist <= freq_cutoff).to(F.real.dtype)         # 1.0 inside radius
    high_mask = 1.0 - low_mask                                 # 1.0 outside radius

    # ── Per-band scaling  (broadcasts over B and C automatically) ─────────
    scale_map = scale_low * low_mask + scale_high * high_mask  # (H, W)
    F_scaled  = F * scale_map

    # ── Inverse FFT back to spatial domain ────────────────────────────────
    return torch.fft.ifft2(torch.fft.ifftshift(F_scaled)).real.to(orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Public post-CFG hook
# ─────────────────────────────────────────────────────────────────────────────

def apply_fresca(
    args: dict,
    scale_low: float,
    scale_high: float,
    freq_cutoff: int,
) -> torch.Tensor:
    """Post-CFG hook: replace standard guidance with frequency-scaled guidance.

    Registered via ``set_model_sampler_post_cfg_function`` at
    ``sorting_priority = 15.2``, placing it after the core CFG computation
    and before MaHiRo (15.5).

    Expected keys in *args*:
        ``cond_denoised``   – conditional denoised prediction  (B, C, H, W)
        ``uncond_denoised`` – unconditional denoised prediction (B, C, H, W)
        ``cond_scale``      – CFG scale (float)
        ``denoised``        – standard CFG output  (returned unchanged on no-op)

    Returns:
        Modified denoised tensor.
    """
    cond_d    = args["cond_denoised"]    # (B, C, H, W)
    uncond_d  = args["uncond_denoised"]  # (B, C, H, W)
    cfg_scale = args["cond_scale"]       # float

    # ── Fast-path: no guidance, or scaling is trivially identity ──────────
    if cfg_scale == 1.0:
        return args["denoised"]
    if scale_low == 1.0 and scale_high == 1.0:
        return args["denoised"]

    # ── Guidance delta in denoised space ──────────────────────────────────
    #   Δ = cond_d − uncond_d  (sign-equivalent to −σ · Δε_t; same frequency
    #   content as the noise-space delta the paper operates on.)
    delta = cond_d - uncond_d  # (B, C, H, W)

    # ── Frequency-aware rescaling ─────────────────────────────────────────
    delta_scaled = _freq_scale_2d(delta, scale_low, scale_high, freq_cutoff)

    # ── Reconstruct denoised output ───────────────────────────────────────
    return uncond_d + cfg_scale * delta_scaled
