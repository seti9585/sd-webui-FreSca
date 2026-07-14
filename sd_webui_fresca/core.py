"""FreSca core: frequency-domain guidance scaling.

Paper : arXiv:2504.02154  "FreSca: Scaling in Frequency Space Enhances
        Diffusion Models"  (CVPR 2025 GMCV Workshop)
Origin: WikiChao/FreSca  (MIT License)

Algorithm (Eq. 5 in the paper)
───────────────────────────────
Given the guidance delta in denoised space:

    Δ = cond_denoised − uncond_denoised        (≡ −σ · Δε_t, same freq content)

Compute its 2-D DFT, shift DC to the centre, build a circular mask that
splits the spectrum into low-frequency and high-frequency components,
apply independent scaling factors l and h respectively, then reconstruct:

    Δ̂ = F⁻¹( l · M_l ⊙ U + h · M_h ⊙ U )

Final output:

    denoised_out = uncond_d + cfg_scale · Δ̂

When l = h = 1 the result is bit-identical to standard CFG.

Cutoff modes
────────────
- fixed  : M_l / M_h split at a constant FFT-index radius (freq_cutoff),
           shared across the whole batch and all channels. This is the
           paper's spatial-ratio style cutoff.
- energy : M_l / M_h split at a radius solved independently per sample
           AND per channel, so that cumulative power-spectrum energy
           reaches a target ratio r0. This is the paper's energy-based
           (adaptive) cutoff. Per-channel independence keeps this
           architecture-agnostic (SDXL's VAE latent and Anima's
           flow-matching latent are not assumed to share frequency
           characteristics across channels).
"""

from __future__ import annotations

import logging
import sys

import torch

logger = logging.getLogger(__name__)


def _emit(fmt, *args):
    """Emit a diagnostic line via BOTH the logger and a stderr print, so it
    shows regardless of a backend's logging configuration (reForge surfaces
    module warnings; some forks, e.g. Forge Neo, may not)."""
    try:
        msg = (fmt % args) if args else fmt
    except Exception:
        msg = str(fmt)
    logger.log(logging.WARNING, msg)
    try:
        print(msg, file=sys.stderr, flush=True)
    except Exception:
        pass


# One-shot latch so the fail-safe warning is emitted only once per process,
# not once per sampling step (a persistent failure would otherwise flood the
# console with identical lines).
_WARNED_HOOK_FAIL = False


# Marker attribute value used to identify this extension's own post-CFG hook
# so it can be removed before re-registration (idempotency / fail-safe against
# double-apply). Versioned string, following the same convention as
# sd-webui-SkimmedCFG / sd-webui-TCFG / sd-webui-MaHiRo / sd-webui-DifferenceCFG.
MARKER = "sd_webui_fresca_v1"


# ─────────────────────────────────────────────────────────────────────────────
# Fail-safe: remove any previously registered FreSca hook
# ─────────────────────────────────────────────────────────────────────────────

def remove_fresca_patches(unet) -> None:
    """Strip this extension's own post-CFG hook from *unet*, in place.

    reForge / Forge store post-CFG functions in
    ``unet.model_options["sampler_post_cfg_function"]`` (a list). Re-running
    ``process`` (re-enable, parameter change, repeated batches) would otherwise
    append a second hook and apply FreSca twice — over-scaling the guidance
    delta. Calling this before registration keeps the operation idempotent:
    at most one FreSca hook is ever present.

    Only hooks whose ``_sd_webui_fresca_marker`` attribute equals
    :data:`MARKER` are removed, so other extensions' post-CFG functions
    (e.g. MaHiRo) are left untouched.
    """
    opts = getattr(unet, "model_options", None)
    if not isinstance(opts, dict):
        return
    fns = opts.get("sampler_post_cfg_function")
    if not fns:
        return
    opts["sampler_post_cfg_function"] = [
        fn for fn in fns
        if getattr(fn, "_sd_webui_fresca_marker", None) != MARKER
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Cached geometry (distance-from-DC map, radial bin index)
# ─────────────────────────────────────────────────────────────────────────────
# Both only depend on (H, W, device), never on tensor content, so they are
# computed once per shape and reused across every sampling step. This mainly
# benefits the energy-based path (which additionally needs the integer bin
# index for scatter_add), but the fixed-radius path is refactored to share the
# same cache since the distance map itself was already being rebuilt from
# scratch every step.

_dist_cache: dict[tuple, torch.Tensor] = {}
_bin_idx_cache: dict[tuple, torch.Tensor] = {}


def _get_dist_map(H: int, W: int, device: torch.device) -> torch.Tensor:
    """Cached (H, W) Euclidean distance from the shifted-spectrum DC centre."""
    key = (H, W, device)
    cached = _dist_cache.get(key)
    if cached is None:
        cy, cx = H // 2, W // 2
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        cached = torch.sqrt((gy - cy) ** 2 + (gx - cx) ** 2)
        _dist_cache[key] = cached
    return cached


def _get_bin_index(H: int, W: int, device: torch.device) -> torch.Tensor:
    """Cached (H*W,) integer radial-bin index (floor of the distance map), flattened."""
    key = (H, W, device)
    cached = _bin_idx_cache.get(key)
    if cached is None:
        cached = _get_dist_map(H, W, device).floor().long().flatten()
        _bin_idx_cache[key] = cached
    return cached


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper — fixed-radius (spatial-ratio style) cutoff
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

    # ── Circular frequency mask  (cached, shared across B and C) ──────────
    dist     = _get_dist_map(H, W, xf.device)                  # (H, W)
    low_mask  = (dist <= freq_cutoff).to(F.real.dtype)         # 1.0 inside radius
    high_mask = 1.0 - low_mask                                 # 1.0 outside radius

    # ── Per-band scaling  (broadcasts over B and C automatically) ─────────
    scale_map = scale_low * low_mask + scale_high * high_mask  # (H, W)
    F_scaled  = F * scale_map

    # ── Inverse FFT back to spatial domain ────────────────────────────────
    return torch.fft.ifft2(torch.fft.ifftshift(F_scaled)).real.to(orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper — energy-based (adaptive) cutoff
# ─────────────────────────────────────────────────────────────────────────────

def _freq_scale_2d_adaptive(
    x: torch.Tensor,
    scale_low: float,
    scale_high: float,
    r0: float,
) -> torch.Tensor:
    """Energy-based counterpart of _freq_scale_2d.

    Instead of a fixed pixel-radius cutoff, the low/high boundary radius is
    solved independently per sample AND per channel, so that the cumulative
    power-spectrum energy within that radius reaches the target ratio r0.
    Per-channel independence (rather than summing energy across channels)
    is intentional: it generalizes across latent spaces with different
    per-channel frequency characteristics (e.g. SDXL's VAE latent vs
    Anima's flow-matching latent), where no single fixed radius, and no
    single shared radius across channels, is assumed to be meaningful.

    Args:
        x         : (B, C, H, W) float tensor to be scaled.
        scale_low : Multiplier l for the low-energy band.
        scale_high: Multiplier h for the high-energy band.
        r0        : Target cumulative energy ratio (0.0-1.0) that defines
                    the low/high boundary.

    Returns:
        Frequency-scaled tensor, same shape and dtype as x.
    """
    orig_dtype = x.dtype
    xf = x.float()

    # The last two axes are the spatial (H, W) plane the 2-D FFT operates on;
    # everything before them is treated as an independent "unit" over which a
    # radius is solved separately. This keeps the function rank-agnostic:
    # SDXL hands back (B, C, H, W) so lead = (B, C), while Anima (NextDiT) hands
    # back (B, C, T, H, W) with a singleton frame axis so lead = (B, C, T). The
    # fixed-radius path already works on any rank because it only reads
    # shape[-2:]; matching that here fixes the "too many values to unpack"
    # crash that a hard B, C, H, W = xf.shape produced on Anima's 5-D latent.
    *lead, H, W = xf.shape
    N = 1
    for d in lead:
        N *= d                                    # number of independent units

    F = torch.fft.fftshift(torch.fft.fft2(xf))    # (*lead, H, W) complex64

    dist    = _get_dist_map(H, W, xf.device)                  # (H, W)
    bin_idx = _get_bin_index(H, W, xf.device)                 # (H*W,) long

    # ── Power spectrum, radially binned per independent unit ──────────────
    power = (F.real ** 2 + F.imag ** 2).reshape(N, H * W)       # (N, H*W)
    n_bins = int(bin_idx.max().item()) + 1
    energy_per_bin = torch.zeros(
        N, n_bins, device=xf.device, dtype=power.dtype
    )
    energy_per_bin.scatter_add_(1, bin_idx.unsqueeze(0).expand(N, -1), power)

    # ── Smallest radius whose cumulative energy reaches r0 of the total ───
    cum_energy = energy_per_bin.cumsum(dim=1)                  # (N, n_bins)
    target     = float(r0) * cum_energy[:, -1:]
    radius_idx = torch.searchsorted(cum_energy, target).squeeze(1)
    radius     = radius_idx.clamp(max=n_bins - 1).float().view(*lead)  # (*lead,)

    # ── Per-unit circular mask ─────────────────────────────────────────────
    # Add the two trailing spatial axes to radius so it broadcasts against the
    # (H, W) distance map; dist gets matching leading singleton axes.
    dist_b   = dist.reshape((1,) * len(lead) + (H, W))         # (1.., H, W)
    radius_b = radius.reshape(tuple(lead) + (1, 1))            # (*lead, 1, 1)
    low_mask  = (dist_b <= radius_b).to(F.real.dtype)          # (*lead, H, W)
    high_mask = 1.0 - low_mask
    scale_map = scale_low * low_mask + scale_high * high_mask  # (*lead, H, W)
    F_scaled  = F * scale_map

    return torch.fft.ifft2(torch.fft.ifftshift(F_scaled)).real.to(orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Public post-CFG hook
# ─────────────────────────────────────────────────────────────────────────────

def apply_fresca(
    args: dict,
    scale_low: float,
    scale_high: float,
    cutoff_mode: str,
    freq_cutoff: int,
    r0: float,
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

    Args:
        cutoff_mode : ``"fixed"`` uses freq_cutoff (constant radius, shared
                      across the whole batch and all channels). ``"energy"``
                      uses r0 (radius solved independently per sample and
                      per channel so cumulative spectrum energy reaches r0).
        freq_cutoff : Only used when cutoff_mode == "fixed".
        r0          : Only used when cutoff_mode == "energy".

    Fail-safe: if anything inside raises (an unexpected latent rank, a missing
    key, a device/dtype edge case, ...), the standard CFG output
    (``args["denoised"]``) is returned unmodified so the generation completes
    instead of crashing. The failure is reported once per process via _emit
    (logger + stderr), matching the convention used by sd-webui-CFGZeroStar.

    Returns:
        Modified denoised tensor (or the unmodified standard CFG output on the
        fast-path no-ops and on internal failure).
    """
    global _WARNED_HOOK_FAIL

    # ── Fast-path: no guidance, or scaling is trivially identity ──────────
    # Read cond_scale defensively; the fast-paths must never be the thing that
    # throws, since they are the cheap "nothing to do" exits.
    if args.get("cond_scale") == 1.0:
        return args["denoised"]
    if scale_low == 1.0 and scale_high == 1.0:
        return args["denoised"]

    try:
        cond_d    = args["cond_denoised"]    # (B, C, H, W) or higher rank
        uncond_d  = args["uncond_denoised"]  # same shape as cond_d
        cfg_scale = args["cond_scale"]       # float

        # ── Guidance delta in denoised space ──────────────────────────────
        #   Δ = cond_d − uncond_d  (sign-equivalent to −σ · Δε_t; same
        #   frequency content as the noise-space delta the paper operates on.)
        delta = cond_d - uncond_d

        # ── Frequency-aware rescaling ─────────────────────────────────────
        if cutoff_mode == "energy":
            delta_scaled = _freq_scale_2d_adaptive(delta, scale_low, scale_high, r0)
        else:
            delta_scaled = _freq_scale_2d(delta, scale_low, scale_high, freq_cutoff)

        # ── Reconstruct denoised output ───────────────────────────────────
        return uncond_d + cfg_scale * delta_scaled

    except Exception as exc:
        if not _WARNED_HOOK_FAIL:
            _emit("[FreSca] hook skipped (returning standard CFG): %r", exc)
            _WARNED_HOOK_FAIL = True
        return args["denoised"]
