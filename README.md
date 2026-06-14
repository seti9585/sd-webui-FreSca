# sd-webui-FreSca

A reForge extension that applies **FreSca** — frequency-domain scaling of the
CFG guidance delta — as a post-CFG hook.

> **Paper:** Chao Huang et al., "FreSca: Scaling in Frequency Space Enhances
> Diffusion Models," arXiv:2504.02154 (CVPR 2025 GMCV Workshop)  
> **Origin:** [WikiChao/FreSca](https://github.com/WikiChao/FreSca) (MIT License)

---

## How it works / 仕組み

Standard CFG computes:

```
denoised = uncond + scale × (cond − uncond)
```

FreSca decomposes the guidance delta `Δ = cond − uncond` into low-frequency
(global structure) and high-frequency (detail / edges) components via 2-D FFT,
applies independent scaling factors to each band, then reconstructs:

```
Δ̂ = F⁻¹( l · M_low ⊙ F(Δ)  +  h · M_high ⊙ F(Δ) )
denoised = uncond + scale × Δ̂
```

When `l = h = 1` the result is identical to standard CFG.

---

通常の CFG は `denoised = uncond + scale × (cond − uncond)` で計算される。
FreSca はガイダンス差分 `Δ = cond − uncond` を 2D FFT で周波数領域に変換し、
低周波（グローバル構造）と高周波（ディテール・エッジ）に分割して独立した
スケーリング係数を適用したのち空間領域に戻す。`l = h = 1` のとき通常 CFG と等価。

---

## Parameters / パラメータ

| Parameter | Default | Description |
|---|---|---|
| **Low-freq Scale (l)** | 1.00 | Multiplier for low-frequency guidance (global structure). |
| **High-freq Scale (h)** | 1.25 | Multiplier for high-frequency guidance (detail / edges). |
| **Freq Cutoff** | 20 | Radius from DC (shifted-FFT centre) separating low and high bands. Matches the ComfyUI node default. |

---

## Pipeline position / 処理順序

```
TCFG          Pre-CFG  (sort 13.0)  SVD projection of uncond
SkimmedCFG    Pre-CFG  (sort 14.0)  outer_influence removal
CFG core      denoised = uncond + scale × (cond − uncond)
FreSca        Post-CFG (sort 15.2)  frequency-scaled guidance delta  ← this extension
MaHiRo        Post-CFG (sort 15.5)  cosine-sim blend CFG vs Leap
```

---

## Requirements / 動作環境

- `stable-diffusion-webui-reForge`
- Python 3.10 / PyTorch (torch.fft available in ≥ 1.7; already present in reForge)
- A1111 非対応（`set_model_sampler_post_cfg_function` は Forge バックエンド専用）

---

## File structure / ファイル構成

```
sd-webui-FreSca/
├── scripts/sd_webui_fresca.py   # UI + hook registration
├── sd_webui_fresca/__init__.py
├── sd_webui_fresca/core.py      # FFT algorithm
├── requirements.txt
└── README.md
```
