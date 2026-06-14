# sd-webui-FreSca

**EN** | [日本語](#日本語)

Post-CFG guidance extension for Stable Diffusion WebUI (Forge-based).  
Scales the CFG guidance delta independently in low- and high-frequency bands via 2-D FFT, giving separate control over global structure and fine detail.

Paper: [arXiv:2504.02154](https://arxiv.org/abs/2504.02154) (CVPR 2025 GMCV)  
Original implementation: [WikiChao/FreSca](https://github.com/WikiChao/FreSca)
> When `l = h = 1` the result is identical to standard CFG.
---


## Installation

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-FreSca

```

---


## Parameters

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| Low-freq Scale (l)  | 1.00 | Low-frequency (global structure) multiplier |
| High-freq Scale (h) | 1.25 | High-frequency (detail / edges) multiplier |
| Freq Cutoff         | 20   | DC-radius dividing low and high bands (ComfyUI default) |

---


## Algorithm

```
Δ      = cond_denoised − uncond_denoised
F      = fftshift( fft2(Δ) )
mask   = (radius ≤ cutoff) ? l : h        →  per-frequency scale
Δ̂      = ifft2( ifftshift( F × mask ) ).real
output = uncond_denoised + scale × Δ̂

```

Low band (≤ cutoff) scaled by `l`, high band (> cutoff) by `h`.  
Decomposing the guidance delta in frequency space lets structure and detail be tuned separately.

---

---


# 日本語

**[English](#sd-webui-fresca)** | 日本語

Forge 系 WebUI 向け Post-CFG ガイダンス拡張機能。  
CFG ガイダンス差分を 2D FFT で低周波帯と高周波帯に分け、それぞれ独立にスケーリングして、グローバル構造とディテールを別々に制御します。

論文: [arXiv:2504.02154](https://arxiv.org/abs/2504.02154)（CVPR 2025 GMCV）  
原実装: [WikiChao/FreSca](https://github.com/WikiChao/FreSca)
> `l = h = 1` のとき通常 CFG と等価です。
---


## インストール

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-FreSca

```

---


## パラメータ

| パラメータ | 既定値 | 説明 |
| --- | --- | --- |
| Low-freq Scale (l)  | 1.00 | 低周波（グローバル構造）の乗数 |
| High-freq Scale (h) | 1.25 | 高周波（ディテール・エッジ）の乗数 |
| Freq Cutoff         | 20   | 低周波帯と高周波帯を分ける DC からの半径（ComfyUI 既定値） |

---


## アルゴリズム

```
Δ      = cond_denoised − uncond_denoised
F      = fftshift( fft2(Δ) )
mask   = (半径 ≤ cutoff) ? l : h          →  周波数ごとのスケール
Δ̂      = ifft2( ifftshift( F × mask ) ).real
output = uncond_denoised + scale × Δ̂

```

低周波帯（≤ cutoff）を `l`、高周波帯（> cutoff）を `h` でスケーリングします。  
ガイダンス差分を周波数領域で分解することで、構造とディテールを別々に調整できます。

---


## ライセンス

MIT License — Original implementation: [WikiChao/FreSca](https://github.com/WikiChao/FreSca)  
Based on: [arXiv:2504.02154](https://arxiv.org/abs/2504.02154)
