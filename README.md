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
| Low-freq Scale (l)  | 1.00 | Multiplier for low-frequency components (global structure, composition) |
| High-freq Scale (h) | 1.25 | Multiplier for high-frequency components (detail, edges, texture) |
| Cutoff Mode         | Fixed radius | `Fixed radius`: constant FFT-index radius, shared across the whole batch and all channels. `Energy-based (adaptive)`: radius solved independently per sample and per channel so cumulative power-spectrum energy reaches the target ratio (`r0`) |
| Freq Cutoff         | 20   | Radius from DC (centre of shifted spectrum) dividing the two bands. ComfyUI default = 20. **Fixed radius mode only** |
| Energy Ratio (r0)   | 0.90 | Target cumulative power-spectrum energy ratio defining the low/high boundary. Matches the paper's Text-to-Image energy-based cutoff configuration (Section 4.1, arXiv:2504.02154). **Energy-based (adaptive) mode only** |

Raising `h` above 1.0 sharpens detail and texture without affecting overall composition.  
Raising `l` above 1.0 strengthens global structure and prompt adherence.  
Both can be tuned independently; setting either to 1.0 leaves that band unchanged.

---

## Cutoff Modes

**Fixed radius** (default) uses a constant FFT-index radius, identical for every sample in the batch and every channel. This is the paper's spatial-ratio style cutoff.

**Energy-based (adaptive)** solves the radius independently per sample AND per channel, so that the cumulative power-spectrum energy within that radius reaches a target ratio `r0`. This is the paper's energy-based cutoff (Section 4.1 / Fig. 8–9 in arXiv:2504.02154). Per-channel independence is intentional: it keeps the cutoff architecture-agnostic, since no single radius, and no single radius shared across channels, is assumed to be meaningful across different latent spaces (SDXL's VAE latent vs. Anima's flow-matching latent).

```
F           = fftshift( fft2(delta) )                     # (B, C, H, W) complex
power       = |F|²
radius[b,c] = min r such that Σ_{|freq| ≤ r} power[b,c] ≥ r0 × Σ power[b,c]
low_mask    = (radius_from_DC ≤ radius[b,c])              # per-sample, per-channel
```

The distance-from-DC map and its radial bin index only depend on (H, W, device), so both are cached and reused across every sampling step; only the per-step energy accumulation is recomputed. The rest of the reconstruction (`scale_map`, `delta_scaled`, `output`) is identical to the Fixed radius path — only the mask now varies per sample and per channel instead of being shared.

---

## Algorithm

```
delta         = cond_denoised − uncond_denoised
F             = fftshift( fft2(delta) )          # shift DC to centre
low_mask      = (radius_from_DC ≤ freq_cutoff)
scale_map     = l × low_mask + h × (1 − low_mask)
delta_scaled  = ifft2( ifftshift( F × scale_map ) ).real
output        = uncond_denoised + cfg_scale × delta_scaled
```

The guidance delta is decomposed in frequency space.  
Low and high bands are scaled independently, then the denoised output is reconstructed.  
When `l = h = 1`, the output is bit-identical to standard CFG.

---

## Compatibility with other extensions

FreSca (Post-CFG, `sorting_priority = 15.2`) sits between the core CFG computation and MaHiRo (15.5).  
The recommended stacking order is:

```
TCFG (Pre-CFG) → CFG core → FreSca (Post-CFG 15.2) → MaHiRo (Post-CFG 15.5)
```

TCFG and FreSca operate on orthogonal axes — TCFG corrects the *direction* of the guidance vector before CFG, while FreSca reshapes the *frequency content* of the guidance delta after CFG — so there is no conflict.  
When stacking multiple CFG-axis extensions, keep CFG in the 7–15 range to avoid cumulative correction breakdown.

### Note on Anima (flow-matching / DiT)

FreSca works at the post-CFG tensor level and is architecture-agnostic.  
However, Anima is typically run at CFG = 1.0, which triggers the fast-path (`return denoised` unchanged) and makes FreSca a no-op in practice.  
FreSca becomes active on Anima only when CFG is set above 1.0.

When active, the Energy-based (adaptive) cutoff mode is the one intended for Anima: it solves the low/high boundary independently per channel rather than assuming a fixed radius meaningful across architectures, since SDXL's VAE latent and Anima's flow-matching latent are not assumed to share frequency characteristics per channel.

---

## Implementation note — `process_before_every_sampling()`

Forge-based WebUIs rebuild `forge_objects.unet` between `process()` and the actual sampling start.  
Any hook registered in `process()` is silently discarded when this rebuild occurs.

This extension registers its hook in `process_before_every_sampling()`, where `forge_objects.unet` is already the same object that `CFGDenoiser` will reference during sampling.  
This guarantees the hook is called on every denoising step, including Hires.fix passes.

This behaviour was confirmed by comparing `id()` of the unet object at both registration sites and at the `CFGDenoiser` call site — a discrepancy that took considerable investigation to isolate.  
**Post-CFG extensions should always register hooks in `process_before_every_sampling()`, not `process()`.**

---

## Tested environments

- reForge (Python 3.10) — SDXL-family models
- Forge Neo (Python 3.12) — SDXL-family models and Anima; txt2img + Hires.fix confirmed

Not compatible with A1111 (`set_model_sampler_post_cfg_function` is Forge-backend only).

---

---

# 日本語

**[English](#sd-webui-fresca)** | 日本語

Forge 系 WebUI 向け Post-CFG ガイダンス拡張機能。  
CFG ガイダンス差分を 2D FFT で低周波帯と高周波帯に分け、それぞれ独立にスケーリングします。グローバル構造とディテールを別軸で制御できます。

論文: [arXiv:2504.02154](https://arxiv.org/abs/2504.02154)（CVPR 2025 GMCV）  
原実装: [WikiChao/FreSca](https://github.com/WikiChao/FreSca)

> `l = h = 1` のとき、通常 CFG と完全に等価です。

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
| Low-freq Scale (l)  | 1.00 | 低周波成分（グローバル構造・構図）の乗数 |
| High-freq Scale (h) | 1.25 | 高周波成分（ディテール・エッジ・テクスチャ）の乗数 |
| Cutoff Mode         | Fixed radius | `Fixed radius`：固定のFFTインデックス半径。バッチ全体・全チャンネルで共通。`Energy-based (adaptive)`：サンプルごと・チャンネルごとに独立して半径を解き、累積パワースペクトラムエネルギーが目標比率（`r0`）に達する半径を用いる |
| Freq Cutoff         | 20   | 低帯と高帯を分ける DC からの半径。ComfyUI 既定値 = 20。**Fixed radius モード専用** |
| Energy Ratio (r0)   | 0.90 | 低帯・高帯の境界を定める目標累積エネルギー比率。原論文（arXiv:2504.02154, Section 4.1）のText-to-Image向けenergy-basedカットオフ設定と同じ値。**Energy-based (adaptive) モード専用** |

`h` を 1.0 より大きくするとディテールとテクスチャが鮮明になり、構図には影響しません。  
`l` を 1.0 より大きくするとグローバル構造やプロンプト追従性が強まります。  
どちらか一方を 1.0 にすればその帯は変化なし、両立調整が可能です。

---

## カットオフモード

**Fixed radius**（既定）は固定のFFTインデックス半径を使い、バッチ内の全サンプル・全チャンネルで同一です。原論文のspatial-ratio方式のカットオフに相当します。

**Energy-based (adaptive)** は、半径をサンプルごと・チャンネルごとに独立して解き、その半径内の累積パワースペクトラムエネルギーが目標比率`r0`に達するようにします。原論文のenergy-basedカットオフ（arXiv:2504.02154, Section 4.1 / Fig. 8–9）に相当します。チャンネルを集約せず独立させているのは意図的な設計です。SDXLのVAE潜在空間とAnimaのflow-matching潜在空間で、チャンネルごとの周波数特性が同じとは限らないため、単一の固定半径、あるいはチャンネル間で共有される単一半径を前提にしない設計にしています。

```
F           = fftshift( fft2(delta) )                     # (B, C, H, W) complex
power       = |F|²
radius[b,c] = min r such that Σ_{|freq| ≤ r} power[b,c] ≥ r0 × Σ power[b,c]
low_mask    = (radius_from_DC ≤ radius[b,c])              # サンプルごと・チャンネルごと
```

DCからの距離マップとその半径ビンインデックスは(H, W, device)にのみ依存するため、キャッシュして毎ステップ使い回します。毎ステップ再計算するのはエネルギー集計部分のみです。再構築部分（`scale_map`、`delta_scaled`、`output`）はFixed radiusモードと同一で、マスクがサンプル・チャンネルごとに変わる点だけが異なります。

---

## アルゴリズム

```
delta         = cond_denoised − uncond_denoised
F             = fftshift( fft2(delta) )          # DC を中央に移動
low_mask      = (DC からの半径 ≤ freq_cutoff)
scale_map     = l × low_mask + h × (1 − low_mask)
delta_scaled  = ifft2( ifftshift( F × scale_map ) ).real
output        = uncond_denoised + cfg_scale × delta_scaled
```

ガイダンス差分を周波数領域で分解し、低周波帯と高周波帯を独立にスケーリングしたあと再構築します。  
`l = h = 1` のとき出力は通常 CFG とビット同一になります。

---

## 他拡張との併用

FreSca（Post-CFG、`sorting_priority = 15.2`）は CFG 演算コアの直後、MaHiRo（15.5）の前に位置します。  
推奨スタック順：

```
TCFG（Pre-CFG） → CFG コア → FreSca（Post-CFG 15.2） → MaHiRo（Post-CFG 15.5）
```

TCFG と FreSca は干渉しません。TCFG は CFG 演算前にガイダンスベクトルの*方向*を補正し、FreSca は CFG 演算後にガイダンス差分の*周波数構成*を整形するため、介入軸が直交しています。  
複数の CFG 軸拡張を重ねる場合は CFG を 7〜15 の範囲に抑えることを推奨します。

### Anima（flow-matching / DiT）について

FreSca は Post-CFG テンソルレベルで動作しアーキテクチャに依存しません。  
ただし Anima は通常 CFG = 1.0 で運用するため、fast-path（`denoised` をそのまま返す）が発動し、実質的に無効になります。  
CFG を 1.0 より大きく設定した場合に限り FreSca が有効になります。

有効な場合、Animaで想定しているのは Energy-based (adaptive) モードです。低帯・高帯の境界をチャンネルごとに独立して解くため、アーキテクチャ間で意味を持つ単一の固定半径を仮定しません。SDXLのVAE潜在空間とAnimaのflow-matching潜在空間で、チャンネルごとの周波数特性が同じとは限らないためです。

---

## 実装上の注意点 — `process_before_every_sampling()` の使用について

Forge 系 WebUI は `process()` の実行後、サンプリング開始前に `forge_objects.unet` を再構築します。  
そのため `process()` 内で登録したフックは再構築時に消えてしまい、サンプリング中に一切呼ばれません。

本拡張はフック登録を `process_before_every_sampling()` で行っています。このタイミングでは `forge_objects.unet` が `CFGDenoiser` の参照先と同一オブジェクトになっており、全ステップ・Hires.fix パスを含めて確実にフックが呼ばれます。

この問題は、登録時と `CFGDenoiser` 呼び出し時の unet オブジェクトの `id()` を照合することで確認しました。二者が異なる別オブジェクトであることを突き止めるまでに相当な調査を要した経緯があります。  
**Post-CFG フックを登録する拡張は `process()` ではなく `process_before_every_sampling()` を使うべきです。**

---

## 動作確認環境

- reForge（Python 3.10）— SDXL 系モデル
- Forge Neo（Python 3.12）— SDXL 系モデルおよび Anima。txt2img + Hires.fix 確認済み

A1111 非対応（`set_model_sampler_post_cfg_function` は Forge バックエンド専用）。

---

## ライセンス・典拠

MIT License — Original implementation © [WikiChao/FreSca](https://github.com/WikiChao/FreSca)  
Based on: [arXiv:2504.02154](https://arxiv.org/abs/2504.02154) "FreSca: Scaling in Frequency Space Enhances Diffusion Models" (CVPR 2025 GMCV Workshop)

Inspired by **Shiba-2-shiba**'s note article on CFG-related ComfyUI nodes (APG / TCFG / FreSca / MaHiRo):  
[ComfyUIのCFG関連の4ノードの勉強＠APG, TCFG, Fresca, Mahiroノードについて](https://note.com/gentle_murre488/n/nc709aac794bc)

本拡張機能は、**Shiba-2-shiba** さんの note 記事「[ComfyUIのCFG関連の4ノードの勉強＠APG, TCFG, Fresca, Mahiroノードについて](https://note.com/gentle_murre488/n/nc709aac794bc)」から着想を得ています。
