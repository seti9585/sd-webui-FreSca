# sd-webui-FreSca

*English section below. 日本語の説明はページ後半にあります。*

---

# English

## Overview

An extension for Forge-based Stable Diffusion WebUIs that applies FreSca —
frequency-domain scaling of the CFG guidance delta — as a post-CFG hook.

It is designed to coexist with the author's sibling CFG-guidance extensions
([sd-webui-TCFG](https://github.com/seti9585/sd-webui-TCFG) and
[sd-webui-MaHiRo](https://github.com/seti9585/sd-webui-MaHiRo)) so the three can
be combined in a single generation pipeline.

## Compatibility

This extension hooks the Forge backend via `forge_objects.unet` and
`set_model_sampler_post_cfg_function`. These APIs are shared across the Forge
family, so it works on Forge derivatives in general — not just one specific fork.

| WebUI | Status |
|---|---|
| **reForge** (`stable-diffusion-webui-reForge`) | ✅ Verified |
| **Forge Neo** | ✅ Verified |
| **Forge** (lllyasviel original) / Forge Classic | ⚪ Expected to work (same hook API; untested) |
| **A1111** (`stable-diffusion-webui`) | ❌ Not supported |

A1111 lacks both `forge_objects` and `set_model_sampler_post_cfg_function`, so it
is out of scope by design.

## How it works

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

> **Note — Pre-CFG vs Post-CFG:** ComfyUI's native FreSca node is categorised as
> a *Pre-CFG* guidance operation. This extension implements the same
> frequency-domain delta scaling as a *Post-CFG* hook (sort 15.2). Because FreSca
> operates purely on the guidance delta `Δ = cond − uncond`, reconstructing `Δ`
> at post-CFG time is mathematically equivalent — the placement differs, the
> result does not.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| **Low-freq Scale (l)** | 1.00 | Multiplier for low-frequency guidance (global structure). |
| **High-freq Scale (h)** | 1.25 | Multiplier for high-frequency guidance (detail / edges). |
| **Freq Cutoff** | 20 | Radius from DC (shifted-FFT centre) separating low and high bands. Matches the ComfyUI node default. |

## Pipeline position

The sort values are chosen so this extension and its siblings stack in the order
recommended for CFG-guidance refinement (stabilise first, then leap):

```
TCFG          Pre-CFG  (sort 13.0)  SVD projection of uncond
SkimmedCFG    Pre-CFG  (sort 14.0)  outer_influence removal
CFG core      denoised = uncond + scale × (cond − uncond)
FreSca        Post-CFG (sort 15.2)  frequency-scaled guidance delta  ← this extension
MaHiRo        Post-CFG (sort 15.5)  cosine-sim blend CFG vs Leap
```

## Requirements

- A Forge-based WebUI (reForge / Forge Neo / Forge — see Compatibility above)
- Python 3.10 / PyTorch (torch.fft available in ≥ 1.7; already present in Forge environments)
- Not compatible with A1111 (`set_model_sampler_post_cfg_function` is Forge-backend only)

## File structure

```
sd-webui-FreSca/
├── scripts/sd_webui_fresca.py   # UI + hook registration
├── sd_webui_fresca/__init__.py
├── sd_webui_fresca/core.py      # FFT algorithm
├── requirements.txt
├── LICENSE
└── README.md
```

## Credits & References

**Paper**

> Chao Huang, Susan Liang, Yunlong Tang, Jing Bi, Li Ma, Yapeng Tian, Chenliang Xu.
> "FreSca: Scaling in Frequency Space Enhances Diffusion Models."
> arXiv:2504.02154 (CVPR 2025 GMCV Workshop). Paper licensed under CC BY 4.0.

```bibtex
@article{huang2025fresca,
    title  = {FreSca: Unveiling the Scaling Space in Diffusion Models},
    author = {Huang, Chao and Liang, Susan and Tang, Yunlong and Ma, Li and Tian, Yapeng and Xu, Chenliang},
    journal= {arXiv preprint arXiv:2504.02154},
    year   = {2025}
}
```

**Reference implementations**

- Original FreSca algorithm: [WikiChao/FreSca](https://github.com/WikiChao/FreSca) — MIT License. The FFT-based frequency-scaling logic in `core.py` is derived from this reference.
- Forge port reference: [Shiba-2-shiba/TCFG-APG-Mahiro-for-ForgeClassic](https://github.com/Shiba-2-shiba/TCFG-APG-Mahiro-for-ForgeClassic) — the Forge-backend hook approach (post-CFG function registration on `forge_objects.unet`) was adapted from this implementation.
- Background article: shiba*2, [Study of ComfyUI's four CFG-related nodes: APG, TCFG, FreSca, MaHiRo](https://note.com/gentle_murre488/n/nc709aac794bc) — explains how APG / TCFG / FreSca / MaHiRo interact and why combining them is beneficial.

**Sibling extensions**

- [sd-webui-TCFG](https://github.com/seti9585/sd-webui-TCFG)
- [sd-webui-MaHiRo](https://github.com/seti9585/sd-webui-MaHiRo)

## License

This extension is released under the MIT License — see [LICENSE](./LICENSE).

The original FreSca reference implementation ([WikiChao/FreSca](https://github.com/WikiChao/FreSca))
is also MIT-licensed; this project follows the same terms for compatibility.
The underlying paper (arXiv:2504.02154) is licensed under CC BY 4.0.

---

# 日本語

## 概要

Forge 派生の Stable Diffusion WebUI 向けの拡張機能で、FreSca —
CFG ガイダンス差分の周波数領域スケーリング — を post-CFG フックとして適用する。

作者の姉妹拡張である CFG ガイダンス系拡張
（[sd-webui-TCFG](https://github.com/seti9585/sd-webui-TCFG) と
[sd-webui-MaHiRo](https://github.com/seti9585/sd-webui-MaHiRo)）と
共存するよう設計されており、3つを単一の生成パイプラインで併用できる。

## 対応 WebUI

本拡張は Forge 系バックエンドの `forge_objects.unet` と
`set_model_sampler_post_cfg_function` を利用してフックする。これらの API は
Forge ファミリー全体で共有されているため、特定の派生に依存せず Forge 派生
WebUI 全般で動作する。

| WebUI | 状態 |
|---|---|
| **reForge** (`stable-diffusion-webui-reForge`) | ✅ 動作確認済み |
| **Forge Neo** | ✅ 動作確認済み |
| **Forge**（lllyasviel オリジナル）/ Forge Classic | ⚪ 動作する見込み（同一フック API・未検証） |
| **A1111** (`stable-diffusion-webui`) | ❌ 非対応 |

A1111 は `forge_objects` と `set_model_sampler_post_cfg_function` のどちらも
持たないため、設計上対象外。

## 仕組み

通常の CFG は次のように計算される。

```
denoised = uncond + scale × (cond − uncond)
```

FreSca はガイダンス差分 `Δ = cond − uncond` を 2D FFT で周波数領域に変換し、
低周波（グローバル構造）と高周波（ディテール・エッジ）に分割して独立した
スケーリング係数を適用したのち空間領域に戻す。

```
Δ̂ = F⁻¹( l · M_low ⊙ F(Δ)  +  h · M_high ⊙ F(Δ) )
denoised = uncond + scale × Δ̂
```

`l = h = 1` のとき通常 CFG と等価。

> **補足 — Pre-CFG と Post-CFG について：** ComfyUI のネイティブ FreSca ノードは
> *Pre-CFG* のガイダンス操作に分類されている。本拡張は同じ周波数領域での差分
> スケーリングを *Post-CFG* フック（sort 15.2）として実装している。FreSca は
> ガイダンス差分 `Δ = cond − uncond` のみを操作するため、post-CFG 時点で `Δ` を
> 再構成しても数学的には等価であり、適用位置は異なるが結果は変わらない。

## パラメータ

| パラメータ | 既定値 | 説明 |
|---|---|---|
| **Low-freq Scale (l)** | 1.00 | 低周波ガイダンス（グローバル構造）の乗数。 |
| **High-freq Scale (h)** | 1.25 | 高周波ガイダンス（ディテール・エッジ）の乗数。 |
| **Freq Cutoff** | 20 | 低周波帯と高周波帯を分ける、DC（シフト後 FFT の中心）からの半径。ComfyUI ノードの既定値に準拠。 |

## 処理順序

sort 値は、本拡張と姉妹拡張が CFG ガイダンス調整に推奨される順序
（まず安定化し、その後にリープ）で積み重なるように選んでいる。

```
TCFG          Pre-CFG  (sort 13.0)  uncond の SVD 射影
SkimmedCFG    Pre-CFG  (sort 14.0)  outer_influence の除去
CFG core      denoised = uncond + scale × (cond − uncond)
FreSca        Post-CFG (sort 15.2)  周波数スケーリングしたガイダンス差分  ← 本拡張
MaHiRo        Post-CFG (sort 15.5)  CFG と Leap のコサイン類似度ブレンド
```

## 動作環境

- Forge 系の WebUI（reForge / Forge Neo / Forge — 上記「対応 WebUI」参照）
- Python 3.10 / PyTorch（torch.fft は 1.7 以降で利用可能。Forge 環境には既に含まれる）
- A1111 非対応（`set_model_sampler_post_cfg_function` は Forge バックエンド専用）

## ファイル構成

```
sd-webui-FreSca/
├── scripts/sd_webui_fresca.py   # UI + フック登録
├── sd_webui_fresca/__init__.py
├── sd_webui_fresca/core.py      # FFT アルゴリズム
├── requirements.txt
├── LICENSE
└── README.md
```

## 出典・クレジット

**論文**

> Chao Huang, Susan Liang, Yunlong Tang, Jing Bi, Li Ma, Yapeng Tian, Chenliang Xu.
> "FreSca: Scaling in Frequency Space Enhances Diffusion Models."
> arXiv:2504.02154（CVPR 2025 GMCV Workshop）。論文は CC BY 4.0 で公開。

```bibtex
@article{huang2025fresca,
    title  = {FreSca: Unveiling the Scaling Space in Diffusion Models},
    author = {Huang, Chao and Liang, Susan and Tang, Yunlong and Ma, Li and Tian, Yapeng and Xu, Chenliang},
    journal= {arXiv preprint arXiv:2504.02154},
    year   = {2025}
}
```

**参考実装**

- FreSca のオリジナルアルゴリズム：[WikiChao/FreSca](https://github.com/WikiChao/FreSca) — MIT ライセンス。`core.py` の FFT ベースの周波数スケーリング処理はこの実装を参考にしている。
- Forge 移植の参考：[Shiba-2-shiba/TCFG-APG-Mahiro-for-ForgeClassic](https://github.com/Shiba-2-shiba/TCFG-APG-Mahiro-for-ForgeClassic) — Forge バックエンドのフック手法（`forge_objects.unet` への post-CFG 関数登録）はこの実装を参考にしている。
- 背景記事：shiba*2「[ComfyUIのCFG関連の4ノードの勉強＠APG, TCFG, Fresca, Mahiroノードについて](https://note.com/gentle_murre488/n/nc709aac794bc)」 — APG / TCFG / FreSca / MaHiRo の相互作用と併用のメリットを解説している。

**姉妹拡張**

- [sd-webui-TCFG](https://github.com/seti9585/sd-webui-TCFG)
- [sd-webui-MaHiRo](https://github.com/seti9585/sd-webui-MaHiRo)

## ライセンス

本拡張は MIT ライセンスで公開する — [LICENSE](./LICENSE) を参照。

原実装である [WikiChao/FreSca](https://github.com/WikiChao/FreSca) も MIT ライセンスで
あり、互換性のため本プロジェクトも同一条件を採用している。
基盤となる論文（arXiv:2504.02154）は CC BY 4.0 で公開されている。
