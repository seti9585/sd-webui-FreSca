"""sd-webui-FreSca — Post-CFG frequency scaling for reForge.

Paper  : arXiv:2504.02154  (CVPR 2025 GMCV)
Hook   : set_model_sampler_post_cfg_function
Sort   : 15.2  (after CFG core, before MaHiRo 15.5)
"""

import gradio as gr
import modules.scripts as scripts

from sd_webui_fresca.core import apply_fresca


class FreScaScript(scripts.Script):
    """FreSca: independent low/high-frequency scaling of the CFG guidance delta."""

    sorting_priority = 15.2

    # ── Script metadata ────────────────────────────────────────────────────
    def title(self) -> str:
        return "FreSca"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # ── Gradio UI ──────────────────────────────────────────────────────────
    def ui(self, is_img2img):
        with gr.Accordion(label="FreSca", open=False):
            enabled = gr.Checkbox(label="Enable FreSca", value=False)
            with gr.Row():
                scale_low = gr.Slider(
                    label="Low-freq Scale (l)",
                    minimum=0.0,
                    maximum=3.0,
                    step=0.01,
                    value=1.0,
                    info="Multiplier for low-frequency (global structure) guidance",
                )
                scale_high = gr.Slider(
                    label="High-freq Scale (h)",
                    minimum=0.0,
                    maximum=3.0,
                    step=0.01,
                    value=1.25,
                    info="Multiplier for high-frequency (detail / edge) guidance",
                )
            freq_cutoff = gr.Slider(
                label="Freq Cutoff (FFT index radius)",
                minimum=1,
                maximum=200,
                step=1,
                value=20,
                info=(
                    "Radius from DC (centre of shifted spectrum) that divides "
                    "low and high bands. ComfyUI default = 20."
                ),
            )
        return [enabled, scale_low, scale_high, freq_cutoff]

    # ── Hook registration ──────────────────────────────────────────────────
    def process(
        self,
        p,
        enabled: bool,
        scale_low: float,
        scale_high: float,
        freq_cutoff: int,
    ):
        if not enabled:
            return

        _l   = float(scale_low)
        _h   = float(scale_high)
        _cut = int(freq_cutoff)

        def fresca_hook(args):
            return apply_fresca(args, scale_low=_l, scale_high=_h, freq_cutoff=_cut)

        fresca_hook.__qualname__ = "fresca_hook"

        unet = p.sd_model.forge_objects.unet.clone()
        unet.set_model_sampler_post_cfg_function(fresca_hook)
        p.sd_model.forge_objects.unet = unet
