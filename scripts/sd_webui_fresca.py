"""sd-webui-FreSca — Post-CFG frequency scaling for reForge.

Paper  : arXiv:2504.02154  (CVPR 2025 GMCV)
Hook   : set_model_sampler_post_cfg_function
Sort   : 15.2  (after CFG core, before MaHiRo 15.5)
"""

import gradio as gr
import modules.scripts as scripts

from sd_webui_fresca.core import (
    FRESCA_HOOK_QUALNAME,
    apply_fresca,
    remove_fresca_patches,
)


class FreScaScript(scripts.Script):
    """FreSca: independent low/high-frequency scaling of the CFG guidance delta."""

    sorting_priority = 15.2

    def title(self) -> str:
        return "FreSca"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(label="FreSca", open=False):
            gr.HTML(
                "<p><i>"
                "<b>Post-CFG</b>: Scales the CFG guidance delta independently "
                "in low-frequency and high-frequency bands via 2D FFT. "
                "Requires Forge backend."
                "</i></p>"
            )
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

    def process(self, p, enabled, scale_low, scale_high, freq_cutoff):
        # Write metadata once per generation (before sampling starts)
        if not enabled:
            return
        p.extra_generation_params["fresca_enabled"]      = True
        p.extra_generation_params["fresca_scale_low"]    = float(scale_low)
        p.extra_generation_params["fresca_scale_high"]   = float(scale_high)
        p.extra_generation_params["fresca_freq_cutoff"]  = int(freq_cutoff)

    def process_before_every_sampling(self, p, *args, **kwargs):
        # Hook registration runs here — at this point forge_objects.unet is
        # the same object cfg_denoiser will reference during sampling.
        if len(args) < 4:
            return
        enabled     = bool(args[0])
        scale_low   = float(args[1])
        scale_high  = float(args[2])
        freq_cutoff = int(args[3])

        if not enabled:
            return

        _l   = scale_low
        _h   = scale_high
        _cut = freq_cutoff

        def fresca_hook(args):
            return apply_fresca(args, scale_low=_l, scale_high=_h, freq_cutoff=_cut)

        fresca_hook.__qualname__ = FRESCA_HOOK_QUALNAME

        unet = p.sd_model.forge_objects.unet.clone()
        remove_fresca_patches(unet)
        unet.set_model_sampler_post_cfg_function(fresca_hook)
        p.sd_model.forge_objects.unet = unet
