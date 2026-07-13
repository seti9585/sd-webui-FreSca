"""sd-webui-FreSca — Post-CFG frequency scaling for reForge.

Paper  : arXiv:2504.02154  (CVPR 2025 GMCV)
Hook   : set_model_sampler_post_cfg_function
Sort   : 15.2  (after CFG core, before MaHiRo 15.5)
"""

import gradio as gr
import modules.scripts as scripts

from sd_webui_fresca.core import (
    MARKER,
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
            gr.HTML("<p style='margin:4px 0 8px'>Post-CFG — frequency-domain guidance scaling</p>")
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
            cutoff_mode = gr.Radio(
                choices=["Fixed radius", "Energy-based (adaptive)"],
                value="Fixed radius",
                label="Cutoff Mode",
                info=(
                    "Fixed: constant FFT-index radius, shared across the whole "
                    "batch and all channels. Energy-based: radius solved "
                    "independently per sample and per channel so cumulative "
                    "power-spectrum energy reaches the target ratio (r0)."
                ),
            )
            freq_cutoff = gr.Slider(
                label="Freq Cutoff (FFT index radius)",
                minimum=1,
                maximum=200,
                step=1,
                value=20,
                info=(
                    "Radius from DC (centre of shifted spectrum) that divides "
                    "low and high bands. ComfyUI default = 20. Only used in "
                    "Fixed radius mode."
                ),
                visible=True,
            )
            r0 = gr.Slider(
                label="Energy Ratio (r0)",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.9,
                info=(
                    "Target cumulative power-spectrum energy ratio that "
                    "defines the low/high boundary, solved independently per "
                    "sample and per channel. Only used in Energy-based mode."
                ),
                visible=False,
            )

            def _toggle_cutoff_ui(mode):
                is_energy = (mode == "Energy-based (adaptive)")
                return (
                    gr.update(visible=not is_energy),
                    gr.update(visible=is_energy),
                )

            cutoff_mode.change(
                fn=_toggle_cutoff_ui,
                inputs=[cutoff_mode],
                outputs=[freq_cutoff, r0],
            )

        # Infotext round-trip (PNG Info -> Send to txt2img / img2img).
        # Keys must match those written to p.extra_generation_params in process().
        # The Enable checkbox uses a callable instead of a plain key string:
        # infotext paste leaves a component untouched when its key is absent
        # (Forge Neo -> gr.skip(), reForge -> gr.update() no-op), so a bare key
        # could never turn FreSca OFF. The callable returns False on a missing
        # key, forcing OFF when an image generated WITHOUT FreSca is sent,
        # which is required for faithful same-seed reproduction.
        #
        # cutoff_mode also uses a callable, for two reasons: (1) it is a
        # label-string component, but process() stores the internal token
        # ("fixed" / "energy"), not the label, so the raw key value needs
        # translation back to the label the Radio expects; (2) the same
        # forced-default reasoning as Enable applies — a PNG generated before
        # this feature existed has no fresca_cutoff_mode key at all, and
        # should restore to "Fixed radius" (the only mode that ever existed),
        # not be left untouched.
        self.infotext_fields = [
            (enabled,     lambda d: d.get("fresca_enabled", "False") == "True"),
            (scale_low,   "fresca_scale_low"),
            (scale_high,  "fresca_scale_high"),
            (cutoff_mode, lambda d: "Energy-based (adaptive)"
                                     if d.get("fresca_cutoff_mode") == "energy"
                                     else "Fixed radius"),
            (freq_cutoff, "fresca_freq_cutoff"),
            (r0,          "fresca_r0"),
        ]

        return [enabled, scale_low, scale_high, cutoff_mode, freq_cutoff, r0]

    def process(self, p, enabled, scale_low, scale_high, cutoff_mode, freq_cutoff, r0):
        # Write metadata once per generation (before sampling starts).
        # Only written when enabled; absence is resolved to OFF on paste by the
        # Enable callable registered in ui(), so no fresca_enabled: False is
        # emitted for disabled runs (keeps infotext free of disabled-extension keys).
        if not enabled:
            return
        p.extra_generation_params["fresca_enabled"]     = True
        p.extra_generation_params["fresca_scale_low"]   = float(scale_low)
        p.extra_generation_params["fresca_scale_high"]  = float(scale_high)
        p.extra_generation_params["fresca_cutoff_mode"] = (
            "energy" if cutoff_mode == "Energy-based (adaptive)" else "fixed"
        )
        if cutoff_mode == "Energy-based (adaptive)":
            p.extra_generation_params["fresca_r0"] = float(r0)
        else:
            p.extra_generation_params["fresca_freq_cutoff"] = int(freq_cutoff)

    def process_before_every_sampling(self, p, *args, **kwargs):
        # Hook registration runs here — at this point forge_objects.unet is
        # the same object cfg_denoiser will reference during sampling.
        if len(args) < 6:
            return
        enabled     = bool(args[0])
        scale_low   = float(args[1])
        scale_high  = float(args[2])
        cutoff_mode = str(args[3])
        freq_cutoff = int(args[4])
        r0          = float(args[5])

        if not enabled:
            return

        _l    = scale_low
        _h    = scale_high
        _mode = "energy" if cutoff_mode == "Energy-based (adaptive)" else "fixed"
        _cut  = freq_cutoff
        _r0   = r0

        def fresca_hook(args):
            return apply_fresca(
                args,
                scale_low=_l,
                scale_high=_h,
                cutoff_mode=_mode,
                freq_cutoff=_cut,
                r0=_r0,
            )

        fresca_hook._sd_webui_fresca_marker = MARKER

        unet = p.sd_model.forge_objects.unet.clone()
        remove_fresca_patches(unet)
        unet.set_model_sampler_post_cfg_function(fresca_hook)
        p.sd_model.forge_objects.unet = unet
