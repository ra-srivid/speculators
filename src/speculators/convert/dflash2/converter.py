"""DFlash2 checkpoint converter.

Converts an external DFlash2 checkpoint (e.g. ``z-lab/*-DFlash2``) to a Speculators
checkpoint that loads with ``DFlash2DraftModel.from_pretrained(path)``, so it can be
fine-tuned. vLLM already serves the external format directly.

The draft body, convolutions, and candidate selector already match
``DFlash2DraftModel`` key for key, so the weight handling is the DFlash converter's.
What differs is the config: the convolution and selector settings live in
``dflash_config``. Causality is mapped as for DFlash.
"""

from typing import Any, ClassVar

from speculators.convert.dflash.converter import DFlashConverter
from speculators.models.dflash2 import DFlash2DraftModel, DFlash2SpeculatorConfig

__all__ = ["DFlash2Converter"]

_DFLASH2_FIELDS = (
    "conv_kernel_size",
    "conv_group_size",
    "selector_rank",
    "selector_top_k",
)

# dflash_config scale fields vLLM applies at serving time that DFlash2DraftModel
# does not model, with the value that makes each a no-op
_UNMODELED_FIELD_DEFAULTS: dict[str, Any] = {
    "input_embedding_scale": 1.0,
    "output_multiplier": 1.0,
}
# vLLM disables softcapping for any falsy value (absent, null, or 0)
_SOFTCAP_KEY = "final_logit_softcapping"


class DFlash2Converter(DFlashConverter):
    """Convert an external DFlash2 checkpoint to speculators format."""

    algorithm: ClassVar[str] = "dflash2"
    config_class: ClassVar[type[DFlash2SpeculatorConfig]] = DFlash2SpeculatorConfig  # type: ignore[assignment]
    model_class: ClassVar[type[DFlash2DraftModel]] = DFlash2DraftModel  # type: ignore[assignment]

    def _algorithm_config_fields(
        self,
        source_config: dict,
        transformer_config: dict,  # noqa: ARG002
    ) -> dict[str, Any]:
        dflash = source_config.get("dflash_config", {})

        missing = [key for key in _DFLASH2_FIELDS if key not in dflash]
        if missing:
            raise ValueError(
                f"Checkpoint `dflash_config` is missing DFlash2 fields: {missing}"
            )

        attention_mode = dflash.get("attention_mode", "gqa")
        if attention_mode != "gqa":
            raise ValueError(
                f"Unsupported `dflash_config.attention_mode`={attention_mode!r}; "
                "DFlash2DraftModel only has GQA attention."
            )
        for key in ("use_swa", "swa_window_size"):
            if dflash.get(key) is not None:
                raise ValueError(
                    f"Unsupported `dflash_config.{key}`: set sliding attention "
                    "through `layer_types` and `sliding_window` instead."
                )
        if dflash.get("sample_from_anchor"):
            raise ValueError(
                "Unsupported `dflash_config.sample_from_anchor=True`; only "
                "anchor-as-bonus-token checkpoints can be converted."
            )
        unmodeled = {
            key: dflash[key]
            for key, default in _UNMODELED_FIELD_DEFAULTS.items()
            if dflash.get(key, default) != default
        }
        if dflash.get(_SOFTCAP_KEY):
            unmodeled[_SOFTCAP_KEY] = dflash[_SOFTCAP_KEY]
        if unmodeled:
            raise ValueError(
                f"Unsupported non-default `dflash_config` values {unmodeled}: vLLM "
                "applies them at serving time but DFlash2DraftModel does not, so "
                "fine-tuning would train a different model than the one served."
            )

        return {key: dflash[key] for key in _DFLASH2_FIELDS}

    def _sliding_window_non_causal(
        self, source_config: dict, transformer_config: dict
    ) -> bool:
        # vLLM's speculators loader currently applies one causality to every DFlash
        # layer, so mixed per-layer causality would not be served as converted.
        layer_types, causal = self._layer_causality(source_config, transformer_config)
        if any(causal) and not all(causal):
            raise ValueError(
                "Checkpoint runs sliding layers causally and full-attention layers "
                "bidirectionally, which vLLM cannot yet serve from a speculators "
                f"checkpoint (layer_types={layer_types})."
            )
        return super()._sliding_window_non_causal(source_config, transformer_config)
