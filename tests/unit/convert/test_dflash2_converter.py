"""Unit tests for DFlash2Converter config building and checkpoint round trip."""

from typing import Any
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config

from speculators.convert.dflash.converter import _VERIFIER_FILLED_KEYS
from speculators.convert.dflash2.converter import DFlash2Converter
from speculators.models.dflash2 import DFlash2DraftModel, DFlash2SpeculatorConfig

_UNSET = object()
_SLIDING = ["sliding_attention"] * 5
_MIXED = ["full_attention"] + ["sliding_attention"] * 4
_FULL = ["full_attention"] * 5

_VERIFIER_CONFIG = {
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "architectures": ["Qwen3_5ForConditionalGeneration"],
}


def _source_config(dflash_overrides=None, **overrides):
    """Config shaped like z-lab/Qwen3.8-27B-DFlash2."""
    config: dict[str, Any] = {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "is_causal": False,
        "dflash_config": {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 248070,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": [5, 19, 33, 47, 61],
        },
        "vocab_size": 248320,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 5,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_target_layers": 64,
        "layer_types": _SLIDING,
        "sliding_window": 2048,
        "use_sliding_window": True,
        "max_window_layers": 5,
    }
    config["dflash_config"].update(dflash_overrides or {})
    config.update(overrides)
    return config


def _build(source, verifier_config=_VERIFIER_CONFIG) -> DFlash2SpeculatorConfig:
    with patch(
        "speculators.convert.dflash.converter.PretrainedConfig.get_config_dict",
        return_value=(verifier_config, None),
    ):
        config = DFlash2Converter()._build_config(source, "Qwen/Qwen3.8-27B", None)
    assert isinstance(config, DFlash2SpeculatorConfig)
    return config


class TestBuildConfig:
    def test_happy_path(self):
        config = _build(_source_config())

        assert config.speculators_model_type == "dflash2"
        assert config.speculators_config.algorithm == "dflash2"
        assert config.block_size == 8
        assert config.mask_token_id == 248070
        assert config.speculators_config.proposal_methods[0].speculative_tokens == 7  # type: ignore[attr-defined]
        assert config.aux_hidden_state_layer_ids == [6, 20, 34, 48, 62]
        assert config.conv_kernel_size == 2
        assert config.conv_group_size == 16
        assert config.selector_rank == 256
        assert config.selector_top_k == 16
        assert config.sliding_window_non_causal is True
        assert config.sample_from_anchor is False
        tl = config.transformer_layer_config
        assert isinstance(tl, Qwen3Config)
        assert tl.layer_types == _SLIDING
        assert tl.sliding_window == 2048
        assert not hasattr(tl, "is_causal")
        assert not hasattr(tl, "dflash_config")

    @pytest.mark.parametrize(
        ("is_causal", "dflash_causal", "layer_types", "expected"),
        [
            # vLLM fallback: sliding layers causal, full layers bidirectional
            (_UNSET, _UNSET, _SLIDING, False),
            (None, _UNSET, _SLIDING, False),
            (_UNSET, False, _SLIDING, True),
            (_UNSET, True, _SLIDING, False),
            # is_causal applies to every layer and wins over dflash_config.causal
            (False, True, _SLIDING, True),
            (True, _UNSET, _SLIDING, False),
            (False, _UNSET, _MIXED, True),
            (False, _UNSET, _FULL, True),
            # Full layers are bidirectional without an override
            (_UNSET, _UNSET, _FULL, True),
            (None, _UNSET, _FULL, True),
        ],
    )
    def test_causality(self, is_causal, dflash_causal, layer_types, expected):
        source = _source_config(layer_types=layer_types)
        del source["is_causal"]
        if is_causal is not _UNSET:
            source["is_causal"] = is_causal
        if dflash_causal is not _UNSET:
            source["dflash_config"]["causal"] = dflash_causal
        assert _build(source).sliding_window_non_causal is expected

    @pytest.mark.parametrize(
        "source",
        [
            _source_config(is_causal=True, layer_types=_MIXED),
            _source_config(
                is_causal=True,
                layer_types=None,
                use_sliding_window=False,
                sliding_window=None,
            ),
        ],
    )
    def test_causal_full_attention_raises(self, source):
        with pytest.raises(ValueError, match="full-attention draft layers causal"):
            _build(source)

    @pytest.mark.parametrize("is_causal", [_UNSET, None])
    def test_mixed_causality_raises(self, is_causal):
        # Without an override, sliding layers are causal and full layers are not.
        source = _source_config(layer_types=_MIXED)
        del source["is_causal"]
        if is_causal is not _UNSET:
            source["is_causal"] = is_causal
        with pytest.raises(ValueError, match="cannot yet serve"):
            _build(source)

    @pytest.mark.parametrize("missing", ["conv_kernel_size", "selector_top_k"])
    def test_missing_dflash2_field_raises(self, missing):
        source = _source_config()
        del source["dflash_config"][missing]
        with pytest.raises(ValueError, match=missing):
            _build(source)

    @pytest.mark.parametrize(
        ("dflash_overrides", "match"),
        [
            ({"attention_mode": "mla"}, "attention_mode"),
            ({"use_swa": True}, "use_swa"),
            ({"swa_window_size": 1024}, "swa_window_size"),
            ({"sample_from_anchor": True}, "sample_from_anchor"),
            ({"output_multiplier": 0.196}, "output_multiplier"),
            ({"input_embedding_scale": 2.0}, "input_embedding_scale"),
            ({"final_logit_softcapping": 20.0}, "final_logit_softcapping"),
        ],
    )
    def test_unsupported_settings_raise(self, dflash_overrides, match):
        with pytest.raises(ValueError, match=match):
            _build(_source_config(dflash_overrides))

    @pytest.mark.parametrize(
        "dflash_overrides",
        [
            {"attention_mode": "gqa"},
            {"sample_from_anchor": False},
            {"output_multiplier": 1.0, "input_embedding_scale": 1.0},
            {"final_logit_softcapping": None},
            {"final_logit_softcapping": 0.0},
        ],
    )
    def test_no_op_settings_accepted(self, dflash_overrides):
        _build(_source_config(dflash_overrides))


def _tiny_source():
    """A tiny z-lab-format DFlash2 config."""
    return _source_config(
        {"selector_rank": 4, "selector_top_k": 4, "target_layer_ids": [0, 2]},
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_target_layers=4,
        layer_types=["sliding_attention"] * 2,
        sliding_window=8,
        max_window_layers=2,
    )


_TINY_VERIFIER_CONFIG = {"hidden_size": 16, "num_hidden_layers": 4}


class TestRoundTrip:
    def test_converted_checkpoint_reloads_with_source_weights(self, tmp_path):
        source = _tiny_source()
        config = _build(source, _TINY_VERIFIER_CONFIG)
        assert config.aux_hidden_state_layer_ids == [1, 3]

        torch.manual_seed(0)
        weights = {
            key: torch.randn_like(value)
            for key, value in DFlash2DraftModel(config=config).state_dict().items()
            if key not in _VERIFIER_FILLED_KEYS
        }
        assert any(k.startswith("candidate_selector.") for k in weights)
        assert any(".attention_conv." in k for k in weights)

        # Verifier-owned weights are out of scope; keep both steps offline.
        with patch.object(DFlash2DraftModel, "load_verifier_weights"):
            DFlash2Converter()._save(config, weights, tmp_path)
            reloaded = DFlash2DraftModel.from_pretrained(str(tmp_path))
        assert reloaded.config.sliding_window_non_causal is True
        state = reloaded.state_dict()
        for key, value in weights.items():
            assert torch.equal(state[key], value), key

    def test_dflash_weights_rejected(self, tmp_path):
        """A DFlash checkpoint lacks the convolution and selector weights."""
        source = _tiny_source()
        config = _build(source, _TINY_VERIFIER_CONFIG)
        weights = {
            key: value
            for key, value in DFlash2DraftModel(config=config).state_dict().items()
            if key not in _VERIFIER_FILLED_KEYS
            and not key.startswith("candidate_selector.")
        }
        with pytest.raises(ValueError, match="candidate_selector"):
            DFlash2Converter()._save(config, weights, tmp_path)
