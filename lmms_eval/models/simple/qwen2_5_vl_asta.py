"""Qwen2.5-VL with ASTA prefill attention and LoRA adapters for lmms-eval.

Registers as model "qwen2_5_vl_asta". Usage:
    --model qwen2_5_vl_asta
    --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,lora_path=outputs/sft_default/final_lora,spatial_window=5,max_num_frames=32"
"""

import os
import sys
from typing import Optional, Union

import torch
from loguru import logger as eval_logger
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration

from lmms_eval.api.registry import register_model

# Ensure project root is on path for ASTA imports
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import the base class we're extending
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL


@register_model("qwen2_5_vl_asta")
class Qwen2_5_VL_ASTA(Qwen2_5_VL):
    """Qwen2.5-VL with ASTA prefill + LoRA, inheriting all eval logic from base."""

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        lora_path: Optional[str] = None,
        spatial_window: Union[int, str] = 5,
        enable_asta: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        # model_args string parsing gives us strings — cast to correct types
        self._lora_path = lora_path
        self._spatial_window = int(spatial_window)
        self._enable_asta = str(enable_asta).lower() not in ("false", "0", "no")

        # Parent __init__ loads the base model
        super().__init__(pretrained=pretrained, **kwargs)

        # Apply LoRA if provided
        if self._lora_path:
            eval_logger.info(f"Loading LoRA adapters from {self._lora_path}")
            # unwrap to get base model for LoRA application
            base_model = self.model
            base_model = PeftModel.from_pretrained(base_model, self._lora_path)
            base_model.eval()
            # Update the internal model reference
            self._model = base_model
            eval_logger.info("LoRA adapters loaded successfully")

        # Setup ASTA attention (prefill only — decode falls back to SDPA)
        if self._enable_asta:
            eval_logger.info(f"Setting up ASTA attention (spatial_window={self._spatial_window})")
            from training.qwen_vl.utils import setup_asta_attention

            setup_asta_attention(self.model, spatial_window=self._spatial_window)
            eval_logger.info("ASTA attention enabled for prefill")
