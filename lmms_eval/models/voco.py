import torch
from transformers import AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True

import copy
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

try:
    from llavadev.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llavadev.conversation import conv_templates
    from llavadev.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llavadev import VoCoLlamaForVideo
except Exception as e:
    eval_logger.debug("LLaVA is not installed. Please install LLaVA to use this model.\nError: %s" % e)


@register_model("voco-llama")
class VoCoLlamaWrapper(lmms):
    def __init__(self, pretrained='path/to/your/model', device='cuda', tokenizer=None):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained) if tokenizer is None else tokenizer
        self.model = VoCoLlamaForVideo.from_pretrained(pretrained).to(self.device)
        self.model.eval()

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        results = []
        for instance in requests:
            context, doc_to_target, doc_to_visual, doc_id, task, split = instance.args

            # Process video if present
            video = None
            if doc_to_visual:
                video = doc_to_visual(doc_id)
                if video is not None:
                    video = video.unsqueeze(0).to(self.device)

            # Tokenize input
            inputs = self.tokenizer(context, return_tensors="pt").to(self.device)

            # Get target
            target = doc_to_target(doc_id)
            target_tokens = self.tokenizer(target, return_tensors="pt").input_ids.to(self.device)

            # Generate logits
            with torch.no_grad():
                outputs = self.model(**inputs, videos=video, labels=target_tokens)

            # Calculate log likelihood
            logits = outputs.logits[:, -target_tokens.shape[1]:].contiguous()
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
            seq_log_prob = token_log_probs.sum().item()

            # Check if it's the greedy sequence
            is_greedy = torch.argmax(logits, dim=-1).equal(target_tokens)

            results.append((seq_log_prob, is_greedy.item()))

        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []
        for instance in requests:
            context, all_gen_kwargs, doc_to_visual, doc_id, task, split = instance.args

            # Process video if present
            video = None
            if doc_to_visual:
                video = doc_to_visual(doc_id)
                if video is not None:
                    video = video.unsqueeze(0).to(self.device)

            # Tokenize input
            inputs = self.tokenizer(context, return_tensors="pt").to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    videos=video,
                    **all_gen_kwargs
                )

            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(generated_text)

        return results

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
