import warnings
from datetime import timedelta
import copy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoProcessor
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from tqdm import tqdm
from decord import VideoReader, cpu

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav

from loguru import logger as eval_logger

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria2,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model_voco import VoCoLlamaForVideo
from llava.train.train_voco_video import process_text

torch.backends.cuda.matmul.allow_tf32 = True
warnings.filterwarnings("ignore")


@register_model("voco_base")
class VoCoLlamaWrapper(lmms):
    def __init__(
            self,
            pretrained: str,
            device: Optional[str] = "cuda:0",
            batch_size: Optional[Union[int, str]] = 1,
            attn_implementation="sdpa",
            # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
            device_map="cuda:0",
            conv_template="vicuna_v1",
            use_cache=True,
            truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
            max_frames_num: int = 3,
            mm_resampler_type: str = "spatial_pool",
            mm_spatial_pool_stride: int = 2,
            mm_spatial_pool_out_channels: int = 1024,
            mm_spatial_pool_mode: str = "average",
            mm_newline_position: str = "grid",
            mm_pooling_position: str = "after",
            overwrite: bool = True,
            video_decode_backend: str = "pyav",
            delay_load: bool = False,
            tie_weights: bool = True,
            **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Initialize accelerator
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        # Instance variables
        self.pretrained = pretrained
        self.model_name = get_model_name_from_path(pretrained)
        self.video_decode_backend = video_decode_backend
        self.overwrite = overwrite
        self.mm_resampler_type = mm_resampler_type
        self.mm_spatial_pool_stride = int(mm_spatial_pool_stride)
        self.mm_spatial_pool_out_channels = int(mm_spatial_pool_out_channels)
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.max_frames_num = int(max_frames_num)
        self.mm_resampler_location = mm_pooling_position
        self.mm_newline_position = mm_newline_position
        self.delay_load = delay_load

        # Fetch & modify model config
        cfg_pretrained = VoCoLlamaForVideo.config_class.from_pretrained(pretrained)
        if self.overwrite:
            overwrite_config = {
                "attn_implementation": attn_implementation,
            }
            if overwrite_config is not None:
                eval_logger.log("INFO", f"Overwriting config with {overwrite_config}")
                for k, v in overwrite_config.items():
                    setattr(cfg_pretrained, k, v)

        # Initialize model
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=False)
        self._model = VoCoLlamaForVideo.from_pretrained(
            pretrained,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
            config=cfg_pretrained
        )
        self._model.get_vision_tower().load_model(device_map=self.device_map)
        self._model.get_vision_tower().eval()
        self._max_length = cfg_pretrained.tokenizer_model_max_length
        self._image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336").image_processor
        self.model.eval()
        self._config = self._model.config
        if tie_weights:
            self.model.tie_weights()
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU,
                                                    DistributedType.DEEPSPEED], \
                "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED."
                                 "Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP \
                    or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for context, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            videos = []
            for visual in visuals:
                video = self.load_video(visual, self.max_frames_num)
                video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(
                    torch.bfloat16).cuda()
                videos.append(video)

            qs = context
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
                0).to(self.device)

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], continuation)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
                0).cuda()

            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100

            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, videos=videos)

            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1]:]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1]: input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

            #######################################
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
            res.append((seq_log_prob, is_greedy.item()))
            #######################################
        pbar.close()
        return res

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            videos = []
            try:
                for visual in visuals:
                    if self.video_decode_backend == "decord":
                        video = self.load_video(visual, self.max_frames_num)
                    elif self.video_decode_backend == "pyav":
                        video = read_video_pyav(visual, num_frm=self.max_frames_num)
                    # video = self.load_video(visual, self.max_frames_num)
                    # video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
                    video = process_images(video, self._image_processor, self.config).half().cuda()
                    videos.append(video)
            except Exception as e:
                eval_logger.info(f"{e}")
                eval_logger.info(f"Video {visuals} can not load, check the source")
                video_path = "\n".join(visuals)
                res.append(f"Video {video_path} can not load, check the source")
                pbar.update(1)
                continue

            videos = torch.stack(videos)
            qs = contexts
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN * len(videos) + "\n" + qs

            # This is much safer for llama3, as we now have some object type in it
            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            ### Important!!!! ###
            prompt = f"<image>\n{'<voco>' * self.config.num_voco_tokens}\n{conv.get_prompt()}"
            #####################
            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            if "llama_3" in self.conv_template:
                pad_token_ids = 0  # lmms-lab/llama3-llava-8b is trained on this pad token id. You may need to customize this for other models.
            attention_masks = input_ids.ne(pad_token_ids).long().cuda()

            keywords = [conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2]
            keywords += ['.', "'", ',']
            stopping_criteria = KeywordsStoppingCriteria2(keywords, self.tokenizer)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            with torch.inference_mode(), self.accelerator.autocast():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    videos=videos,
                    attention_mask=attention_masks,
                    use_cache=self.use_cache,
                    stopping_criteria=[stopping_criteria],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            res.append(outputs)
            pbar.update(1)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def flatten(self, input_):
        new_list = []
        for i in input_:
            for j in i:
                new_list.append(j)
        return new_list

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def load_video(self, video_path, max_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        # fps = round(vr.get_avg_fps())
        # frame_idx = [i for i in range(0, len(vr), fps)]
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length
