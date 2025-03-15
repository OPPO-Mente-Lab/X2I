import os,re
import torch
import torch.distributed as dist
import argparse
import functools
import gc
import logging
import math
import random
import shutil
from pathlib import Path
import copy
import inspect

import datasets
import numpy as np
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import transformers
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, AutoModel

import diffusers

from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module,randn_tensor

from utils.datamodule_qwenvl import DataModuleCustom

from proj import create_proj3_qwen3b, create_proj3_qwen7b

from transformers import Qwen2_5_VLForConditionalGeneration
from typing import Callable, List, Optional, Union
from transformers import T5Tokenizer,MT5EncoderModel,AutoModel,AutoModelForCausalLM
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from transformers import T5EncoderModel, T5TokenizerFast, CLIPTokenizer, CLIPTextModel,T5ForConditionalGeneration,AutoProcessor

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.loaders.lora_pipeline import SD3LoraLoaderMixin
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from transformers.utils import ContextManagers
from core.data.dataloader import Preprocess
from diffusers.utils import get_peft_kwargs, get_adapter_name

import time

from core.pipeline import train_and_infer as tai
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sdxl-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=200000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default = None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument('--qwen_size', type=str, choices=['3b', '7b'], help="Model size selection (3b/7b)")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )


    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser = DataModuleCustom.add_data_specific_args(parser)


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    return args


def getActivationList(lists):
    # the hook signature
    def hook(model, input, output):
        lists.append(output)
    return hook

def getTwoActivationList(list0, list1):
    # the hook signature
    def hook(model, input, output):
        list0.append(output[0])
        list1.append(output[1])
    return hook
    
def get_max_numbered_filename(directory):
    filenames = os.listdir(directory)
    pattern = re.compile(r'\d+')
    numbers = [int(pattern.search(filename).group()) for filename in filenames if pattern.search(filename)]
    return max(numbers) if numbers else None


def cast_hook_list(unet, lists):
    lists.append([])
    lists.append([])
    lists.append([])
    for i,net in enumerate(unet.transformer_blocks):
        net.attn.register_forward_hook(getTwoActivationList(lists[0], lists[1]))

    for i,net in enumerate(unet.single_transformer_blocks):
        net.attn.register_forward_hook(getActivationList(lists[2]))

def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps



class InferPreprocess(Preprocess):

    def __init__(self, infer_pg, is_infer_rank, batch_size, infer_rank):
        self.infer_pg = infer_pg
        self.is_infer_rank = is_infer_rank
        self.bsz_org = batch_size
        self.infer_rank = infer_rank

    def has_gpu_preprocess(self):
        return True
    
    def gpu_preprocess(self, batch, stream):
        train_device = "cuda"
        weight_dtype = torch.bfloat16

        
        input_ids_t5_en = batch["input_ids_t5_en"].to(device=train_device)
        input_ids_en = batch["input_ids_en"].to(device=train_device)
        input_ids_t5 = batch["input_ids_t5"].to(device=train_device)
        attention_mask = batch["attention_mask"].to(device=train_device)
        # image_grid_thw = batch["image_grid_thw"].to(device=train_device)
        # pixel_values = batch["pixel_values"].to(device=train_device)

        # Send the batch data of the training gpu to the inference gpu
        torch.cuda.synchronize()
        tai.send_to_infer_device(input_ids_t5_en, self.infer_pg, self.is_infer_rank, self.infer_rank)
        torch.cuda.synchronize()
        tai.send_to_infer_device(input_ids_en, self.infer_pg, self.is_infer_rank, self.infer_rank)
        torch.cuda.synchronize()

        tai.send_to_infer_device(input_ids_t5, self.infer_pg, self.is_infer_rank, self.infer_rank)
        torch.cuda.synchronize()
       
        tai.send_to_infer_device(attention_mask, self.infer_pg, self.is_infer_rank, self.infer_rank)
        torch.cuda.synchronize()
  
        # tai.send_to_infer_device(pixel_values, self.infer_pg, self.is_infer_rank, self.infer_rank)
        # tai.send_to_infer_device(image_grid_thw, self.infer_pg, self.is_infer_rank, self.infer_rank)

        KD_teacher_tensor0 = torch.zeros((self.bsz_org, 19, 4096, 3072), dtype=weight_dtype, device=train_device)
        KD_teacher_tensor1 = torch.zeros((self.bsz_org, 19, 512, 3072), dtype=weight_dtype, device=train_device)
        KD_teacher_tensor2 = torch.zeros((self.bsz_org, 38, 4608, 3072), dtype=weight_dtype, device=train_device)
        
        latents = torch.zeros((self.bsz_org, 4096, 64), dtype=weight_dtype, device=train_device)
        
        if args.qwen_size == "3b":
            text_embeddings = torch.zeros((self.bsz_org, 37, 512, 2048),dtype=weight_dtype,  device=train_device)
        if args.qwen_size == "7b":
            text_embeddings = torch.zeros((self.bsz_org, 29, 512, 3584),dtype=weight_dtype,  device=train_device)
        timestep = torch.zeros((self.bsz_org), dtype=torch.bfloat16,  device=train_device)
        torch.cuda.synchronize()

        # Get inference results from inference gpu
        KD_teacher_tensor0 = tai.receive_from_infer_device(KD_teacher_tensor0, self.infer_pg, self.is_infer_rank, self.infer_rank)
        torch.cuda.synchronize()
 
        KD_teacher_tensor1 = tai.receive_from_infer_device(KD_teacher_tensor1, self.infer_pg, self.is_infer_rank, self.infer_rank)
        torch.cuda.synchronize()
       
        KD_teacher_tensor2 = tai.receive_from_infer_device(KD_teacher_tensor2, self.infer_pg, self.is_infer_rank, self.infer_rank)
        torch.cuda.synchronize()
     
        latents = tai.receive_from_infer_device(latents, self.infer_pg, self.is_infer_rank, self.infer_rank)
        torch.cuda.synchronize()
      
        text_embeddings = tai.receive_from_infer_device(text_embeddings, self.infer_pg, self.is_infer_rank, self.infer_rank)
        torch.cuda.synchronize()
   
        timestep = tai.receive_from_infer_device(timestep, self.infer_pg, self.is_infer_rank, self.infer_rank)
        torch.cuda.synchronize()
 
        batch["KD_teacher_tensor0"] = KD_teacher_tensor0
        batch["KD_teacher_tensor1"] = KD_teacher_tensor1
        batch["KD_teacher_tensor2"] = KD_teacher_tensor2
        batch["latents"] = latents
        batch["text_embeddings"] = text_embeddings
        batch["timestep"] = timestep

        return batch

def train(args, infer_pg, is_infer_rank, train_pg, group_rank, infer_rank,teacher,paths):
    train_device = torch.device("cuda")
    weight_dtype = torch.bfloat16
    logging_dir = Path(args.output_dir, args.logging_dir)
    is_main_process = dist.get_rank(train_pg) == 0
    is_local_main_process = is_main_process

    rank = dist.get_rank()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

  
    if is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(teacher, subfolder="scheduler")
    torch.cuda.synchronize()

    tokenizer_mllm = AutoProcessor.from_pretrained(paths, padding_side='left', trust_remote_code=True)
    if args.qwen_size == "3b":
        proj_t5 = create_proj3_qwen3b(in_channels=37, use_t5=False, use_scale=False, use_cnn=True).to(dtype=weight_dtype)
    if args.qwen_size == "7b":
        proj_t5 = create_proj3_qwen7b(in_channels=29, use_t5=False, use_scale=False, use_cnn=True).to(dtype=weight_dtype)

    # Get the most recent checkpoint
    last_checkpoint_step = get_max_numbered_filename(args.output_dir)
    if last_checkpoint_step is not None:
        proj_t5_checkpoint_path = os.path.join(args.output_dir, str(last_checkpoint_step), "diffusion_pytorch_model.bin")
        state_dict = torch.load(proj_t5_checkpoint_path, map_location="cpu")
        proj_t5.load_state_dict(state_dict)
        print(f"xxxxxx load last_checkpoint_step: {last_checkpoint_step}, proj_t5_checkpoint_path: {proj_t5_checkpoint_path}")
    proj_t5.to(dtype=weight_dtype)

    tokenizer_clip = CLIPTokenizer.from_pretrained(teacher, subfolder="tokenizer", revision="refs/pr/1")
    tokenizer_t5 = T5TokenizerFast.from_pretrained(teacher, subfolder="tokenizer_2", revision="refs/pr/1")

    
    torch.cuda.synchronize()
    transformer_student = FluxTransformer2DModel.from_pretrained(teacher, subfolder="transformer")
    torch.cuda.synchronize()


    transformer_student.eval()
    transformer_student.requires_grad_(False)
    proj_t5.train()

    torch.cuda.synchronize()


    
    transformer_student.to(train_device, dtype=weight_dtype)
    proj_t5.to(train_device)

    
    KD_student= []
    cast_hook_list(transformer_student,KD_student)

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    total = sum(p.numel() for p in proj_t5.parameters())
    print("**********paras***********",total) 

    # Optimizer creation
    optimizer = optimizer_class(
        proj_t5.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    print(f"rank: {rank}, train finish init optimizer")

    preprocess = InferPreprocess(infer_pg, is_infer_rank, args.batch_size, infer_rank)

    datamodule = DataModuleCustom(args, tokenizer_t5=tokenizer_mllm, tokenizer_t5_en=tokenizer_t5, tokenizer_en=tokenizer_clip)
    train_dataloader = datamodule._train_dataloader(preprocess=preprocess, process_group=train_pg, group_rank=group_rank)


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = 10e10
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    print(f"train finish init lr_scheduler")
    proj_t5 = DistributedDataParallel(proj_t5, process_group=train_pg, find_unused_parameters=True)
    print(f"train finish accelerator.prepare")
    
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)



    ############## Train!
    total_batch_size = args.batch_size * dist.get_world_size(train_pg) * args.gradient_accumulation_steps
    if is_main_process:
        print("***** Running training *****")
        print(f"  Num Epochs = {args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {args.batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) =   {total_batch_size}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint is not None:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    # last checkpoint step
    if last_checkpoint_step is not None:
        global_step = last_checkpoint_step
        initial_global_step = last_checkpoint_step

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not is_local_main_process,
    )

    bsz_org = args.batch_size
    height = 128
    width = 128
    seq_len = 512
    seq_len_1 = 77
    text_ids = torch.zeros(seq_len, 3).to(device=train_device, dtype=weight_dtype)
    latent_image_ids = _prepare_latent_image_ids(bsz_org, height, width, train_device, weight_dtype)
    guidance = torch.tensor([3.5], device=train_device, dtype=weight_dtype)
    guidance = guidance.expand(bsz_org)

    for epoch in range(first_epoch, args.num_train_epochs):

        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            sync_gradients = step % args.gradient_accumulation_steps == 0

            # pixel_values = batch["pixel_values"].to(device=train_device,dtype=weight_dtype)
            input_ids_t5_en = batch["input_ids_t5_en"].to(device=train_device)
            input_ids_en = batch["input_ids_en"].to(device=train_device)
            input_ids_t5 = batch["input_ids_t5"].to(device=train_device)

            KD_teacher_tensor0 = batch["KD_teacher_tensor0"].to(device=train_device)
            KD_teacher_tensor1 = batch["KD_teacher_tensor1"].to(device=train_device)
            KD_teacher_tensor2 = batch["KD_teacher_tensor2"].to(device=train_device)

            latents = batch["latents"].to(device=train_device)
            text_embeddings = batch["text_embeddings"].to(device=train_device)
            timestep = batch["timestep"].to(device=train_device)

            add_text_embeds, prompt_embeds_zh = proj_t5(text_embeddings)

            noise_pred_student = transformer_student(
                hidden_states=latents.to(train_device),
                timestep=timestep / 1000,
                txt_ids=text_ids.to(train_device),
                guidance=guidance,
                img_ids=latent_image_ids.to(train_device),
                encoder_hidden_states=prompt_embeds_zh.to(device=train_device, dtype=transformer_student.dtype),  # b*512*4096
                pooled_projections=add_text_embeds.to(device=train_device, dtype=transformer_student.dtype), # b*768
                return_dict=False,
            )[0]


            KD_student_tensor0 = torch.stack(KD_student[0], dim=1).to(train_device)
            KD_student_tensor1 = torch.stack(KD_student[1], dim=1).to(train_device)
            KD_student_tensor2 = torch.stack(KD_student[2], dim=1).to(train_device)
            KD_student[0].clear()
            KD_student[1].clear()
            KD_student[2].clear()


            # loss  = F.mse_loss(KD_teacher_tensor0, KD_student_tensor0, reduction="none").mean([0, 2, 3]).sum()
            # loss += F.mse_loss(KD_teacher_tensor1, KD_student_tensor1, reduction="none").mean([0, 2, 3]).sum()
            # loss += F.mse_loss(KD_teacher_tensor2, KD_student_tensor2, reduction="none").mean([0, 2, 3]).sum()
            loss = 0
            temperature0 = 3
            for i in range(19):
                down_feature = F.kl_div(F.softmax(normalize(KD_teacher_tensor0[:,i])/temperature0, dim=-1).log(), F.softmax(normalize(KD_student_tensor0[:,i])/temperature0, dim=-1), reduction='batchmean')
                # down_feature = F.kl_div(F.softmax(KD_student_tensor0[:,i]/temperature0, dim=-1).log(), F.softmax(KD_teacher_tensor0[:,i]/temperature0, dim=-1), reduction='batchmean')
                if not (torch.isinf(down_feature).any() or torch.isnan(down_feature).any()):
                    loss=loss+down_feature
                else:
                    print(f"down_feature:{i}")
                down_feature1 = F.kl_div(F.softmax(normalize(KD_teacher_tensor1[:,i])/temperature0, dim=-1).log(), F.softmax(normalize(KD_student_tensor1[:,i])/temperature0, dim=-1), reduction='batchmean')
                if not (torch.isinf(down_feature1).any() or torch.isnan(down_feature1).any()):
                    loss=loss+down_feature1
                else:
                    print(f"down_feature1:{i}")
            for i in range(38):
                down_feature2 = F.kl_div(F.softmax(normalize(KD_teacher_tensor2[:,i])/temperature0, dim=-1).log(), F.softmax(normalize(KD_student_tensor2[:,i])/temperature0, dim=-1), reduction='batchmean')
                if not (torch.isinf(down_feature2).any() or torch.isnan(down_feature2).any()):
                    loss=loss+down_feature2
                else:
                    print(f"down_feature2:{i}")         

            train_loss = loss

            # Backpropagate
            loss.backward()
            if sync_gradients:
                params_to_clip = proj_t5.parameters()
                torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            if sync_gradients:

                progress_bar.update(1)
                global_step += 1
                train_loss = 0.0

                if is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        print(f"#########{args.checkpointing_steps} saving model #######")
                        save_path = os.path.join(args.output_dir, f"{global_step}")
                        state_dict = proj_t5.module.state_dict()
                        os.makedirs(save_path, exist_ok=True)
                        torch.save(state_dict, os.path.join(save_path, "diffusion_pytorch_model.bin"))


            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        


def infer(args, infer_pg, is_infer_rank, infer_rank,teacher,paths):
    weight_dtype = torch.bfloat16
    infer_device = torch.device("cuda")

    
    text_encoder_t5 = Qwen2_5_VLForConditionalGeneration.from_pretrained(paths, torch_dtype=torch.bfloat16)

    text_encoder = CLIPTextModel.from_pretrained(teacher, revision="refs/pr/1",subfolder="text_encoder", torch_dtype=torch.bfloat16)
    text_encoder_2 = T5EncoderModel.from_pretrained(teacher, revision="refs/p1", subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
    transformer_teacher = FluxTransformer2DModel.from_pretrained(teacher, subfolder="transformer")
    tokenizer_1 = AutoTokenizer.from_pretrained(paths, trust_remote_code=True)


    text_encoder_t5.eval()
    text_encoder_t5.requires_grad_(False)

    transformer_teacher.train()
    transformer_teacher.requires_grad_(False)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    text_encoder_2.eval()
    text_encoder_2.requires_grad_(False)

    text_encoder.to(infer_device, dtype=weight_dtype)
    text_encoder_2.to(infer_device, dtype=weight_dtype)
    text_encoder_t5.to(infer_device, dtype=weight_dtype)

    transformer_teacher.to(infer_device, dtype=weight_dtype)


    KD_teacher = []
    cast_hook_list(transformer_teacher, KD_teacher)

    height = 128
    width = 128
    seq_len = 512
    seq_len_1 = 77
    bsz_org = args.batch_size
    bsz = (dist.get_world_size(infer_pg) - 1) * bsz_org

    print(f"infer bsz_org: {bsz_org}, bsz: {bsz}")

    text_ids = torch.zeros(seq_len, 3).to(device=infer_device, dtype=weight_dtype)
    latent_image_ids = _prepare_latent_image_ids(bsz, height, width, infer_device, weight_dtype)

    input_ids_t5_en_ = torch.empty(size=(bsz_org, seq_len), dtype=torch.long, device=infer_device)
    input_ids_en_ = torch.empty(size=(bsz_org, seq_len_1), dtype=torch.long, device=infer_device)
    input_ids_t5_ = torch.empty(size=(bsz_org, seq_len), dtype=torch.long, device=infer_device)
    attention_mask_ = torch.empty(size=(bsz_org, seq_len), dtype=torch.long, device=infer_device)
    # pixel_values_ = torch.empty(size=(bsz_org, 1296, 1176), dtype=torch.float32, device=infer_device)
    # image_grid_thw_ = torch.empty(size=(bsz_org, 3), dtype=torch.long, device=infer_device)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(teacher,  subfolder="scheduler")
    guidance = torch.tensor([3.5], device=infer_device, dtype=weight_dtype)
    guidance = guidance.expand(bsz_org)
    print(f"infer finish load model, will get data by nccl")

    step = 0

    while True:
        with torch.no_grad():
            try:
                torch.cuda.synchronize()
             
                input_ids_t5_en = tai.send_to_infer_device(input_ids_t5_en_, infer_pg, is_infer_rank, infer_rank)
                torch.cuda.synchronize()
             
                input_ids_en = tai.send_to_infer_device(input_ids_en_, infer_pg, is_infer_rank, infer_rank)
                torch.cuda.synchronize()
        
                input_ids_t5 = tai.send_to_infer_device(input_ids_t5_, infer_pg, is_infer_rank, infer_rank)
                torch.cuda.synchronize()
              
                attention_mask = tai.send_to_infer_device(attention_mask_, infer_pg, is_infer_rank, infer_rank)
                torch.cuda.synchronize()


            except Exception as e:
                print(f"****** Error: infer_pg: {dist.get_process_group_ranks(infer_pg)}, is_infer_rank: {is_infer_rank}, infer_rank: {infer_rank}")
                time.sleep(30)
                raise e


            prompts = [input_ids_t5_en, input_ids_en]
            batch_size = len(input_ids_t5)
            height = 128
            width = 128
            num_channels_latents = transformer_teacher.config.in_channels // 4
            shape = (batch_size, num_channels_latents, height, width)
            generator = torch.Generator(device=infer_device).manual_seed(step)
            latents = randn_tensor(shape, generator=generator, device=infer_device, dtype=weight_dtype) # b*16*128*128
            latents = _pack_latents(latents, batch_size, num_channels_latents, height, width)
            latent_image_ids = _prepare_latent_image_ids(batch_size, height, width, infer_device, weight_dtype)

   
            num_inference_steps = 1
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            image_seq_len = latents.shape[1]
 
            mu = calculate_shift(
                image_seq_len,
                scheduler.config.base_image_seq_len,
                scheduler.config.max_image_seq_len,
                scheduler.config.base_shift,
                scheduler.config.max_shift,
            )
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler,
                num_inference_steps,
                infer_device,
                None,
                sigmas,
                mu=mu,
            )

            inputs = {"input_ids":input_ids_t5.to(infer_device), "attention_mask":attention_mask.to(infer_device)} 
            generated_ids = text_encoder_t5.generate(**inputs, max_new_tokens=1, output_hidden_states=True, return_dict_in_generate=True)
            text_embeddings = torch.stack(generated_ids["hidden_states"][0],dim=1)
            
            # print(f"text_embeddings: {text_embeddings.shape}")
            prompt_embeds_en = text_encoder_2(prompts[0].to(infer_device), output_hidden_states=False)[0]
            add_text_embeds_en = text_encoder(prompts[1].to(infer_device), output_hidden_states=False).pooler_output
            text_ids = torch.zeros(prompt_embeds_en.shape[1], 3).to(device=infer_device, dtype=weight_dtype)
            timestep = timesteps[0].expand(batch_size).to(device=infer_device, dtype=weight_dtype)

            noise_pred_teacher = transformer_teacher(
                hidden_states=latents,
                timestep=timestep / 1000,
                txt_ids=text_ids,
                guidance=guidance,
                img_ids=latent_image_ids,
                encoder_hidden_states=prompt_embeds_en, 
                pooled_projections=add_text_embeds_en, # 
                return_dict=False,
            )[0]


            KD_teacher_tensor0 = torch.stack(KD_teacher[0], dim=1).to(infer_device)
            KD_teacher_tensor1 = torch.stack(KD_teacher[1], dim=1).to(infer_device)
            KD_teacher_tensor2 = torch.stack(KD_teacher[2], dim=1).to(infer_device)

            KD_teacher[0].clear()
            KD_teacher[1].clear()
            KD_teacher[2].clear()

            torch.cuda.synchronize()
            tai.receive_from_infer_device(KD_teacher_tensor0, infer_pg, is_infer_rank, infer_rank)
            torch.cuda.synchronize()
            tai.receive_from_infer_device(KD_teacher_tensor1, infer_pg, is_infer_rank, infer_rank)
            torch.cuda.synchronize()
            tai.receive_from_infer_device(KD_teacher_tensor2, infer_pg, is_infer_rank, infer_rank)
            torch.cuda.synchronize()
            tai.receive_from_infer_device(latents, infer_pg, is_infer_rank, infer_rank)
            torch.cuda.synchronize()
            tai.receive_from_infer_device(text_embeddings, infer_pg, is_infer_rank, infer_rank)
            torch.cuda.synchronize()
            tai.receive_from_infer_device(timestep, infer_pg, is_infer_rank, infer_rank)
            torch.cuda.synchronize()
            step += 1




def seed_torch(seed=2024):
    random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)    
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)   
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False   
    torch.backends.cudnn.deterministic = True  

def main(args):

    # local_rank = int(os.environ["LOCAL_RANK"])
    # rank = int(os.environ["RANK"])
    

    local_infer_world_size = 2
    local_rank, local_world_size, rank, world_size, group_rank, group_world_size = tai.dist_info1()
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    seed_torch(seed=rank*2024)

   
    infer_pg, infer_rank, is_infer_rank, infer_pg_world_size, infer_pgs = tai.new_infer_pg(rank, local_infer_world_size, local_world_size, world_size)
    train_pg, _ = tai.new_train_pg(rank, local_infer_world_size, local_world_size, world_size)
    _ZERO_PARAM_INTRA_PARALLEL_GROUP = train_pg
    _DATA_PARALLEL_GROUP = train_pg

    teacher = "/mnt/data/group/models/flux/FLUX.1-dev"
    if args.qwen_size == "3b":
        paths = "/mnt/data/group/models/Qwen2.5-VL-3B-Instruct"
    if args.qwen_size == "7b":
        paths = "/mnt/data/group/models/Qwen2.5-VL-7B-Instruct"

    if is_infer_rank:

        infer(args, infer_pg, is_infer_rank, infer_rank,teacher,paths)
    else:
        train(args, infer_pg, is_infer_rank, train_pg, group_rank, infer_rank,teacher,paths)


if __name__ == "__main__":
    args = parse_args()
    main(args)