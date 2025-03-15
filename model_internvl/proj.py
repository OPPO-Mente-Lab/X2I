import os
import json
import torch
import torch.nn as nn
from einops import rearrange
import inspect
import argparse
from pytorch_lightning import (
    LightningModule,
    Trainer,
)
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)
from torch.nn import functional as F
from tqdm.auto import tqdm
from typing import Callable, List, Optional, Union
import deepspeed



from torchvision.utils import save_image
from diffusers.image_processor import VaeImageProcessor

from transformers import T5Tokenizer, T5ForConditionalGeneration,T5EncoderModel,MT5EncoderModel,AutoTokenizer,AutoModel,AutoModelForCausalLM,AutoTokenizer
from PIL import Image

from diffusers import StableDiffusion3Pipeline
from typing import Any, Callable, Dict, List, Optional, Union
import inspect
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
from diffusers.models.attention import JointTransformerBlock

from transformers import BertModel, BertTokenizer
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from transformers import T5EncoderModel, T5TokenizerFast, CLIPTokenizer, CLIPTextModel, T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from diffusers import FluxPipeline, AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
import gc

import numpy as np


Image.MAX_IMAGE_PIXELS = None
        
class MLP(nn.Module):
    def __init__(self, in_dim=4096, out_dim=4096, hidden_dim=4096, out_dim1=768, layer_norm_eps=1e-5, use_residual=True):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_dim, eps=layer_norm_eps)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        self.fc = nn.Linear(out_dim, out_dim1)

    def forward(self, x):
        # print(f"****** x.device = {x.device}, rank = {torch.distributed.get_rank()}")
        x = self.layernorm(x)
        x = self.projector(x)
        x2 = nn.GELU()(x)
        x1 = self.fc(x2)
        x1 = torch.mean(x1,1)
        return x1,x2


class MLP2(nn.Module):
    def __init__(self, in_dim=4096, out_dim=4096, hidden_dim=4096, out_dim1=768, layer_norm_eps=1e-5, use_residual=True):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_dim, eps=layer_norm_eps)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        self.fc = nn.Sequential(
            nn.Linear(out_dim, out_dim1, bias=False),
            nn.GELU(),
            nn.Linear(out_dim1, out_dim1, bias=False),
            nn.GELU(),
            nn.Linear(out_dim1, out_dim1, bias=False)
        )

    def forward(self, x):
        # print(f"****** x.device = {x.device}, rank = {torch.distributed.get_rank()}")
        x = self.layernorm(x)
        x = self.projector(x)
        x2 = nn.GELU()(x)
        x1 = self.fc(x2)
        x1 = torch.mean(x1,1)
        return x1,x2

class MLP_plus(nn.Module):
    def __init__(self, in_dim=4096, out_dim=4096, hidden_dim=4096, out_dim1=768, use_residual=True):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_dim)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        self.fc = nn.Linear(out_dim, out_dim1)

    def forward(self, x):
        # print(f"****** x.device = {x.device}, rank = {torch.distributed.get_rank()}")
        x = self.layernorm(x)
        x = self.projector(x)
        x2 = nn.GELU()(x)
        x1 = self.fc(x2)
        x1 = torch.mean(x1,1)
        return x1,x2


class Transformer_proj(nn.Module):
    def __init__(self, d_model,  n_heads, out_dim1, out_dim2,num_layers=3) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=2048, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear1 = nn.Linear(d_model, out_dim1)
        self.linear2 = nn.Linear(d_model, out_dim2)
    
    ## B*77*4096 --> B*1280   B*77*2048  
    def forward(self, x):
        x = self.transformer_encoder(x)
        x1 = self.linear1(x)
        x1 = torch.mean(x1,1)
        x2 = self.linear2(x)
        return x1,x2

class Proj(nn.Module):
    def __init__(self, in_channels=2, kernel_size=5, input_dim=896, output_dim0=768, output_dim1=4096, num_layers=4, num_heads=12, layer_norm_eps=1e-6, head_dim=64) -> None:
        super().__init__()
        config = T5Config(num_heads=num_heads, num_layers=num_layers, num_decoder_layers=0, layer_norm_epsilon=layer_norm_eps, is_encoder_decoder=False, 
            is_decoder=False, d_ff=input_dim*4, d_kv=head_dim, d_model=input_dim, dense_act_fn="gelu_new", feed_forward_proj="gated-gelu", use_cache=False)
        print(f"config: {config}")
        self.norm0 = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.norm1 = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.t5stack = T5Stack(config)
        self.mlp = MLP(input_dim, output_dim1, output_dim1, output_dim0, layer_norm_eps)
        
    ## B*77*4096 --> B*1280   B*77*2048  
    def forward(self, x):
        x = self.norm0(x)
        x = self.conv(x).squeeze(1)
        x = self.norm1(x)
        x = self.t5stack(inputs_embeds=x).last_hidden_state
        return self.mlp(x)

class Proj2(nn.Module):
    def __init__(self, in_channels=2, kernel_size=5, input_dim=896, output_dim0=768, output_dim1=4096, num_layers=4, num_heads=12, layer_norm_eps=1e-6, head_dim=64) -> None:
        super().__init__()
        config = T5Config(num_heads=num_heads, num_layers=num_layers, num_decoder_layers=0, layer_norm_epsilon=layer_norm_eps, is_encoder_decoder=False, 
            is_decoder=False, d_ff=input_dim*4, d_kv=head_dim, d_model=input_dim, dense_act_fn="gelu_new", feed_forward_proj="gated-gelu", use_cache=False)
        print(f"config: {config}")
        self.norm0 = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.norm1 = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.t5stack = T5Stack(config)
        self.mlp = MLP2(input_dim, output_dim1, output_dim1, output_dim0, layer_norm_eps)
        
    ## B*77*4096 --> B*1280   B*77*2048  
    def forward(self, x):
        x = self.norm0(x)
        x = self.conv(x).squeeze(1)
        x = self.norm1(x)
        x = self.t5stack(inputs_embeds=x).last_hidden_state
        return self.mlp(x)


class Proj3(nn.Module):
    def __init__(self, in_channels=2, kernel_size=5, input_dim=896, output_dim0=768, output_dim1=4096, num_layers=4, num_heads=12, layer_norm_eps=1e-6, head_dim=64) -> None:
        super().__init__()
        config = T5Config(num_heads=num_heads, num_layers=num_layers, num_decoder_layers=0, layer_norm_epsilon=layer_norm_eps, is_encoder_decoder=False, 
            is_decoder=False, d_ff=input_dim*4, d_kv=head_dim, d_model=input_dim, dense_act_fn="gelu_new", feed_forward_proj="gated-gelu", use_cache=False)
        print(f"config: {config}")
        
        self.t5stack = T5Stack(config)
        self.norm0 = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.norm1 = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.mlp = MLP2(input_dim, output_dim1, output_dim1, output_dim0, layer_norm_eps)
        
    ## B*77*4096 --> B*1280   B*77*2048  
    def forward(self, x):
        B, C, S, H = x.shape
        x = self.t5stack(inputs_embeds=x.contiguous().view(B * C, S, H)).last_hidden_state
        x = self.norm0(x)
        x = self.conv(x.view(B, C, S, H)).squeeze(1)
        x = self.norm1(x)
        return self.mlp(x)
