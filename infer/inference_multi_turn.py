import sys
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import T5EncoderModel, T5TokenizerFast, CLIPTokenizer, CLIPTextModel, T5Config
from transformers.models.t5.modeling_t5 import T5Stack

from diffusers import FluxPipeline, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import torch 
import torch.nn as nn
import time
import os

from utils.proj import create_proj3_qwen3b, create_proj3_qwen7b
from qwen_vl_utils import process_vision_info
from PIL import Image
import shutil

import numpy as np
import argparse
import json

dtype = torch.bfloat16
device = "cuda:0"
parser = argparse.ArgumentParser("Infer Pipeline", add_help=True)
parser.add_argument('--num_processors', default=1, type=int)
parser.add_argument('--process_id', default=0, type=int)
args = parser.parse_args()

ckpt_id = "/mnt/data/group/models/flux/shuttle-3-diffusion"



def load_qwen2_5():
    path = "/mnt/data/group/models/Qwen2.5-VL-7B-Instruct"
    text_encoder_t5 = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16).eval().to(device=device)
    tokenizer_t5 = AutoProcessor.from_pretrained(path)
    return text_encoder_t5, tokenizer_t5

pipeline = FluxPipeline.from_pretrained(
    ckpt_id, text_encoder=None, text_encoder_2=None,
    tokenizer=None, tokenizer_2=None, 
    revision="refs/pr/1",
    torch_dtype=torch.bfloat16
).to(device)


vae = AutoencoderKL.from_pretrained(
    ckpt_id, 
    revision="refs/pr/1",
    subfolder="vae",
    torch_dtype=torch.bfloat16
).to(device)

    
load_fn = load_qwen2_5
text_encoder_t5, tokenizer_t5 = load_fn()


file_caption = "prompts/2025_all.txt"
raw_texts = [line.strip() for line in open(file_caption).readlines()][:50]

proj_t5_save_path = "/mnt/data/group/majian/flux/result_fit_speed/qwenvl25_dev_norm/57000/diffusion_pytorch_model.bin"

proj_t5 = create_proj3_qwen7b(in_channels=29, use_t5=False, use_scale=False, use_cnn=True).to(device=device,dtype=dtype)

state_dict = torch.load(proj_t5_save_path, map_location="cpu")
state_dict_new = {}
for k,v in state_dict.items():
    k_new = k.replace("module.","")
    state_dict_new[k_new] = v

proj_t5.load_state_dict(state_dict_new)
proj_t5.eval()

outputs = f"/mnt/data/group/majian/flux/outputs_new/answer_id"
os.makedirs(outputs, exist_ok=True)


message=[]
while True:
    raw_text = input("\nPlease Input Query (stop to exit) >>> ")
    if not raw_text:
        print('Query should not be empty!')
        continue
    if raw_text == "stop":
        break

    filepath = "/mnt/data/group/majian/flux/images/000002040.1.0.jpg"
    t1 = time.time()
    Instructions = {"Text input":raw_text,"Instruction editing description":"","image input":"yes"}

    image_inputs0 = Image.open(filepath).convert('RGB').resize(size=(256, 256))
    message.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image_inputs0},
                {"type": "text", "text": str(Instructions)}
            ]
        })
    with torch.no_grad():     
        print(message)
        text = tokenizer_t5.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(message)

        inputs = tokenizer_t5(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            # padding="max_length",
            # max_length=1024, 
            # padding=True,
            # truncation=True, 
            return_tensors="pt"
        ).to(device)


        ## 新版
        generated_ids = text_encoder_t5.generate(**inputs, max_new_tokens=64,output_hidden_states=True,return_dict_in_generate=True)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids[0])]
        answer = tokenizer_t5.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(answer)

        new_turn = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer}
            ]
        }
        message.append(new_turn)
        

        text_embeddings = []
        for hidden_states in  generated_ids["hidden_states"][1:]:
            text_embeddings.append(torch.cat(hidden_states))

        ## answer 答案信息
        text_embeddings_out = torch.cat(text_embeddings,dim=1).unsqueeze(0)
        ## input(i+t) 输入信息
        text_embeddings_in = torch.cat(generated_ids["hidden_states"][0]).unsqueeze(0)
        ##all
        text_embeddings_all = torch.cat([text_embeddings_in,text_embeddings_out],dim=2)
        print(f"text_embeddings_in.shape:{text_embeddings_in.shape},text_embeddings_out.shape:{text_embeddings_out.shape}")

        pooled_prompt_embeds,prompt_embeds = proj_t5(text_embeddings_all)
        t2 = time.time()
        height, width = 1024, 1024
        # No need to wrap it up under `torch.no_grad()` as pipeline call method
        # is already wrapped under that.
        latents = pipeline(
            prompt_embeds=prompt_embeds, 
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=4, guidance_scale=3.5, 
            height=height, width=width,
            output_type="latent",
            generator=torch.Generator(device).manual_seed(0)
        ).images


        vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

        latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        image = vae.decode(latents, return_dict=False)[0]
        image = image_processor.postprocess(image, output_type="pil")
        filename = 1
        image[0].save(f"{outputs}/{filename}.jpg")