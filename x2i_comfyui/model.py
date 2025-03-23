import torch
from torch import nn
from safetensors.torch import save_model, load_model
from transformers.models.t5.modeling_t5 import T5Stack
from transformers import PreTrainedModel, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info



class MLP(nn.Module):
    def __init__(self, in_dim=4096, out_dim=4096, hidden_dim=4096, out_dim1=768, layer_norm_eps=1e-5, use_residual=True):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_dim, eps=layer_norm_eps)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.fc = nn.Sequential(
            nn.GELU(),
            nn.Linear(out_dim, out_dim1)
        )

    def forward(self, x):
        x = self.layernorm(x)
        x2 = self.projector(x)
        x1 = self.fc(x2)
        x1 = torch.mean(x1,1)
        return x1,x2

class Proj(nn.Module):

    @staticmethod
    def load(path):
        all_dict = torch.load(path, map_location="cpu", weights_only=True)
        proj = Proj(**all_dict["config"])
        proj.load_state_dict(all_dict["state_dict"])
        proj.eval()
        return proj
    
    @staticmethod
    def config(model_type):
        if model_type == "0_5b":
            return {
                "in_channels": 25,
                "kernel_size": 5,
                "input_dim": 896,
                "output_dim0": 768,
                "output_dim1": 4096,
                "num_layers": 2,
                "num_heads": 12,
                "norm_eps": 1e-6,
                "head_dim": 64,
                "use_t5": False,
                "use_scale": False,
                "use_cnn": True,
            }
        elif model_type == "3b":
            return {
                "in_channels": 37,
                "kernel_size": 5,
                "input_dim": 2048,
                "output_dim0": 768,
                "output_dim1": 4096,
                "num_layers": 2,
                "num_heads": 16,
                "norm_eps": 1e-6,
                "head_dim": 128,
                "use_t5": False,
                "use_scale": False,
                "use_cnn": True,
            }
        else:
            return {
                "in_channels": 29,
                "kernel_size": 5,
                "input_dim": 3584,
                "output_dim0": 768,
                "output_dim1": 4096,
                "num_layers": 2,
                "num_heads": 28,
                "norm_eps": 1e-6,
                "head_dim": 128,
                "use_t5": False,
                "use_scale": False,
                "use_cnn": True,
            }

    @staticmethod
    def transfrom(config, state_path, save_path):
        
        state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
        all_dict = {
            "config": config,
            "state_dict": state_dict,
        }
        torch.save(all_dict, save_path)



    def __init__(self, in_channels=25, kernel_size=5, input_dim=896, output_dim0=768, output_dim1=4096, num_layers=2, num_heads=12, norm_eps=1e-6, head_dim=64, use_t5=True, use_scale=True, use_cnn=True) -> None:
        super().__init__()
        self.use_t5 = use_t5
        self.use_scale = use_scale
        self.use_cnn = use_cnn
        if self.use_t5:
            config = T5Config(num_heads=num_heads, num_layers=num_layers, num_decoder_layers=0, layer_norm_epsilon=norm_eps, is_encoder_decoder=False, 
                is_decoder=False, d_ff=input_dim*4, d_kv=head_dim, d_model=input_dim, dense_act_fn="gelu_new", feed_forward_proj="gated-gelu", use_cache=False)
            # print(f"config: {config}")
            self.t5stack = T5Stack(config)
        if self.use_scale:
            self.cha_scale = nn.Parameter(torch.empty(1, in_channels, 1, 1), requires_grad=True)
        elif self.use_cnn:
            self.conv = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.mlp = MLP(input_dim, output_dim1, output_dim1, output_dim0, norm_eps)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        if self.use_scale:
            nn.init.xavier_normal_(self.cha_scale, gain=1)

    def enable_gradient_checkpointing(self):
        if self.use_t5:
            self.t5stack.gradient_checkpointing_enable()

    def forward(self, x):
        B, C, S, H = x.shape
        if self.use_t5:
            x = self.t5stack(inputs_embeds=x.contiguous().view(B * C, S, H)).last_hidden_state
        if self.use_scale:
            x = (self.cha_scale * x.view(B, C, S, H)).mean(dim=1)
        elif self.use_cnn:
            x = self.conv(x.view(B, C, S, H)).squeeze(1)
        else:
            x = x.view(B, C, S, H).mean(dim=1)
        return self.mlp(x)


# class MLLM():

#     def encode(self, proj, images, videos, audios, text):
#         pass

#     def forward(self, proj, images, videos, audios, text):
#         return self.encode(images, videos, audios, text, proj)


class QwenVL2_5():

    processor = None
    qwenvl = None
    device = None

    @staticmethod
    def load(path, device="cpu"):
        all_dict = torch.load(path, weights_only=False)
        qwenvl = QwenVL2_5()
        qwenvl.processor = all_dict["processor"]
        model = Qwen2_5_VLForConditionalGeneration(all_dict["config"])
        model.load_state_dict(all_dict["state_dict"])
        model.to(dtype=torch.bfloat16, device=device).eval()
        qwenvl.qwenvl = model
        qwenvl.device = device
        return qwenvl

    @staticmethod
    def transform(path="/home/notebook/data/group/model_hub/qwen2.5-vl/Qwen/Qwen2___5-VL-3B-Instruct", save_path=None):
        # qwenvl = QwenVL2_5()
        processor = Qwen2_5_VLProcessor.from_pretrained(path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16).eval()
        config = model.config
        state_dict = model.state_dict()
        all_dict = {
            "config": config,
            "state_dict": state_dict,
            "processor": processor,
        }
        torch.save(all_dict, save_path)

    # @staticmethod
    @torch.no_grad()
    def encode(self, proj, images, videos, audios, text):
        # print(f"encode images: {images}")
        if isinstance(images, str):
            images = [images]
        
        if isinstance(videos, str):
            videos = [videos]
           
        instructions = {
            "Text input":"" if text is None else text,
            "Instruction editing description":"",
            "image input":"no" if videos is None and images is None else "yes",
        }
        message = [{"role": "user", "content": []}]
        need_chat_template = ((text is None) or ("<|image_pad|>" not in text) or ("<|video_pad|>" not in text))
        need_process_vision = False
        if images is not None and len(images) > 0:
            need_process_vision = True
            for image in images:
                message[0]["content"].append({"type": "image", "image": image, "max_pixels": 224 * 224})
        if videos is not None and len(videos) > 0:
            need_process_vision = True
            for video in videos:
                message[0]["content"].append({"type": "video", "video": video, "max_pixels": 128 * 128, "fps": 1.0})
        if images is not None or videos is not None:
            image_inputs, video_inputs = process_vision_info(message)
        text = None
        if need_chat_template:
            message[0]["content"].append({"type": "text", "text": str(instructions)})
            text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True, add_vision_id=True)
        else:
            text = str(instructions)
        inputs = self.processor(
            text=[text],
            images=None if images is None else image_inputs,
            videos=None if videos is None else video_inputs,
            # padding="max_length",
            # max_length=512,  
            # truncation=True,
            return_tensors="pt")
        input_ids_len = inputs.input_ids.shape[1]
        max_len = max(int(input_ids_len * 1.3), 512)
        print(f"get_inputs input_ids len: {input_ids_len}, max_len: {max_len}")
        inputs = self.processor(
            text=[text],
            images=None if images is None else image_inputs,
            videos=None if videos is None else video_inputs,
            padding="max_length",
            max_length=max_len,  
            # truncation=True,
            return_tensors="pt").to(self.device)

        
        # print(f"type(mllm): {type(mllm)}, type(qwenvl): {type(mllm.qwenvl)}")
        output = self.qwenvl.generate(**inputs, max_new_tokens=1, output_hidden_states=True, return_dict_in_generate=True)
        hidden_states = output["hidden_states"]
        text_embeddings_in = torch.cat(hidden_states[0]).unsqueeze(0)
        text_embeddings = []
        for hidden_state in hidden_states[1:]:
            text_embeddings.append(torch.cat(hidden_state))
        if len(text_embeddings) > 0:
            text_embeddings_out = torch.cat(text_embeddings, dim=1).unsqueeze(0)
            text_embeddings_all = torch.cat([text_embeddings_in, text_embeddings_out], dim=2)
        pooled_prompt_embeds, prompt_embeds = proj(text_embeddings_in)
        return [[prompt_embeds, {'pooled_output': pooled_prompt_embeds}]]



if __name__ == "__main__":
    # qwenvl = QwenVL2_5.from_pretrained()
    # qwenvl.save("./qwenvl.pt")
    # qwenvl = QwenVL2_5.load("./qwenvl.pt")
    # proj = Proj.from_pretrained()
    # proj.save("./proj.pt")
    QwenVL2_5.transform("/mnt/data/group/models/Qwen2.5-VL-3B-Instruct", "/mnt/data/group/pqr/AndesDiT/x2i_comfyui_models/qwen2.5-vl-3b-instruct.pt")
    qwenvl = QwenVL2_5.load("/mnt/data/group/pqr/AndesDiT/x2i_comfyui_models/qwen2.5-vl-3b-instruct.pt")
    QwenVL2_5.transform("/mnt/data/group/models/Qwen2.5-VL-7B-Instruct", "/mnt/data/group/pqr/AndesDiT/x2i_comfyui_models/qwen2.5-vl-7b-instruct.pt")
    qwenvl = QwenVL2_5.load("/mnt/data/group/pqr/AndesDiT/x2i_comfyui_models/qwen2.5-vl-7b-instruct.pt")
    
    
    config = Proj.config("3b")
    Proj.transfrom(config, "/mnt/data/group/majian/flux/result_fit_speed/3b/13000/diffusion_pytorch_model.bin", "/mnt/data/group/pqr/AndesDiT/x2i_comfyui_models/qwen2.5-vl-3b_proj.pt")
    proj = Proj.load("/mnt/data/group/pqr/AndesDiT/x2i_comfyui_models/qwen2.5-vl-3b_proj.pt")

    config = Proj.config("7b")
    Proj.transfrom(config, "/mnt/data/group/majian/flux/result_fit_speed/qwenvl25_dev_norm/57000/diffusion_pytorch_model.bin", "/mnt/data/group/pqr/AndesDiT/x2i_comfyui_models/qwen2.5-vl-7b_proj.pt")
    proj = Proj.load("/mnt/data/group/pqr/AndesDiT/x2i_comfyui_models/qwen2.5-vl-7b_proj.pt")

