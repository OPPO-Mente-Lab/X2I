import os
import torch
from torch import nn

import node_helpers
import folder_paths
import latent_preview
import model_management


from safetensors.torch import save_model, load_model
from transformers.models.t5.modeling_t5 import T5Stack
from transformers import PreTrainedModel, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from .model import Proj, QwenVL2_5


class LoadVideoPath:

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"video": (sorted(files), {"video_upload": True})},
                }

    CATEGORY = "x2i"

    RETURN_TYPES = ("VIDEO_PATH",)
    FUNCTION = "load_video_path"

    def load_video_path(self, video):
        video_path = folder_paths.get_annotated_filepath(video)
        return (video_path, )


class LoadImagePath:

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
               }

    CATEGORY = "x2i"

    RETURN_TYPES = ("IMAGE_PATH",)
    FUNCTION = "load_image_path"

    def load_image_path(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        return (image_path, )

class MultiImagePaths:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_path1": ("IMAGE_PATH",),
            },
            "optional": {
                "image_path2": ("IMAGE_PATH",),
                "image_path3": ("IMAGE_PATH",),
                "image_path4": ("IMAGE_PATH",),
            },
        }
    RETURN_TYPES = ("IMAGE_PATH",)
    FUNCTION = "pack_images"

    CATEGORY = "x2i"

    DESCRIPTION = ""

    def pack_images(self, image_path1, image_path2=None, image_path3=None, image_path4=None):
        image_paths = [image_path1, image_path2, image_path3, image_path4]
        image_paths = [image_path for image_path in image_paths if image_path is not None]
        return (image_paths,)


class MLLMLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "mllm_name": (folder_paths.get_filename_list("text_encoders"), ),
                              "type": (["qwenvl2.5", "internvl2.5", "minicpm-o"], ),
                              },
                "optional": {
                              "device": (["default", "cpu"], {"advanced": True}),
                             }}
    RETURN_TYPES = ("MLLM",)
    FUNCTION = "load_mllm"

    CATEGORY = "x2i"

    DESCRIPTION = "[Recipes]\n\nqwenvl2.5\ninternvl2.5\nsminicpm-o"

    def load_mllm(self, mllm_name, type="qwenvl2.5", device="default"):
        mllm_path = folder_paths.get_full_path_or_raise("text_encoders", mllm_name)
        mllm = None
        if device == "default":
            device = model_management.text_encoder_device()
        print(f"device: {device}")
        if type == "qwenvl2.5":
            mllm = QwenVL2_5.load(mllm_path, device=device)
        
        return (mllm,)


class MLLMEncode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mllm": ("MLLM",),
                "proj": ("PROJ",),
            },
            "optional": {
                "image_paths": ("IMAGE_PATH",),
                "video_paths": ("VIDEO_PATH",),
                "audio_paths": ("VIDEO_PATH",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "x2i"

    DESCRIPTION = "mllm encode"

    def encode(self, mllm, proj, image_paths=None, video_paths=None, audio_paths=None, text=None):
        # print(f"encode mllm: {mllm}")
        conditioning = mllm.encode(proj, image_paths, video_paths, audio_paths, text)
        return (conditioning,)


class ProjLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "proj_name": (folder_paths.get_filename_list("text_encoders"), ),},
                "optional": { "device": (["default", "cpu"], {"advanced": True}), }}
    RETURN_TYPES = ("PROJ",)
    FUNCTION = "load_proj"
    CATEGORY = "x2i"
    # CATEGORY = "advanced/loaders"

    DESCRIPTION = "[Recipes]\n\nqwenvl2.5\ninternvl2.5\nsminicpm-o"

    def load_proj(self, proj_name, device="default"):
        proj_path = folder_paths.get_full_path_or_raise("text_encoders", proj_name)
        proj = Proj.load(proj_path)
        if device == "default":
            device = model_management.text_encoder_device()
        print(f"device: {device}")
        proj.to(dtype=torch.bfloat16, device=device).eval()
        return (proj,)


NODE_CLASS_MAPPINGS = {
    "MLLMLoader": MLLMLoader,
    "MLLMEncode": MLLMEncode,
    "ProjLoader": ProjLoader,
    "MultiImagePaths": MultiImagePaths,
    "LoadImagePath": LoadImagePath,
    # "LoadVideoPath": LoadVideoPath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLLMLoader": "MLLMLoader",
    "MLLMEncode": "MLLMEncode",
    "ProjLoader": "ProjLoader",
    "MultiImagePaths": "MultiImagePaths",
    "LoadImagePath": "LoadImagePath",
    # "LoadVideoPath": "LoadVideoPath",
}