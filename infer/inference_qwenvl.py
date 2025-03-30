import torch
import torch.nn as nn
import time
import json
import os 

from diffusers import FluxPipeline, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor


from transformers import T5EncoderModel, T5TokenizerFast, CLIPTokenizer, CLIPTextModel, T5Config
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info
from utils.proj import create_proj3_qwen3b, create_proj3_qwen7b
from PIL import Image
import argparse
# from qwen_vl_utils import process_vision_info

import librosa
import soundfile as sf
from decord import VideoReader, cpu, gpu




parser = argparse.ArgumentParser("Inference", add_help=True)
# parser.add_argument('--minicpm_path', type=str, default="openbmb/MiniCPM-o-2_6")
parser.add_argument('--qwen_size', type=str, default='7b', choices=['3b', '7b'], help="Model size: 1b or 4b")
parser.add_argument('--qwen3b_path', type=str, default="/mnt/data/group/models/Qwen2.5-VL-3B-Instruct")
# parser.add_argument('--flux_path', type=str,  default="shuttleai/shuttle-3-diffusion")
parser.add_argument('--flux_path', type=str,  default="/mnt/data/group/models/flux/shuttle-3-diffusion")
parser.add_argument('--use_answer', type=bool,  default=False)
parser.add_argument('--num_steps', type=int, default=4)
parser.add_argument('--num_gen_imgs', type=int, default=1)
parser.add_argument('--task', type=str, default="all")
args = parser.parse_args()

device = "cuda:0"
dtype = torch.bfloat16

outputs = "./outputs_qwen3b"

if args.qwen_size == "7b":
    outputs = "./outputs_qwen7b"
    qwen_path = '/mnt/data/group/models/Qwen2.5-VL-7B-Instruct'
    qwen_proj_path = '/mnt/data/group/xuguo/X2I_qwen/checkpoints/qwen7b/diffusion_pytorch_model.bin'
if args.qwen_size == "3b":
    outputs = "./outputs_qwen3b"
    qwen_path = '/mnt/data/group/models/Qwen2.5-VL-3B-Instruct'
    qwen_proj_path = '/mnt/data/group/xuguo/X2I_qwen/checkpoints/qwen3b/diffusion_pytorch_model.bin'


num_steps = args.num_steps
flux_path = args.flux_path
use_answer = args.use_answer
num_gen_imgs = args.num_gen_imgs
task = args.task


torch.cuda.set_device(device)


qwen_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_path, torch_dtype=torch.bfloat16).eval().to(device=device)
qwen_processor = AutoProcessor.from_pretrained(qwen_path)

clip_tokenizer = CLIPTokenizer.from_pretrained(flux_path, revision="refs/pr/1", subfolder="tokenizer", torch_dtype=dtype)
t5_tokenizer = T5TokenizerFast.from_pretrained(flux_path, revision="refs/pr/1", subfolder="tokenizer_2", torch_dtype=dtype)
clip_model = CLIPTextModel.from_pretrained(flux_path, revision="refs/pr/1", subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
t5_model = T5EncoderModel.from_pretrained(flux_path, revision="refs/pr/1", subfolder="text_encoder_2", torch_dtype=dtype).to(device).eval()

pipeline = FluxPipeline.from_pretrained(flux_path, text_encoder=None, text_encoder_2=None,
    tokenizer=None, tokenizer_2=None, vae=None, revision="refs/pr/1", torch_dtype=dtype).to(device)

vae = AutoencoderKL.from_pretrained(flux_path, revision="refs/pr/1", subfolder="vae", torch_dtype=dtype).to(device)

def get_proj(proj_path):

    if args.qwen_size == "3b":
        proj = create_proj3_qwen3b(in_channels=37, use_t5=False, use_scale=False, use_cnn=True)
    if args.qwen_size == "7b":
        proj = create_proj3_qwen7b(in_channels=29, use_t5=False, use_scale=False, use_cnn=True)


    state_dict = torch.load(proj_path, map_location="cpu")
    state_dict_new = {}
    for k,v in state_dict.items():
        k_new = k.replace("module.","")
        state_dict_new[k_new] = v

    proj.load_state_dict(state_dict_new)
    proj.to(device=device, dtype=dtype)
    proj.eval()
    return proj

qwen_proj = get_proj(qwen_proj_path)
def get_t5_input_embeds(text_prompt=None):
    text_input_ids = clip_tokenizer(
        text_prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    ).input_ids
    pooled_prompt_embeds = clip_model(text_input_ids.to(device), output_hidden_states=False).pooler_output.to(dtype=torch.bfloat16, device=device)
    text_input_ids = t5_tokenizer(
        text_prompt,
        # padding="max_length",
        # max_length=512,
        # truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    ).input_ids
    print(f"get_t5_input_embeds input_ids: {text_input_ids.shape}")
    prompt_embeds = t5_model(text_input_ids.to(device), output_hidden_states=False)[0].to(dtype=torch.bfloat16, device=device)
    return pooled_prompt_embeds, prompt_embeds

def get_text_embeddings(output_hidden_state):
    if args.qwen_size == "3b":
        text_embeddings = torch.cat(output_hidden_state["hidden_states"][0]).unsqueeze(0)
    if args.qwen_size == "7b":
        if use_answer:
            text_embeddings = []
            for hidden_states in  output_hidden_state["hidden_states"][1:]:
                text_embeddings.append(torch.cat(hidden_states))
            text_embeddings = torch.cat(text_embeddings,dim=1).unsqueeze(0)
        else:
            text_embeddings = torch.cat(output_hidden_state["hidden_states"][0]).unsqueeze(0)
    return text_embeddings



def get_qwen_inputs_embeds(videos=None, images=None, text_prompt=None, proj=qwen_proj):
    message = [{"role": "user", "content": []}]
    image_list = []
    if images is not None and len(images) > 0:
        for image in images:
            image_input = Image.open(image).convert('RGB').resize(size=(128, 128))
            message[0]["content"].append({"type": "image", "image": image_input})
            image_list.append(image_input)
    
    if videos is not None and len(videos) > 0:
        assert len(videos) == 1
        video = videos[0]
        message[0]["content"].append({
                    "type": "video",
                    "video": video,
                    "max_pixels": 128 * 128,
                    "fps": 1.0,
                })
        _, video_inputs = process_vision_info(message)

    if videos is None:
        video_inputs = None
             

    if text_prompt is not None:
        message[0]["content"].append({"type": "text", "text": text_prompt})
    
    prompt = qwen_processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    inputs = qwen_processor(
    text=[prompt],
    images=None if len(image_list) == 0 else image_list,
    videos= video_inputs,
    padding="max_length",
    max_length=512, 
    truncation=True, 
    return_tensors="pt",
    
    ).to(device)

    output_hidden_state = qwen_encoder.generate(**inputs, max_new_tokens=128,output_hidden_states=True,return_dict_in_generate=True)

    text_embeddings = get_text_embeddings(output_hidden_state)
    pooled_prompt_embeds, prompt_embeds = proj(text_embeddings)
    return pooled_prompt_embeds, prompt_embeds


@torch.no_grad()
def generate(pooled_prompt_embeds, prompt_embeds, outputs, filename, seed=None, height=1024, width=1024):
    os.makedirs(outputs, exist_ok=True)

    if seed is not None:
        latents = pipeline(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=num_steps,
            guidance_scale=3.5,
            height=height,
            width=width,
            output_type="latent",
            generator=torch.Generator(device).manual_seed(seed)
        ).images
    else:
        latents = pipeline(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=num_steps,
            guidance_scale=3.5,
            height=height,
            width=width,
            output_type="latent",
        ).images

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    image = vae.decode(latents, return_dict=False)[0]

    image = image_processor.postprocess(image, output_type="pil")
    image[0].save(f"{outputs}/{filename}.jpg")


def text2image(outputs=outputs):
    outputs = os.path.join(outputs, "text2image")
    os.makedirs(outputs, exist_ok=True)

    prompts = [{
        "EN": "A majestic elephant stands gracefully in a sun-drenched savannah, its textured gray skin glistening under the warm golden light of the late afternoon sun. The elephant, with large, expressive ears and a gently curved trunk, is posed mid-stride, kicking up a cloud of dust as it moves towards a shimmering waterhole surrounded by lush green acacia trees. The scene is painted in a vibrant impressionistic style, utilizing a rich palette of earthy tones, soft greens, and warm yellows that evoke a sense of tranquility and connection to nature. The camera angle is low, capturing the elephant’s grandeur against the expansive sky, dotted with wispy clouds. In the foreground, a few colorful wildflowers bloom, adding splashes of color, while a distant herd of antelope grazes peacefully, enhancing the serene atmosphere of this enchanting moment in the wild.",
        "ZH": "一只雄伟的大象优雅地站在阳光普照的草原上，它那纹理分明的灰色皮肤在午后温暖的金色阳光下闪闪发光。大象有着大而富有表现力的耳朵和微微弯曲的鼻子，正处于行走的姿态中，踢起一团尘土，向被茂密的绿色金合欢树环绕的波光粼粼的水坑走去。这幅场景以一种充满活力的印象派风格绘制，使用了丰富的泥土色调、柔和的绿色和温暖的黄色，唤起了一种宁静与自然相连的感觉。拍摄角度较低，捕捉到了大象在点缀着缕缕白云的广阔天空下的壮观景象。前景中，几朵色彩斑斓的野花盛开，增添了颜色的斑驳，而远处一群羚羊正在平静地吃草，增强了这片野外迷人时刻的宁静氛围。",
        "DE": "Ein majestätischer Elefant steht anmutig in einer sonnenüberfluteten Savanne, seine texturierte graue Haut glänzt im warmen goldenen Licht der Nachmittagssonne. Der Elefant mit großen ausdrucksstarken Ohren und einem sanft gebogenen Rüssel ist mitten im Schritt, wirbelt eine Staubwolke auf, während er sich auf einen schimmernden Wasserloch zubewegt, umgeben von üppigen grünen Akazienbäumen. Die Szene ist in einem lebendigen impressionistischen Stil gemalt, mit einem reichen Farbspektrum aus Erdtönen, weichen Grüntönen und warmen Gelbtönen, die ein Gefühl von Ruhe und Verbundenheit mit der Natur vermitteln. Der Kamerawinkel ist niedrig, um die Erhabenheit des Elefanten vor dem weiten Himmel, gesprenkelt mit hauchdünnen Wolken, einzufangen. Im Vordergrund blühen einige bunte Wildblumen und setzen farbliche Akzente, während sich in der Ferne eine Herde Antilopen friedlich grasend aufhält und die ruhige Atmosphäre dieses bezaubernden Moments in der Wildnis verstärkt.",
        "FR": "Un éléphant majestueux se tient gracieusement dans une savane baignée de soleil, sa peau grise texturée étincelle sous la douce lumière dorée du soleil de l'après-midi. L'éléphant, avec ses grandes oreilles expressives et sa trompe légèrement courbée, est représenté en plein mouvement, soulevant un nuage de poussière alors qu'il se dirige vers un point d'eau scintillant entouré d'acacias verts luxuriants. La scène est peinte dans un style impressionniste dynamique, utilisant une palette riche de tons terreux, de verts doux et de jaunes chauds qui évoquent un sentiment de tranquillité et de connexion à la nature. L'angle de la caméra est bas, capturant la grandeur de l'éléphant contre le vaste ciel parsemé de légers nuages. En avant-plan, quelques fleurs sauvages colorées sont en pleine floraison, ajoutant des touches de couleur, tandis qu'un troupeau d'antilopes paît paisiblement au loin, renforçant l'atmosphère sereine de ce moment enchanteur dans la nature.",
        "JA": "荘厳な象が日差しに照らされたサバンナに優雅に立ち、そのテクスチャーのある灰色の肌は午後の暖かい黄金色の光の中で輝いています。大きな表現力豊かな耳と優しく曲がった鼻を持つ象は歩みを進める姿勢で、足元から塵の雲を巻き上げながら、緑豊かなアカシアの木々に囲まれたキラキラとした水たまりに向かって移動しています。このシーンは活気に満ちた印象派のスタイルで描かれ、落ち着いた大地の色調、柔らかな緑、そして温かい黄色の豊かなパレットを使用して、平穏さと自然とのつながりを感じさせるものです。カメラアングルは低く設定され、ほんのりとした雲が浮かぶ広大な空を背景に象の壮大さを捉えています。手前にいくつかの色彩豊かな野の花が咲き、彩りを添え、遠くでは一群のインパラが平和に草を食んでおり、野生の中のこの魅惑的な瞬間の静けさを強調しています。",
        "VI": "Một con voi uy nghi đứng thanh lịch trên thảo nguyên đầy nắng, làn da xám của nó có kết cấu sáng lên dưới ánh nắng vàng ấm áp của buổi chiều muộn. Con voi với đôi tai lớn biểu cảm và chiếc vòi cong nhẹ đang đi, tạo ra một đám bụi khi nó di chuyển về phía một hồ nước lấp lánh được bao quanh bởi những cây keo xanh tươi. Cảnh tượng này được vẽ theo phong cách ấn tượng sống động, sử dụng bảng màu đa dạng gồm các sắc độ đất, màu xanh mềm mại và màu vàng ấm áp gợi lên cảm giác yên bình và sự gắn kết với thiên nhiên. Góc máy thấp, chụp được sự hùng vĩ của con voi trước bầu trời rộng lớn điểm xuyết những đám mây mỏng. Phía trước, một vài loài hoa dại nhiều màu sắc nở rộ, thêm vào đó là hình ảnh đàn linh dương ở xa đang ăn cỏ một cách yên bình, làm tăng thêm không khí tĩnh lặng của khoảnh khắc quyến rũ giữa thiên nhiên hoang dã.",
    }]

    for index, prompt_dict in enumerate(prompts):
        for key, prompt in prompt_dict.items():
            for i in range(num_gen_imgs):
                pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(text_prompt=prompt)
                generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"{index}_{key}_{i}")


def image2image(outputs=outputs):
    outputs = os.path.join(outputs, "image2image")
    os.makedirs(outputs, exist_ok=True)

    for i in range(num_gen_imgs):
        pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(images=["./data/image/sea_moon.jpg"])
        generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"sea_moon_{i}")

        pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(images=["./data/image/Sailor_Moon.jpg"])
        generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"Sailor_Moon_{i}")

        pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(images=["./data/image/dog.jpg", "./data/image/hat.jpg"])
        generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"dog_hat_{i}")

        pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(images=["./data/image/dog2.jpg", "./data/image/duck.jpg", "./data/image/glasses.jpg", "./data/image/background.png"])
        generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"dog2_duck_glasses_background_{i}")


def imagetext2image(outputs=outputs):
    outputs = os.path.join(outputs, "imagetext2image")
    os.makedirs(outputs, exist_ok=True)

    for i in range(num_gen_imgs):
        pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(images=["./data/image/yarn_ball.jpg"], text_prompt="Refer to the image style and generate a cute giant panda")
        generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"yarn_ball_panda_{i}")

        for emoji in ["laugh out loud", "sad", "smile"]:
            pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(images=["./data/image/man.jpg"], text_prompt=f"Make the person in the picture {emoji}")
            generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"man_{emoji}_{i}")

        pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(images=["./data/image/hutong.jpg"], text_prompt="Add a car in the picture")
        generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"hutong_car_{i}")

        pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(images=["./data/image/berry_bowl.jpg"], text_prompt="A berry_bowl with a blue house in the background.")
        generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"berry_bowl_house_{i}")

        pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(images=["./data/image/dog2.jpg", "./data/image/backpack.jpg"], text_prompt="with the Eiffel Tower in the background.")
        generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"dog2_backpack_Eiffel_Tower_{i}")

        pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(images=["./data/image/ocr.png"], text_prompt="OCR text recognition.")
        generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"ocr_{i}")


def video2image(outputs=outputs):
    outputs = os.path.join(outputs, "video2image")
    os.makedirs(outputs, exist_ok=True)

    for i in range(num_gen_imgs):
        pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(videos=["./data/video/particle_collision.mp4"])
        generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"sea_moon_{i}")

        pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(videos=["./data/video/mixkit-paper.mp4"])
        generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"mixkit-paper_{i}")

        pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(videos=["./data/video/Skiing.mp4"])
        generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"Skiing_{i}")




def x2image(outputs=outputs):
    outputs = os.path.join(outputs, "x2image")
    os.makedirs(outputs, exist_ok=True)

    for i in range(num_gen_imgs):
        pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(images=["./data/image/Sanxingdui.jpg"], text_prompt="In the abandoned cyberpunk ruins, a mysterious symbiotic relationship has been established between nanobots and humans. These nanobots zip back and forth through the dilapidated buildings, repairing the decaying parts of the city, while humans, by merging with the nanobots, gain extraordinary abilities. In this sci-fi world, a dazzling yet perilous new ecosystem is created, captivating people in the fantasies of the future.")
        generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"Sanxingdui_Mechanical_operation_{i}")

        pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(images=["./data/image/Shuimohua.jpg"])
        generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"Shuimohua_Moonlight_Sonata_{i}")

        pooled_prompt_embeds, prompt_embeds = get_qwen_inputs_embeds(videos=["./data/video/particle_collision.mp4"])
        generate(pooled_prompt_embeds, prompt_embeds, outputs=outputs, filename=f"particle_collision_Electronic_music_{i}")

        


if __name__ == "__main__":
    if task in ["all", "text2image"]:
        text2image()
    if task in ["all", "image2image"]:
        image2image()
    if task in ["all", "imagetext2image"]:
        imagetext2image()
    if task in ["all", "video2image"]:
        video2image()
    if task in ["all", "x2image"]:
        x2image()

    
    

    


