
import json
import braceexpand
import webdataset as wds
from tqdm import tqdm
import torch
from torchvision.transforms.functional import crop
import re
from torchvision import transforms
import random
import numpy as np
import os,glob
from transformers import T5Tokenizer, T5ForConditionalGeneration,T5EncoderModel,MT5EncoderModel,AutoTokenizer,AutoModel,AutoModelForCausalLM
from transformers import T5EncoderModel, T5TokenizerFast, CLIPTokenizer, AutoProcessor
import pdb 

from pytorch_lightning import LightningDataModule
from typing import Optional
from torch.utils.data import random_split
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from core.data.dataloader import PreprocessDataLoader

class DataModuleCustom(LightningDataModule):
    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Universal DataModule')
        parser.add_argument('--webdataset_base_urls', type=str, nargs="+")
        parser.add_argument('--num_workers', default=2, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        # parser.add_argument('--start_shard', default=0, type=int)
        # parser.add_argument('--end_shard', default=1000, type=int)
        parser.add_argument('--shard_width', default=5, type=int)
        parser.add_argument('--hr_size', default=-1, type=int)
        parser.add_argument('--train_split', default=1.0, type=float)
        parser.add_argument('--val_split', default=0.0, type=float)
        parser.add_argument('--test_split', default=0.0, type=float)
        parser.add_argument('--shuffle_train',default=True, action="store_true")
        parser.add_argument('--resample_train',default=True, action="store_true")
        parser.add_argument('--shuffle_num', default=None, type=int)
        parser.add_argument('--test_prompts', type=str,
                            default="./test_prompts.txt")
        parser.add_argument('--test_repeat', default=1, type=int)

        parser.add_argument(
            "--resolution", type=int, default=512,
            help=(
                "The resolution for input images, all the images in the train/validation dataset will be resized to this"
                " resolution"
            ),
        )
        parser.add_argument(
            "--center_crop", action="store_true", default=False,
            help="Whether to center crop images before resizing to resolution"
        )
        return parent_args

    def __init__(
        self,
        args,
        tokenizer_t5,
        tokenizer_t5_en,
        tokenizer_en,
        custom_collate_fn=None,
        use_worker_init_fn=None,
    ):
        super().__init__()
        # self.available_shards = list(range(args.start_shard, args.end_shard + 1))
        # if splits is None:
        #     splits = []
        splits = {
            'train': args.train_split,
            'val': args.val_split,
            'test': args.test_split,
        }
        self.webdataset_base_urls = args.webdataset_base_urls
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.shuffle_train = args.shuffle_train
        self.resample_train = args.resample_train
        self.shard_width = args.shard_width
        self.hr_size = args.hr_size
        self.use_worker_init_fn = use_worker_init_fn
        self.shuffle_num = args.shuffle_num
        self.tokenizer_t5 = tokenizer_t5
        self.tokenizer_t5_en = tokenizer_t5_en
        self.tokenizer_en = tokenizer_en
        self.collate_fn = custom_collate_fn if custom_collate_fn is not None else collate_fn
        self.center_crop = args.center_crop
        self.resolution = args.resolution

        self.train_prop = self.val_prop = self.test_prop = 0
        self.datasets = {}
        if splits['train'] > 0:
            self.train_prop = splits['train']
            self.train_dataloader = self._train_dataloader
            self.datasets['train'] = None


        self.prepare_data()
        self.setup()

    def prepare_data(self):
        assert self.train_prop + self.test_prop + self.val_prop == 1

        all_urls = []
        for url in self.webdataset_base_urls:
            if "*" in url:
                all_urls += expand_urls1(url)
            else:
                all_urls += expand_urls(url)
        num_train = round(self.train_prop*len(all_urls))
        num_test = round(self.test_prop*len(all_urls))
        num_val = len(all_urls) - num_train - num_test
        assert num_train + num_test + \
            num_val == len(
                all_urls), f"{num_train} + {num_test} + {num_val} = {num_train + num_test + num_val} != {len(all_urls)}"
        self.train_urls, self.test_urls, self.val_urls = random_split(
            all_urls, [num_train, num_test, num_val])  # , generator=torch.Generator().manual_seed(self.seed)
    

    def setup(self, stage=None):
        if 'train' in self.datasets:
            self.datasets['train'] = ImageEmbeddingDataset(
                self.train_urls,
                self.tokenizer_t5,
                self.tokenizer_t5_en,
                self.tokenizer_en,
                shuffle_shards=self.shuffle_train,
                resample=self.resample_train,
                hr_size=self.hr_size,
                handler=wds.handlers.warn_and_continue,
                center_crop=self.center_crop,
                size=self.resolution,
            )

            if self.shuffle_num is not None and self.shuffle_num > 0:
                self.datasets['train'].shuffle(self.shuffle_num)

    def _train_dataloader(self, preprocess=None, process_group=None, group_rank=None):

        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        # print(f"rank: {dist.get_global_rank(process_group, group_rank)}, group_rank: {group_rank}, local_rank: {dist.get_rank(process_group)}")
        # sampler = DistributedSampler(dataset=self.datasets['train'], num_replicas=dist.get_world_size(process_group), rank=dist.get_rank(process_group))
        sampler=None
        # return DataLoader(
        # num_workers=self.num_workers,
        return PreprocessDataLoader(
            preprocess=preprocess,
            sampler=sampler,
            dataset=self.datasets['train'],
            num_workers=0,
            batch_size=self.batch_size,
            prefetch_factor=None,  # This might be good to have high so the next npy file is prefetched
            pin_memory=True,
            shuffle=False,
            worker_init_fn=init_fn,
            collate_fn=self.collate_fn,
            timeout=600,
        )



TETX_ENCODER = "chinese_clip"  ## mul_clip  chinese_clip  mt5  alt_clip

USED_KEYS = {"json": "instance_prompt_ids"}

def str_contain_chinese(str):
    for ch in str:
        if u'\u4e00'<=ch<=u'\u9fff':
            return True
    return False

def expand_urls(urls):
    if isinstance(urls, str):
        urllist = urls.split("::")
        result = []
        for url in urllist:
            result.extend(braceexpand.braceexpand(url))
        return result
    else:
        return list(urls)

def expand_urls1(urls):
    result = []
    for file_ in glob.glob(urls):
        result.append(file_)
    return result


def verify_keys(samples, required_keys, handler=wds.handlers.reraise_exception):
    """
    Requires that both the image and embedding are present in the sample
    This is important to do as a user may forget they do not have embeddings in their webdataset and neglect to add them using the embedding_folder_url parameter.
    """

    for sample in samples:
        try:
            sample_json = sample["json"]
        except:
            print("#######sample",sample)
            continue

        yield sample


key_verifier = wds.filters.pipelinefilter(verify_keys)


def crop_left_upper(image, size):
    w, h = image.size

    detla_w = w-size[0]
    detla_h = h-size[1]
    x = random.randint(0, detla_w)
    y = random.randint(0, detla_h)
    return (y, x), crop(image, y, x, size[1], size[0])


class ImageEmbeddingDataset(wds.DataPipeline, wds.compat.FluidInterface):
    """
    A fluid interface wrapper for DataPipline that returns image embedding pairs
    Reads embeddings as npy files from the webdataset if they exist. If embedding_folder_url is set, they will be inserted in from the alternate source.
    """

    def __init__(
            self,
            urls,
            tokenizer_t5 = None,
            tokenizer_t5_en = None,
            tokenizer_en = None,
            hr_size=-1,
            size=512,
            handler=wds.handlers.reraise_exception,
            resample=True,
            shuffle_shards=True,
            center_crop=False
    ):

        super().__init__()
        keys = list(USED_KEYS.keys())
        # self.key_map = {key: i for i, key in enumerate(keys)}
        self.resampling = resample
        self.hr_size = hr_size
        self.center_crop = center_crop
        self.crop = transforms.CenterCrop(
            size) if center_crop else crop_left_upper
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.tokenizer_t5 = tokenizer_t5
        self.tokenizer_t5_en = tokenizer_t5_en
        self.tokenizer_en = tokenizer_en

        self.append(wds.ResampledShards(urls))

        self.append(wds.tarfile_to_samples(handler=handler))

        self.append(wds.decode("pilrgb", handler=handler))

        self.append(key_verifier(required_keys=keys, handler=handler))
        # Apply preprocessing
        self.append(wds.map(self.preproc))
        # self.append(wds.to_tuple(*keys))

    def preproc(self, sample):
        """Applies the preprocessing for images"""
        
        example = {}
        sample_json = sample["json"]        
        example["instance_en"] = sample_json["caption_en"]
        
        Instructions = {"Text input":example["instance_en"],"Instruction editing description":"no","image input":"no"}
        message = [
            {
                "role": "user",
                "content": str(Instructions)
            }
        ]
        prompt = self.tokenizer_t5.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)


        inputs = self.tokenizer_t5(
            [prompt],
            images=None,
            audios=None,
            max_slice_nums=1,
            use_image_id=False,
            chunk_input=True,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            sampling_rate=16000,
             add_special_tokens=True
        )
        example["input_ids_t5"] = inputs.input_ids
        example["attention_mask"] = inputs.attention_mask
        # ['input_ids', 'attention_mask', 'pixel_values', 'image_sizes', 'image_bound', 'tgt_sizes', 'audio_bounds', 'spk_bounds', 'audio_features', 'audio_feature_lens']
        # t5_en
        text_inputs = self.tokenizer_t5_en(
            [example["instance_en"]],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        example["input_ids_t5_en"] = text_inputs.input_ids


        # en
        text_inputs = self.tokenizer_en(
            [example["instance_en"]],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        example["input_ids_en"] = text_inputs.input_ids


        # text_inputs = reward_model.blip.tokenizer(example["instance_en"], padding='max_length', truncation=True, max_length=77, return_tensors="pt")
        # example["en_input_ids"] = text_inputs.input_ids
        # example["en_attention_mask"] = text_inputs.attention_mask

        return example


def collate_fn(examples):
    instance_prompt_ids = [example["instance_en"] for example in examples]
    input_ids_t5 = [example["input_ids_t5"] for example in examples]

    # en_attention_mask = [example["en_attention_mask"] for example in examples]
    # en_input_ids = [example["en_input_ids"] for example in examples]

    input_ids_t5_en = [example["input_ids_t5_en"] for example in examples]
    input_ids_en = [example["input_ids_en"] for example in examples]
    attention_mask = [example["attention_mask"] for example in examples]

    batch = {
        "input_ids_t5": torch.cat(input_ids_t5),
        "input_ids_t5_en": torch.cat(input_ids_t5_en),
        "input_ids_en": torch.cat(input_ids_en),
        "attention_mask": torch.cat(attention_mask),
    }

    return batch


if __name__ == '__main__':
    device="cuda"
    urls=["/mnt/data/group/text2img_data/sd3/*/*"] # caption_en,  caption_zh,  
    # urls=["/mnt/data/group/text2img_data/qwen_vl/journeyDB/*/*"] # ## caption_qwen_en caption_qwen
    urls=["/mnt/data/group/text2img_data/data_process/laion0.3B_trans_webdataset/{00000..51975}.tar"]
    # urls=["/mnt/data/group/text2img_data/data_process/aesthetics_tar_5/*"]
    urls=["/mnt/data/group/text2img_data/data_process/laion2b_5.5/{00000..99999}.tar"]
    # urls=["/mnt/data/group/text2img_data/data_process/coyo1/{00000..51975}.tar"]
    all_urls = []
    for url in urls:
        if "*" in url:
            all_urls += expand_urls1(url)
        elif ".." in url:
            all_urls += expand_urls(url)
        else:
            all_urls = urls
    print(len(all_urls))
    model_path = "/mnt/data/group/models/flux/MiniCPM-o-2_6"
    text_encoder_t5 = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation='sdpa', # sdpa or flash_attention_2
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=True
        ).eval().to(device)
    tokenizer_1 = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer_t5 = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    teacher = "/mnt/data/group/models/flux/shuttle-3-diffusion"
    tokenizer = CLIPTokenizer.from_pretrained(teacher, subfolder="tokenizer", revision="refs/pr/1")
    tokenizer_2 = T5TokenizerFast.from_pretrained(teacher, subfolder="tokenizer_2", revision="refs/pr/1")

    ds = ImageEmbeddingDataset(
        all_urls,
        tokenizer_t5,
        tokenizer_2,
        tokenizer,
        resample=True,
        hr_size=512,
        handler=wds.handlers.warn_and_continue
    )

    dl = DataLoader(
            ds,
            num_workers=1,
            batch_size=1,
            prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
            pin_memory=True,
            shuffle=False,
            collate_fn=collate_fn
        )

    fw = open("tmp.txt","w")
    outputs = "images/dalle3"
    os.makedirs(outputs, exist_ok=True)
    for i, batch in enumerate(tqdm(dl)):
        # inputs={"input_ids":batch["input_ids_t5"].to(device),"attention_mask":batch["attention_mask"].to(device),"tgt_sizes":[[]],"pixel_values":[[]]}
        inputs={"input_ids":batch["input_ids_t5"].to(device),"attention_mask":batch["attention_mask"].to(device),"pixel_values":[[]]}
        output_hidden_state= text_encoder_t5.generate(**inputs,tokenizer=tokenizer_1,max_new_tokens=2048,decode_text=False)
        print(output_hidden_state.hidden_states[0].shape)
        print(len(output_hidden_state.hidden_states))
