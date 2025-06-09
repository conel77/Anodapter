from diffusers import StableDiffusionAdapterPipeline, T2IAdapter
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import json
import argparse
from diffusers import StableDiffusionAdapterPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig
from diffusers import AutoencoderKL, DDIMScheduler
from safetensors.torch import load_file


def load_prompt_mapping(txt_path):
    mapping = {}
    with open(txt_path, "r") as f:
        for line in f:
            if "->" in line:
                key, value = line.strip().split("->")
                mapping[key.strip()] = value.strip()
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--object_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--prompt_txt_path", type=str)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--anomaly_type", type=str, required=True)
    args = parser.parse_args()

    prompt_dict = load_prompt_mapping(args.prompt_txt_path)

    if args.anomaly_type not in prompt_dict:
        raise ValueError(f"Anomaly type '{args.anomaly_type}' not found in prompt TXT.")

    type_prompt = prompt_dict[args.anomaly_type]
    print(type_prompt)
    mask_prompt = f"a photo of sks with {type_prompt}"
    img_prompt = "a photo of dqd"

    mask_dir = os.path.join(args.save_dir, args.anomaly_type, "mask")
    img_dir = os.path.join(args.save_dir, args.anomaly_type, "image")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    object_mask_path = os.path.join(args.object_path, "0001.png")
    if not os.path.exists(object_mask_path):
        raise FileNotFoundError(f"Object mask not found at {object_mask_path}")

    first_image = Image.open(object_mask_path)
    original_width, original_height = first_image.size
    if original_height >= original_width:
        height = 512
        new_width = (original_width * 512) / original_height
        width = int(round(new_width / 8) * 8)
    else:
        width = 512
        new_height = (original_height * 512) / original_width
        height = int(round(new_height / 8) * 8)

    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ])

    mask_transforms = transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    object_mask = transform(first_image.convert("L"))[None, :, :, :].expand(args.batch, -1, -1, -1)

    

    # UNet
    unet_config_path = os.path.join(args.model_path, "unet", "config.json")
    unet_ckpt_path = os.path.join(args.model_path, "unet", "diffusion_pytorch_model.safetensors")

    unet = UNet2DConditionModel.from_config(unet_config_path)
    unet_state_dict = load_file(unet_ckpt_path)
    missing, unexpected = unet.load_state_dict(unet_state_dict, strict=False)
    print("UNet Missing:", missing)
    print("UNet Unexpected:", unexpected)

    # T2IAdapter (Mask)
    mask_adapter_config = os.path.join(args.adapter_path, "adapter_1", "config.json")
    mask_adapter_ckpt = os.path.join(args.adapter_path, "adapter_1", "diffusion_pytorch_model.safetensors")
    mask_adapter = T2IAdapter.from_config(mask_adapter_config)
    mask_state_dict = load_file(mask_adapter_ckpt)
    mask_adapter.load_state_dict(mask_state_dict, strict=False)

    # T2IAdapter (Image)
    img_adapter_config = os.path.join(args.adapter_path, "adapter", "config.json")
    img_adapter_ckpt = os.path.join(args.adapter_path, "adapter", "diffusion_pytorch_model.safetensors")
    img_adapter = T2IAdapter.from_config(img_adapter_config)
    img_state_dict = load_file(img_adapter_ckpt)
    img_adapter.load_state_dict(img_state_dict, strict=False)

    # Text Encoder
   
    text_encoder_ckpt_path = os.path.join(args.model_path, "text_encoder", "model.safetensors")
    text_encoder_config_path = os.path.join(args.model_path, "text_encoder", "config.json")
    text_encoder_config = CLIPTextConfig.from_json_file(text_encoder_config_path)

    # config로부터 모델 초기화
    text_encoder = CLIPTextModel(config=text_encoder_config)


    text_encoder_state_dict = load_file(text_encoder_ckpt_path)

    # 가중치 적용
    missing, unexpected = text_encoder.load_state_dict(text_encoder_state_dict, strict=False)
    print("Text Encoder Missing:", missing)
    print("Text Encoder Unexpected:", unexpected)

    # tokenizer는 pretrained 방식 그대로 사용
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(args.model_path, "tokenizer"))

    
    # VAE
    vae_config_path = os.path.join(args.model_path, "vae", "config.json")
    vae = AutoencoderKL.from_config(vae_config_path)

    # 2. safetensors 가중치 로드
    vae_weights_path = os.path.join(args.model_path, "vae", "diffusion_pytorch_model.safetensors")
    vae_state_dict = load_file(vae_weights_path)

    # 3. 가중치 로드 (strict=False로 누락 키 무시)
    missing, unexpected = vae.load_state_dict(vae_state_dict, strict=False)

    # Scheduler
    scheduler = DDIMScheduler.from_pretrained(args.model_path, subfolder="scheduler")

    # 어댑터 로드
    #mask_adapter = T2IAdapter.from_pretrained(adapter_path, torch_dtype=torch.float32, subfolder="mask_t2iadapter")
    #img_adapter = T2IAdapter.from_pretrained(adapter_path, torch_dtype=torch.float32, subfolder="image_t2iadapter")

    # 파이프라인 구성 (mask adapter용)
    mask_pipe = StableDiffusionAdapterPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        adapter=mask_adapter,
        safety_checker=None,
        feature_extractor=None,
    ).to("cuda")

    # 파이프라인 구성 (image adapter용)
    img_pipe = StableDiffusionAdapterPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        adapter=img_adapter,
        safety_checker=None,
        feature_extractor=None,
    ).to("cuda")
    

    for i in range(0, args.num, args.batch):
        prompts = [mask_prompt] * args.batch
        mask_output = mask_pipe(prompts, image=object_mask, height=height, width=width, num_inference_steps=args.step)
        mask_list = []
        for j, mask_img in enumerate(mask_output.images):
            image_np = np.array(mask_img)
            gray = image_np.mean(2)  # RGB 평균 -> grayscale
            binarized = (gray > 200).astype(np.uint8) * 255
            bin_img = Image.fromarray(binarized, mode="L")
            mask_path_full = os.path.join(mask_dir, f"{i + j}.png")
            bin_img.save(mask_path_full)
            mask_tensor = mask_transforms(bin_img)[None, :, :, :]
            mask_list.append(mask_tensor)

        images = img_pipe([img_prompt] * args.batch, image=torch.cat(mask_list).to("cuda"), height=height, width=width, num_inference_steps=args.step)
        for j, out_img in enumerate(images.images):
            out_path = os.path.join(img_dir, f"{i + j}.png")
            out_np = np.array(out_img)
            if out_img.mode == "RGB" and np.allclose(out_np[..., 0], out_np[..., 1]) and np.allclose(out_np[..., 1], out_np[..., 2]):
                out_img = out_img.convert("L")
                print(f"[INFO] Converted to grayscale (8-bit): {out_path}")
            out_img.save(out_path)


if __name__ == "__main__":
    main()
