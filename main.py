# requires python 3.10-3.12
from diffusers import DiffusionPipeline
import torch
from torchvision.utils import save_image
import sys
from datetime import datetime
import json
import torch._dynamo
torch._dynamo.config.suppress_errors = True


def main(use_refiner: bool, use_only_gpu: bool, compile_unet: bool):
    if not use_only_gpu:
        print("[INFO] Using CPU instead of GPU (unsupported)")
    print(f"Loading {model_name} (model) from huggingface...")
    # load both base & refiner
    base = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if use_only_gpu else None,
        use_safetensors=True,
        variant="fp16", # Nvidia drives only
    )
    if not use_only_gpu:
        base.enable_attention_slicing() # Saves VRAM
        base.enable_model_cpu_offload()  # Moves layers to CPU when needed
        base.to("cpu", non_blocking=True)
    else:
        base.to("cuda", non_blocking=True)
    print(f"{model_name} loaded.")

    if use_refiner:
        print(f"Loading {refiner_name} (refiner) from huggingface...")
        refiner = DiffusionPipeline.from_pretrained(
            refiner_name,
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16 if use_only_gpu else None,
            use_safetensors=True,
            variant="fp16",
        )
        if not use_only_gpu:
            refiner.enable_attention_slicing() # Saves VRAM
            refiner.enable_model_cpu_offload()  # Moves layers to CPU when needed
            refiner.to("cpu", non_blocking=True)
        else:
            refiner.to("cuda", non_blocking=True)
        print(f"{refiner_name} loaded.")
    
    # Compile the UNet for better performance
    if torch.__version__ >= "2.0" and compile_unet:
        print("Compiling UNet for improved performance...")
        base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
        
        if use_refiner:
            refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)

    # run both experts
    image = base(
        height=image_height,
        width=image_width,
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
        guidance_scale=guidance_scale,
    ).images

    if use_refiner:
        image = refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
            guidance_scale=guidance_scale,
        ).images[0]
    else:
        image = image[0]
    # Save image
    if model_size == "base":
        save_image(image, file_name, format="png", normalize=True)
    if model_size == "xl":
        image.save(file_name)
    print(f"File saved: {file_name}")


def get_models_from_json():
    with open("./models.json", encoding="utf-8", mode="r") as model_json:
        return json.load(model_json)


if __name__ == '__main__':
    # Get prompt from the command line
    args = sys.argv[1:]
    if len(args) != 2:
        print("usage: python nvidia.py \"A majestic lion jumping frmo big stone at night\" xl|base")
        sys.exit(2)
    # "A majestic lion jumping from a big stone at night"
    print("[INFO] Executing stable diffusion model")
    prompt = args[0]
    model_size = args[1]
    # Get model names from json
    models_json = get_models_from_json()
    model_name: str = models_json[model_size]["model_name"]
    refiner_name: str = models_json[model_size]["refiner_name"]
    # Define how many steps and what % of steps to be run on each experts (80/20) here
    n_steps = 40
    # This parameter controls when to stop adding noise during the generation process. A lower value might help preserve more details.
    high_noise_frac = 0.5 # Adjust from 0.5 to 0.8
    guidance_scale = 7.5  # Enable classifier-free guidance
    # save file params
    prompt_to_file_name = "-".join(prompt.split(" ")[1:3])
    file_name = f"images/{prompt_to_file_name}-{model_size}-{datetime.now().timestamp()}.png"
    
    image_height = 512
    image_width = 512
    print(f"Prompt: {prompt}")
    main(use_refiner=len(refiner_name) > 0, use_only_gpu=False, compile_unet=False)
