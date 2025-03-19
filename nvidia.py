# requires python 10
from diffusers import DiffusionPipeline
import torch
from torchvision.utils import save_image
import sys
from datetime import datetime
import json


def gpu_is_supported():
    if torch.version.hip is not None and torch.cuda.is_available():
        return True
    return False


def main(use_refiner: bool, use_cpu: bool):
    if use_cpu:
        print("[INFO] Using CPU instead of GPU (unsupported)")
    print(f"Loading {model_name} (model) from huggingface...")
    # load both base & refiner
    base = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if not use_cpu else None,
        use_safetensors=True,
        variant="fp16", # Nvidia drives only
    )
    if use_cpu:
        base.to("cpu")
    else:
        base.enable_attention_slicing() # Saves VRAM
        base.enable_model_cpu_offload()  # Moves layers to CPU when needed
        base.to("cuda", non_blocking=True)
    print(f"{model_name} loaded.")

    if use_refiner:
        print(f"Loading {refiner_name} (refiner) from huggingface...")
        refiner = DiffusionPipeline.from_pretrained(
            refiner_name,
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16 if not use_cpu else None,
            use_safetensors=True,
            variant="fp16",
        )
        if use_cpu:
            base.to("cpu")
        else:
            refiner.enable_attention_slicing() # Saves VRAM
            refiner.enable_model_cpu_offload()  # Moves layers to CPU when needed
            refiner.to("cuda", non_blocking=True)
        print(f"{refiner_name} loaded.")

    # run both experts
    image = base(
        height=image_height,
        width=image_width,
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    if use_refiner:
        image = refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]
    else:
        image = image[0]
    # Save image
    save_image(image, file_name, format="png", normalize=True)
    print(f"File saved: {file_name}")


def get_models_from_json():
    with open("./models.json", encoding="utf-8", mode="r") as model_json:
        return json.load(model_json)


if __name__ == '__main__':
    # "A majestic lion jumping from a big stone at night"
    print("[INFO] Executing stable diffusion model")
    # Get prompt from the command line
    args = sys.argv[1:]
    if len(args) != 1:
        raise ("You must provide the prompt as the argument")
    prompt = args[0]
    # Get model names from json
    models_json = get_models_from_json()
    model_name: str = models_json["base"]["model_name"]
    refiner_name: str = models_json["base"]["refiner_name"]
    # Define how many steps and what % of steps to be run on each experts (80/20) here
    n_steps = 40
    # This parameter controls when to stop adding noise during the generation process. A lower value might help preserve more details.
    high_noise_frac = 0.5 # Adjust from 0.5 to 0.8
    # save file params
    prompt_to_file_name = "-".join(prompt.split(" ")[1:3])
    file_name = f"{prompt_to_file_name}-{datetime.now().timestamp()}.png"
    
    image_height = 512
    image_width = 512
    print(f"Prompt: {prompt}")
    main(use_refiner=len(refiner_name) > 0, use_cpu=gpu_is_supported())