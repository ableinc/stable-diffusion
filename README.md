# Stable Diffusion

The models used are small and generic, this should be used for PoC. 

Note: This assumes you're using a Nvidia Cuda-friendly GPU

## Install Python & Create Virtual Environment

Firstly, create a Python virtual environment using Python 3.10. You can download this directly from Python.org, if you're using Linux you must follow your linux distribution install instructions. Once you have Python 3.10 installed, create your virtual environment:

```bash
python3.10 -m venv stable-diffusion
```

***macOS, Linux or WSL2***
```bash
source stable-diffusion/bin/activate
```

***Windows (Powershell)***
```bash
stable-diffusion\Scripts\activate
```

Your Python virtual environment is now active

## Download Python packages

Clone/Download this repository and cd into the root directory.

Note: Ensure your python virtual environment is active before running these scripts. If not, follow steps above.

***macOS, Linux or WSL2***
```bash
sh install.sh
```

***Windows (Powershell)***
```bash
.\install.ps1
```

## Run scripts

Note: Ensure your python virtual environment is active before running these scripts. If not, follow steps above.

Using the XL model (decent):

```bash
python main.py "A majestic lion jumping from a big stone at night" xl
```

Using the base model (less decent):

```bash
python main.py "A majestic lion jumping from a big stone at night" base
```

## Models

The models used can be found in the ```models.json``` file. These are Huggingface models. You can visit the stable diffusion Huggingface page to get other models. Just create a new object in the ```models.json```. Stable diffusion Huggingface site: https://huggingface.co/stabilityai


## Troubleshooting

The script by default uses the CPU and offloads computation to any available CUDA-friendly GPU installed. If you only want to use your GPU (one with large VRAM), you can edit the ```main.py``` file on line 117. Change it from:

```python
main(use_refiner=len(refiner_name) > 0, use_only_gpu=False, compile_unet=False)
```

to 

```python
main(use_refiner=len(refiner_name) > 0, use_only_gpu=True, compile_unet=False)
```

Its important to note, by doing this it may require additional libraries, especially if you want to compile unet prior to running the pipeline. The image will be saved to the ```images``` directory in the repo.
