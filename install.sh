#!/bin/sh
# requires python 3.10
python -m pip uninstall -r requirements.txt -y
python -m pip cache purge
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt