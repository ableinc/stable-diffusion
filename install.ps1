#!/usr/bin/env pwsh
# Requires Python 3.10
# Uninstall packages listed in requirements.txt
python -m pip uninstall -r requirements.txt -y
# Purge pip cache
python -m pip cache purge
# Install PyTorch packages
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# Install packages from requirements.txt
python -m pip install -r requirements.txt
