#!/bin/bash

# Create Python 3.10 environment
conda create --name whisperx python=3.10
conda activate whisperx

# Install PyTorch 2.0 and other dependencies
conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

apt install ffmpeg

# Install WhisperX
pip install git+https://github.com/m-bain/whisperx.git

# Additional installations
# You may add additional commands for ffmpeg, rust, etc.
