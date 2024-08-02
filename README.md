
# Voice Cloning with Tacotron 2 and WaveGlow

Welcome to the Voice Cloning with Tacotron 2 and WaveGlow project! This project focuses on cloning voices using Tacotron 2 and WaveGlow models.

## Introduction

Voice cloning involves generating speech that mimics a target voice. In this project, we leverage the power of Tacotron 2 and WaveGlow to perform voice cloning using a dataset of audio samples and their transcriptions.

## Dataset

For this project, we will use a custom dataset of audio samples and their transcriptions. You can create your own dataset and place it in the `data/voice_cloning_data.csv` file.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- PyTorch
- TensorFlow
- Numpy
- Librosa
- Pandas

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/your-username/voice_cloning_tacotron2_waveglow.git
cd voice_cloning_tacotron2_waveglow

# Install the required packages:
pip install -r requirements.txt

# Ensure your data includes audio samples and their transcriptions. Place these files in the data/ directory.
# The data should be in a CSV file with two columns: audio_path and text.

# To fine-tune the Tacotron 2 and WaveGlow models for voice cloning, run the following command:
python scripts/train.py --data_path data/voice_cloning_data.csv

# To evaluate the performance of the fine-tuned models, run:
python scripts/evaluate.py --model_path models/ --data_path data/voice_cloning_data.csv
