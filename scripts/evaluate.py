
import torch
import argparse
import numpy as np
import pandas as pd
from tacotron2 import Tacotron2
from waveglow import WaveGlow
from utils import get_device, preprocess_audio_data, VoiceCloningDataset

def main(model_path, data_path):
    # Load Models
    tacotron2 = Tacotron2()
    waveglow = WaveGlow()
    tacotron2.load_state_dict(torch.load(os.path.join(model_path, 'tacotron2.pth')))
    waveglow.load_state_dict(torch.load(os.path.join(model_path, 'waveglow.pth')))

    # Device
    device = get_device()
    tacotron2.to(device)
    waveglow.to(device)

    # Load Dataset
    dataset = pd.read_csv(data_path)
    preprocessed_data = preprocess_audio_data(dataset)

    # DataLoader
    eval_dataset = VoiceCloningDataset(preprocessed_data)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Evaluation Function
    def evaluate(tacotron2, waveglow, data_loader, device):
        tacotron2.eval()
        waveglow.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                mel_outputs, mel_outputs_postnet, gate_outputs, alignments = tacotron2(inputs)
                mel_loss = torch.nn.functional.mse_loss(mel_outputs, targets) + torch.nn.functional.mse_loss(mel_outputs_postnet, targets)
                gate_loss = torch.nn.functional.binary_cross_entropy(gate_outputs, torch.zeros_like(gate_outputs))

                loss = mel_loss + gate_loss
                total_loss += loss.item()
                total_samples += 1

        avg_loss = total_loss / total_samples
        return avg_loss

    # Evaluate
    avg_loss = evaluate(tacotron2, waveglow, eval_loader, device)
    print(f'Average Loss: {avg_loss}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the directory containing the trained models')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing evaluation data')
    args = parser.parse_args()
    main(args.model_path, args.data_path)
