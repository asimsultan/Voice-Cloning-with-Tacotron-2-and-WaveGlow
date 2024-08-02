
import os
import torch
import argparse
import numpy as np
import pandas as pd
from tacotron2 import Tacotron2
from waveglow import WaveGlow
from datasets import load_dataset
from utils import get_device, preprocess_audio_data, VoiceCloningDataset

def main(data_path):
    # Parameters
    batch_size = 16
    epochs = 100
    learning_rate = 1e-3

    # Load Dataset
    dataset = pd.read_csv(data_path)

    # Preprocess Data
    preprocessed_data = preprocess_audio_data(dataset)

    # DataLoader
    train_dataset = VoiceCloningDataset(preprocessed_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Models
    device = get_device()
    tacotron2 = Tacotron2().to(device)
    waveglow = WaveGlow().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(list(tacotron2.parameters()) + list(waveglow.parameters()), lr=learning_rate)

    # Training Function
    def train_epoch(tacotron2, waveglow, data_loader, optimizer, device):
        tacotron2.train()
        waveglow.train()
        total_loss = 0

        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = tacotron2(inputs)
            mel_loss = torch.nn.functional.mse_loss(mel_outputs, targets) + torch.nn.functional.mse_loss(mel_outputs_postnet, targets)
            gate_loss = torch.nn.functional.binary_cross_entropy(gate_outputs, torch.zeros_like(gate_outputs))

            loss = mel_loss + gate_loss
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    # Training Loop
    for epoch in range(epochs):
        train_loss = train_epoch(tacotron2, waveglow, train_loader, optimizer, device)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss}')

    # Save Models
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(tacotron2.state_dict(), os.path.join(model_dir, 'tacotron2.pth'))
    torch.save(waveglow.state_dict(), os.path.join(model_dir, 'waveglow.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing voice cloning data')
    args = parser.parse_args()
    main(args.data_path)
