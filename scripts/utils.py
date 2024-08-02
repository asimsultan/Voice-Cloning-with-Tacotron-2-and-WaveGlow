
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

class VoiceCloningDataset(Dataset):
    def __init__(self, data):
        self.audio_paths = data['audio_path']
        self.texts = data['text']

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        text = self.texts[idx]

        # Load and preprocess audio
        waveform, sr = librosa.load(audio_path, sr=22050)
        mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=80)

        return torch.tensor(mel_spectrogram).float(), torch.tensor(text).long()

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_audio_data(dataset):
    preprocessed_data = {
        "audio_path": dataset['audio_path'].tolist(),
        "text": dataset['text'].tolist()
    }
    return preprocessed_data
