import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from .utils import load_and_process_audio
from .config import *


class AudioEmotionDataset(Dataset):
    def __init__(self, audio_dir, metadata_file):
        self.audio_dir = Path(audio_dir)
        self.metadata = pd.read_csv(metadata_file, sep="\t")

        print("Validating audio files...")
        self._validate_files()
        print("Normalizing features...")
        self._normalize_features()

    def _validate_files(self):
        """Remove entries where audio files don't exist or are corrupted"""
        valid_files = []
        for idx, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            audio_path = self.audio_dir / f"{row['id']}.mp3"
            if audio_path.exists():
                if load_and_process_audio(str(audio_path)) is not None:
                    valid_files.append(idx)

        self.metadata = self.metadata.loc[valid_files].reset_index(drop=True)
        print(f"Found {len(self.metadata)} valid audio files")

    def _normalize_features(self):
        """Normalize all emotional features using z-score normalization"""
        for feature in VALENCE_FEATURES + AROUSAL_FEATURES:
            mean = self.metadata[feature].mean()
            std = self.metadata[feature].std()
            self.metadata[feature] = (self.metadata[feature] - mean) / (std + 1e-8)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_id = self.metadata.iloc[idx]["id"]
        audio_path = self.audio_dir / f"{audio_id}.mp3"

        mel_spec = load_and_process_audio(str(audio_path))
        if mel_spec is None:
            mel_spec = torch.zeros((N_MELS, DURATION * TARGET_SR // HOP_LENGTH))
        mel_spec = mel_spec.unsqueeze(0)

        valence_features = torch.tensor(
            self.metadata.iloc[idx][VALENCE_FEATURES].values, dtype=torch.float32
        )
        arousal_features = torch.tensor(
            self.metadata.iloc[idx][AROUSAL_FEATURES].values, dtype=torch.float32
        )

        emotional_features = torch.cat([valence_features, arousal_features])

        return {
            "mel_data": mel_spec,
            "emotional_features": emotional_features,
            "id": audio_id,
        }
