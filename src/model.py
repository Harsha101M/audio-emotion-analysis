import torch
import torch.nn as nn
from .config import *


class EmotionCNN(nn.Module):
    def __init__(
        self, input_size=N_MELS, num_emotional_features=NUM_EMOTIONAL_FEATURES
    ):
        super().__init__()

        # First conv block
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )

        # Second conv block
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )

        # Third conv block
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )

        # Fourth conv block
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )

        # Adaptive pooling to handle variable length inputs
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Emotional features processing
        self.emotional_fc = nn.Sequential(
            nn.Linear(num_emotional_features, 256), nn.ReLU(), nn.Dropout(0.3)
        )

        # Combine CNN and emotional features
        self.final_fc = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_emotional_features),
        )

    def forward(self, x):
        # Process mel spectrogram through CNN
        x_mel = x["mel_data"]

        x_mel = self.conv1(x_mel)
        x_mel = self.conv2(x_mel)
        x_mel = self.conv3(x_mel)
        x_mel = self.conv4(x_mel)

        # Global average pooling
        x_mel = self.adaptive_pool(x_mel)
        x_mel = x_mel.squeeze(-1)

        # Process emotional features
        x_emo = self.emotional_fc(x["emotional_features"])

        # Combine features
        combined = torch.cat([x_mel, x_emo], dim=1)

        # Final prediction
        output = self.final_fc(combined)

        return output
