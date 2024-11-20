import torch
import torch.nn as nn
from .config import *


class EmotionPredictor(nn.Module):
    def __init__(
        self,
        input_size=N_MELS,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_emotional_features=NUM_EMOTIONAL_FEATURES,
        dropout_p=DROPOUT,
    ):
        super().__init__()

        # LSTM layers for processing mel spectrogram
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_p
        )
        self.lstm2 = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout_p
        )

        # Additional layers for processing emotional features
        self.emotional_fc = nn.Sequential(
            nn.Linear(num_emotional_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        # Combined processing
        self.final_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, num_emotional_features),
        )

    def forward(self, x):
        # Process mel spectrogram through LSTM
        lstm_out, _ = self.lstm(x["mel_data"])
        lstm_out2, _ = self.lstm2(lstm_out)
        audio_features = lstm_out2[:, -1, :]

        # Process emotional features
        emotional_features = self.emotional_fc(x["emotional_features"])

        # Combine features
        combined = torch.cat([audio_features, emotional_features], dim=1)

        # Final prediction
        output = self.final_fc(combined)
        return output
