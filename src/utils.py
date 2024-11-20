import torch
import librosa
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from .config import *


def load_and_process_audio(audio_path, sr=TARGET_SR, duration=DURATION):
    """Load and process audio file to mel spectrogram"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)

        # Ensure consistent length
        if len(y) > MAX_FILE_SIZE:
            y = y[:MAX_FILE_SIZE]
        elif len(y) < MAX_FILE_SIZE:
            y = np.pad(y, (0, MAX_FILE_SIZE - len(y)))

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=sr / 2
        )

        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        return torch.FloatTensor(mel_spec)

    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None


def plot_losses(train_losses, val_losses, save_path):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def save_checkpoint(
    model, optimizer, epoch, train_loss, val_loss, scaler, path, is_best=False
):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "scaler": scaler.state_dict(),
        "model_config": {
            "input_size": model.lstm.input_size,
            "hidden_size": model.lstm.hidden_size,
            "num_layers": model.lstm.num_layers,
            "num_emotional_features": model.emotional_fc[0].in_features,
        },
    }

    torch.save(checkpoint, path)

    if is_best:
        best_path = Path(path).parent / "best_model.pth"
        torch.save(checkpoint, best_path)


def load_checkpoint(model, optimizer, scaler, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint["epoch"], checkpoint["train_loss"], checkpoint["val_loss"]
