import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import AudioEmotionDataset
from src.model import EmotionCNN
from src.trainer import Trainer
from src.config import *


def main():
    # GPU setup and debugging
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Create dataset
    print("\nCreating dataset...")
    dataset = AudioEmotionDataset(
        audio_dir=AUDIO_DIR,
        metadata_file=DATA_DIR / "id_lyrics_sentiment_functionals.tsv",
    )

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize model
    model = EmotionCNN()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer)

    # Train model
    try:
        train_losses, val_losses = trainer.train()
        print("Training completed successfully!")
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
