import torch
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
from .utils import plot_losses, save_checkpoint
from .config import *


class Trainer:
    def __init__(
        self, model, train_loader, val_loader, criterion, optimizer, save_dir=MODELS_DIR
    ):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_dir = Path(save_dir)
        self.scaler = GradScaler()

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()

            mel_data = batch["mel_data"].to(DEVICE)
            emotional_features = batch["emotional_features"].to(DEVICE)

            with autocast():
                outputs = self.model(
                    {"mel_data": mel_data, "emotional_features": emotional_features}
                )
                loss = self.criterion(outputs, emotional_features)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                mel_data = batch["mel_data"].to(DEVICE)
                emotional_features = batch["emotional_features"].to(DEVICE)

                with autocast():
                    outputs = self.model(
                        {"mel_data": mel_data, "emotional_features": emotional_features}
                    )
                    loss = self.criterion(outputs, emotional_features)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, num_epochs=NUM_EPOCHS, patience=PATIENCE):
        print(f"Starting training for {num_epochs} epochs...")
        best_val_loss = float("inf")
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            train_loss = self.train_epoch()
            train_losses.append(train_loss)

            val_loss = self.validate()
            val_losses.append(val_loss)

            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")

            # Save checkpoint
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                train_loss,
                val_loss,
                self.scaler,
                self.save_dir / f"checkpoint_epoch_{epoch}.pth",
                is_best=(val_loss < best_val_loss),
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after epoch {epoch+1}")
                break

        # Plot training history
        plot_losses(train_losses, val_losses, self.save_dir / "training_history.png")
        return train_losses, val_losses
