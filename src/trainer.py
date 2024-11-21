import torch
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
from .utils import plot_losses
from .config import *


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = GradScaler()

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()

            # Move data to device - no need to reshape for CNN
            mel_data = batch["mel_data"].to(DEVICE).squeeze(1)
            # mel_data = mel_data.squeeze(1).permute(
            #     0, 2, 1
            # )  # Shape: [batch, channels, time]
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
                mel_data = batch["mel_data"].to(DEVICE).squeeze(1)
                # mel_data = mel_data.squeeze(1).permute(0, 2, 1)
                emotional_features = batch["emotional_features"].to(DEVICE)

                with autocast():
                    outputs = self.model(
                        {"mel_data": mel_data, "emotional_features": emotional_features}
                    )
                    loss = self.criterion(outputs, emotional_features)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, num_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE):
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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "scaler": self.scaler.state_dict(),
                    },
                    MODELS_DIR / "best_model.pth",
                )
                print("Saved best model!")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after epoch {epoch+1}")
                break

        # Plot training history
        plot_losses(train_losses, val_losses, OUTPUTS_DIR / "training_history.png")
        return train_losses, val_losses
