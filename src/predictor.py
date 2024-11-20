import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import load_and_process_audio
from .config import *


class EmotionPredictor:
    def __init__(self, model, model_path):
        self.model = model
        self.load_model(model_path)
        self.model.eval()

        # Feature names for interpretation
        self.feature_names = {
            "valence": [f.replace("V_", "") for f in VALENCE_FEATURES],
            "arousal": [f.replace("A_", "") for f in AROUSAL_FEATURES],
        }

    def load_model(self, model_path):
        """Load the trained model"""
        checkpoint = torch.load(model_path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(DEVICE)

    def predict_single(self, audio_path):
        """Make prediction for a single audio file"""
        mel_spec = load_and_process_audio(audio_path)
        if mel_spec is None:
            raise ValueError(f"Could not process audio file: {audio_path}")

        mel_spec = mel_spec.unsqueeze(0).to(DEVICE)
        emotional_features = torch.zeros(1, NUM_EMOTIONAL_FEATURES).to(DEVICE)

        with torch.no_grad():
            prediction = self.model(
                {"mel_data": mel_spec, "emotional_features": emotional_features}
            )

        return prediction.cpu().numpy()[0]

    def interpret_prediction(self, prediction):
        """Convert raw predictions to interpretable format"""
        valence_pred = prediction[:11]
        arousal_pred = prediction[11:]

        results = {
            "valence": dict(zip(self.feature_names["valence"], valence_pred)),
            "arousal": dict(zip(self.feature_names["arousal"], arousal_pred)),
        }

        return results

    def visualize_emotions(self, results, save_path=None):
        """Create visualization of emotional predictions"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot Valence features
        sns.barplot(
            x=list(results["valence"].values()),
            y=list(results["valence"].keys()),
            ax=ax1,
            palette="coolwarm",
        )
        ax1.set_title("Valence Features")
        ax1.set_xlabel("Value")

        # Plot Arousal features
        sns.barplot(
            x=list(results["arousal"].values()),
            y=list(results["arousal"].keys()),
            ax=ax2,
            palette="coolwarm",
        )
        ax2.set_title("Arousal Features")
        ax2.set_xlabel("Value")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def predict_batch(self, audio_folder, output_folder=OUTPUTS_DIR):
        """Process multiple audio files and save results"""
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)

        audio_files = list(Path(audio_folder).glob("*.mp3"))
        results = {}

        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            try:
                prediction = self.predict_single(str(audio_file))
                interpreted = self.interpret_prediction(prediction)

                results[audio_file.stem] = interpreted

                self.visualize_emotions(
                    interpreted,
                    save_path=output_folder / f"{audio_file.stem}_emotion.png",
                )

            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                continue

        # Save results to CSV
        self._save_results_to_csv(results, output_folder / "predictions.csv")

        return results

    def _save_results_to_csv(self, results, output_path):
        """Save predictions to CSV file"""
        rows = []
        for audio_id, pred in results.items():
            row = {"audio_id": audio_id}
            for feature, value in pred["valence"].items():
                row[f"valence_{feature.lower()}"] = value
            for feature, value in pred["arousal"].items():
                row[f"arousal_{feature.lower()}"] = value
            rows.append(row)

        pd.DataFrame(rows).to_csv(output_path, index=False)
