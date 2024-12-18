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
        # Fix the torch.load warning by adding weights_only=True
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(DEVICE)

    def predict_single(self, audio_path):
        """Make prediction for a single audio file with diagnostics"""
        # Load and process audio
        mel_spec = load_and_process_audio(audio_path)
        if mel_spec is None:
            raise ValueError(f"Could not process audio file: {audio_path}")

        # Print mel spectrogram stats
        print(f"\nDiagnostics for {audio_path}:")
        print(f"Mel spectrogram shape: {mel_spec.shape}")
        print(f"Mel spectrogram range: [{mel_spec.min():.3f}, {mel_spec.max():.3f}]")
        print(f"Mel spectrogram mean: {mel_spec.mean():.3f}")
        print(f"Mel spectrogram std: {mel_spec.std():.3f}")

        mel_spec = mel_spec.unsqueeze(0).to(DEVICE)
        mel_spec = mel_spec.squeeze(1).permute(0, 2, 1)
        emotional_features = torch.zeros(1, NUM_EMOTIONAL_FEATURES).to(DEVICE)

        # Print transformed shape
        print(f"Input shape to model: {mel_spec.shape}")

        with torch.no_grad():
            # Get intermediate activations
            self.model.eval()
            prediction = self.model(
                {"mel_data": mel_spec, "emotional_features": emotional_features}
            )

            # Print prediction stats
            print(
                f"Prediction range: [{prediction.min().item():.3f}, {prediction.max().item():.3f}]"
            )
            print(f"Prediction mean: {prediction.mean().item():.3f}")
            print(f"Prediction std: {prediction.std().item():.3f}")

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

        # Fix seaborn warning by using data parameter and explicit hue
        valence_data = pd.DataFrame(
            {
                "Feature": list(results["valence"].keys()),
                "Value": list(results["valence"].values()),
            }
        )

        arousal_data = pd.DataFrame(
            {
                "Feature": list(results["arousal"].keys()),
                "Value": list(results["arousal"].values()),
            }
        )

        # Updated barplots with proper parameters
        sns.barplot(
            data=valence_data,
            x="Value",
            y="Feature",
            ax=ax1,
            palette="coolwarm",
            hue="Value",
            legend=False,
        )
        ax1.set_title("Valence Features")
        ax1.set_xlabel("Value")

        sns.barplot(
            data=arousal_data,
            x="Value",
            y="Feature",
            ax=ax2,
            palette="coolwarm",
            hue="Value",
            legend=False,
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
        """Process multiple audio files and save results with additional stats"""
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)

        audio_files = list(Path(audio_folder).glob("*.mp3"))
        results = {}
        stats = {
            "mel_spec_shapes": [],
            "mel_spec_means": [],
            "mel_spec_stds": [],
            "prediction_means": [],
            "prediction_stds": [],
        }

        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            try:
                # Load mel spectrogram for stats
                mel_spec = load_and_process_audio(str(audio_file))
                if mel_spec is not None:
                    stats["mel_spec_shapes"].append(mel_spec.shape)
                    stats["mel_spec_means"].append(mel_spec.mean().item())
                    stats["mel_spec_stds"].append(mel_spec.std().item())

                # Get prediction
                prediction = self.predict_single(str(audio_file))
                interpreted = self.interpret_prediction(prediction)

                stats["prediction_means"].append(np.mean(prediction))
                stats["prediction_stds"].append(np.std(prediction))

                results[audio_file.stem] = interpreted

                # self.visualize_emotions(
                #     interpreted,
                #     save_path=output_folder / f"{audio_file.stem}_emotion.png"
                # )

            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                continue

        # Save results to CSV
        self._save_results_to_csv(results, output_folder / "predictions.csv")

        # Save statistics
        stats_df = pd.DataFrame(
            {
                "mel_spec_mean": stats["mel_spec_means"],
                "mel_spec_std": stats["mel_spec_stds"],
                "prediction_mean": stats["prediction_means"],
                "prediction_std": stats["prediction_stds"],
            }
        )
        stats_df.to_csv(output_folder / "prediction_stats.csv", index=False)

        # Print summary statistics
        print("\nPrediction Statistics:")
        print(f"Number of files processed: {len(results)}")
        print(f"Average mel spectrogram mean: {np.mean(stats['mel_spec_means']):.3f}")
        print(f"Average mel spectrogram std: {np.mean(stats['mel_spec_stds']):.3f}")
        print(f"Average prediction mean: {np.mean(stats['prediction_means']):.3f}")
        print(f"Average prediction std: {np.mean(stats['prediction_stds']):.3f}")
        print(
            f"Prediction mean range: [{np.min(stats['prediction_means']):.3f}, {np.max(stats['prediction_means']):.3f}]"
        )

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
