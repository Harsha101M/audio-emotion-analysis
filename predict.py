import torch
from pathlib import Path
from src.model import EmotionPredictor as Model
from src.predictor import EmotionPredictor
from src.config import *


def main():
    # Initialize model
    model = Model()
    predictor = EmotionPredictor(model, MODELS_DIR / "best_model.pth")

    # Single file prediction example
    audio_path = "path/to/test/audio.mp3"
    if Path(audio_path).exists():
        try:
            prediction = predictor.predict_single(audio_path)
            results = predictor.interpret_prediction(prediction)

            print("\nPrediction Results:")
            print("\nValence Features:")
            for feature, value in results["valence"].items():
                print(f"{feature}: {value:.3f}")

            print("\nArousal Features:")
            for feature, value in results["arousal"].items():
                print(f"{feature}: {value:.3f}")

            predictor.visualize_emotions(results, OUTPUTS_DIR / "single_prediction.png")

        except Exception as e:
            print(f"Error processing single file: {str(e)}")

    # Batch prediction example
    try:
        results = predictor.predict_batch(
            audio_folder=AUDIO_DIR, output_folder=OUTPUTS_DIR
        )
        print(f"\nProcessed {len(results)} audio files successfully")

    except Exception as e:
        print(f"Error during batch processing: {str(e)}")


if __name__ == "__main__":
    main()
