import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler

EMOTIONS = {
    0: "happy",  # Joy, elation, cheerfulness
    1: "sad",  # Melancholy, sorrow, grief
    2: "angry",  # Rage, frustration, annoyance
    3: "love",  # Romance, affection, passion
    4: "fear",  # Anxiety, worry, dread
    5: "peaceful",  # Calm, serenity, tranquility
    6: "nostalgic",  # Reminiscence, longing, remembrance
    7: "hopeful",  # Optimism, anticipation, desire
}


def compute_emotional_profile(features):
    """Compute emotional profile using scaled features"""

    # Calculate composite scores
    valence_score = (
        features["valence_mean"] * 0.2
        + features["valence_median"] * 0.15
        + features["valence_skewness"] * 0.15
        + features["valence_kurtosis"] * 0.15
        + features["valence_variation"] * 0.15
        + (features["valence_max"] - features["valence_min"]) * 0.2
    )

    arousal_score = (
        features["arousal_mean"] * 0.2
        + features["arousal_median"] * 0.15
        + features["arousal_skewness"] * 0.15
        + features["arousal_kurtosis"] * 0.15
        + features["arousal_variation"] * 0.15
        + (features["arousal_max"] - features["arousal_min"]) * 0.2
    )

    # Compute intensity
    intensity = (
        abs(features["valence_quartile3"] - features["valence_quartile1"]) * 0.3
        + abs(features["arousal_quartile3"] - features["arousal_quartile1"]) * 0.3
        + abs(features["valence_variation"]) * 0.2
        + abs(features["arousal_variation"]) * 0.2
    )

    return valence_score, arousal_score, intensity


def map_to_emotion(valence_score, arousal_score, intensity):
    """Map scores to emotions using adaptive thresholds"""
    # Use median as dynamic threshold
    v_threshold = 0
    a_threshold = 0
    i_threshold = np.median([abs(valence_score), abs(arousal_score)])

    if valence_score > v_threshold:
        if arousal_score > a_threshold:
            return 0 if intensity > i_threshold else 7  # happy vs hopeful
        else:
            return 5 if intensity > i_threshold else 3  # peaceful vs love
    else:
        if arousal_score > a_threshold:
            return 2 if intensity > i_threshold else 4  # angry vs fear
        else:
            return 1 if intensity > i_threshold else 6  # sad vs nostalgic


def process_predictions(csv_path):
    """Process predictions with increased sensitivity to small variations"""
    # Read predictions
    df = pd.read_csv(csv_path)

    # Scale all numeric features
    feature_columns = [col for col in df.columns if col != "audio_id"]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_columns])

    # Convert back to DataFrame with feature names
    scaled_df = pd.DataFrame(scaled_features, columns=feature_columns)
    scaled_df["audio_id"] = df["audio_id"]

    # Process each row
    results = {}
    for idx, row in scaled_df.iterrows():
        # Compute emotional profile
        v_score, a_score, intensity = compute_emotional_profile(row)

        # Map to emotion
        emotion_id = map_to_emotion(v_score, a_score, intensity)
        emotion = EMOTIONS[emotion_id]

        # Store results
        results[row["audio_id"]] = {
            "emotion": emotion,
            "emotion_id": emotion_id,
            "valence_score": float(v_score),
            "arousal_score": float(a_score),
            "intensity": float(intensity),
            "scaled_features": {
                "valence_mean": float(row["valence_mean"]),
                "valence_std": float(row["valence_std"]),
                "valence_median": float(row["valence_median"]),
                "arousal_mean": float(row["arousal_mean"]),
                "arousal_std": float(row["arousal_std"]),
                "arousal_median": float(row["arousal_median"]),
            },
        }

    return results


def main():
    predictions_path = Path("outputs/predictions.csv")
    results = process_predictions(predictions_path)

    # Analyze distribution
    emotions = [data["emotion"] for data in results.values()]
    emotion_counts = pd.Series(emotions).value_counts()

    print("\nEmotion Distribution:")
    print("--------------------")
    for emotion, count in emotion_counts.items():
        percentage = (count / len(emotions)) * 100
        print(f"{emotion.capitalize()}: {count} ({percentage:.1f}%)")

    # Save detailed results
    output_path = Path("outputs/emotion_predictions.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    # Create detailed CSV
    details = []
    for audio_id, data in results.items():
        row = {
            "audio_id": audio_id,
            "emotion": data["emotion"],
            "valence_score": data["valence_score"],
            "arousal_score": data["arousal_score"],
            "intensity": data["intensity"],
        }
        details.append(row)

    pd.DataFrame(details).to_csv("outputs/detailed_emotions.csv", index=False)

    print("\nSample Predictions:")
    print("------------------")
    for audio_id, data in list(results.items())[:5]:
        print(f"\nAudio ID: {audio_id}")
        print(f"Emotion: {data['emotion']}")
        print(f"Scores:")
        print(f"  Valence: {data['valence_score']:.3f}")
        print(f"  Arousal: {data['arousal_score']:.3f}")
        print(f"  Intensity: {data['intensity']:.3f}")


if __name__ == "__main__":
    main()
