# Audio Emotion Analysis Project

This project implements a deep learning model to predict emotional characteristics (valence and arousal) from audio files using a combination of LSTM networks and mel spectrograms.

## Project Structure
```
audio_emotion_analysis/
├── data/
│   ├── audio/                  # Your audio files (.mp3)
│   └── id_lyrics_sentiment_functionals.tsv
├── models/                     # Saved model checkpoints
├── outputs/                    # Prediction results and visualizations
├── src/
│   ├── config.py              # Configuration settings
│   ├── dataset.py             # Dataset handling
│   ├── model.py               # Neural network architecture
│   ├── trainer.py             # Training logic
│   ├── predictor.py           # Prediction pipeline
│   └── utils.py               # Utility functions
├── requirements.txt           # Project dependencies
├── train.py                   # Training script
└── predict.py                 # Prediction script
```

## Setup

1. Create a new virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For GPU support, install PyTorch with CUDA:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# OR for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Data Preparation

1. Place your audio files in the `data/audio/` directory:
   - Files should be in MP3 format
   - Each file should be approximately 30 seconds long
   - Naming format: `<id>.mp3`

2. Place your metadata file (`id_lyrics_sentiment_functionals.tsv`) in the `data/` directory:
   - TSV file should contain columns for:
     - Valence features (V_mean, V_std, etc.)
     - Arousal features (A_mean, A_std, etc.)
   - Each row should correspond to an audio file

## Training

1. Verify your GPU setup (if using GPU):
```python
python -c "import torch; print(torch.cuda.is_available())"
```

2. Start training:
```bash
python train.py
```

The training process will:
- Validate all audio files
- Create train/validation splits
- Train the model with early stopping
- Save checkpoints and visualizations in the `models/` directory
- Generate training history plots

Training parameters can be adjusted in `src/config.py`.

## Making Predictions

1. Using trained model for predictions:
```bash
python predict.py
```

This will:
- Load the best model from training
- Process audio files and generate predictions
- Save visualizations and results in the `outputs/` directory

## Output Files

After prediction, you'll find:
- Individual emotion plots for each audio file
- A CSV file containing all predictions
- Visualization plots showing valence and arousal features

## Model Architecture

The model uses:
- Mel spectrogram representations of audio
- Dual LSTM layers for temporal processing
- Additional pathways for emotional feature processing
- Combined processing for final predictions

## Customization

You can modify various parameters in `src/config.py`:
- Audio processing parameters (sample rate, mel bands, etc.)
- Model architecture (hidden size, layers, dropout)
- Training parameters (batch size, learning rate, etc.)
- File paths and directories

## Troubleshooting

1. GPU Issues:
   - Verify CUDA installation: `nvidia-smi`
   - Check PyTorch CUDA: `torch.cuda.is_available()`
   - Update GPU drivers if necessary

2. Audio Processing Issues:
   - Check audio file format and duration
   - Verify sample rate compatibility
   - Ensure sufficient disk space for processed files

3. Memory Issues:
   - Reduce batch size in `config.py`
   - Decrease number of workers
   - Process shorter audio segments
