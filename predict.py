import torch
import torch.nn as nn
import librosa
import numpy as np
import sys
import os
import argparse

class AlexNet_Emotions(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        # BLOCK 1: Large stride conv + maxpool (AlexNet signature)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 96, 11, stride=4, padding=2),   # C1: 128×173→32×43×96
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),                   # S2: →15×21×96
            nn.BatchNorm2d(96)                           # Phase 2 ✓
        )

        # BLOCK 2: 5×5 convs (parallel in original, sequential here)
        self.block2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, padding=2),            # C2: 15×21→15×21×256
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),                   # S3: →7×10×256
            nn.BatchNorm2d(256)                          # Phase 2 ✓
        )

        # BLOCKS 3-5: 3×3 convs (AlexNet modernization)
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(384)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(384)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(3, stride=2), nn.BatchNorm2d(256)  # Pooling ✓
        )

        # Phase 2: GAP + Dropout
        self.global_pool = nn.AdaptiveAvgPool2d(1)       # GAP ✓

        # AlexNet FC layers + Phase 2 Dropout ✓
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5), nn.Linear(256, 4096), nn.ReLU(),  # FC6
            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(), # FC7
            nn.Dropout(0.3), nn.Linear(4096, num_classes)      # FC8: 8 emotions
        )

    def forward(self, x):
        x = self.block1(x)  # 128×173 → 15×21×96
        x = self.block2(x)  # 15×21 → 7×10×256
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)  # 7×10 → 3×4×256
        x = self.global_pool(x)  # GAP → 1×1×256 ✓
        x = self.classifier(x)
        return x

# RAVDESS 8-class emotion mapping (indices 0-7)
emotion_names = [
    'neutral', 'calm', 'happy', 'sad', 
    'angry', 'fearful', 'disgust', 'surprised'
]

def wav_to_melspec(wav_file, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    """Match your exact preprocessing: 128x173 mel-spectrogram"""
    y, sr = librosa.load(wav_file, sr=sr)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db[np.newaxis, np.newaxis, :, :]  # (1,1,H,W)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotion from WAV file")
    parser.add_argument("wav_file", help="Path to input .wav file")
    parser.add_argument("--weights", default="current_alexnet_weights.pth", 
                       help="Path to model weights (default: current_alexnet_weights.pth)")
    args = parser.parse_args()

    # Setup device (MPS/CUDA/CPU fallback)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load model + weights
    model = AlexNet_Emotions(8).to(device)
    if not os.path.exists(args.weights):
        print(f"Weights not found: {args.weights}")
        print("Expected: current_alexnet_weights.pth (~86MB)")
        sys.exit(1)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Preprocess WAV → mel-spec
    if not os.path.exists(args.wav_file):
        print(f"WAV file not found: {args.wav_file}")
        sys.exit(1)

    mel_spec = wav_to_melspec(args.wav_file)
    mel_t = torch.FloatTensor(mel_spec).to(device)

    # Inference
    with torch.no_grad():
        logits = model(mel_t)
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(1).item()
        confidence = probs[0, pred_idx].item()

    print(f" File: {os.path.basename(args.wav_file)}")
    print(f"Predicted: {emotion_names[pred_idx]} (confidence: {confidence:.1%})")
