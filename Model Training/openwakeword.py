# -*- coding: utf-8 -*-

#!pip install openwakeword
#!pip install librosa torch scikit-learn tqdm

import os
import torch
import librosa
import numpy as np
import openwakeword
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')

# --- CONFIGURATION ---
# Replace these with the actual paths
POSITIVE_DIR = "/content/drive/MyDrive/raw_recordings/positive"
NEGATIVE_DIR = "/content/drive/MyDrive/raw_recordings/negative"

MODEL_SAVE_PATH = "/content/drive/MyDrive/Morning-Virtual-Assistant/Hey_Atlas_WakeWord_best.pth"

# OpenWakeWord models expect 16kHz audio
TARGET_SAMPLE_RATE = 16000

import openwakeword
import warnings
import numpy as np
import librosa
import os
from tqdm import tqdm

F = openwakeword.utils.AudioFeatures()

FIXED_LENGTH_SAMPLES = int(TARGET_SAMPLE_RATE * 2.0)

def extract_embedding(y):
    """Standardizes length and extracts embeddings from a raw audio array."""
    if len(y) < FIXED_LENGTH_SAMPLES:
        pad_length = FIXED_LENGTH_SAMPLES - len(y)
        y = np.pad(y, (0, pad_length), mode='constant')
    elif len(y) > FIXED_LENGTH_SAMPLES:
        y = y[:FIXED_LENGTH_SAMPLES]

    y_int16 = (y * 32767).astype(np.int16)
    y_batch = y_int16[np.newaxis, :]
    features = F.embed_clips(x=y_batch)

    return np.mean(features[0], axis=0)

def augment_audio(y, sr):
    """Applies random noise, volume changes, and time shifts to the audio."""
    # Pick a random augmentation strategy for this specific copy
    aug_type = np.random.choice(['noise', 'volume', 'shift', 'all'])
    y_aug = np.copy(y)

    if aug_type in ['noise', 'all']:
        # Add subtle white noise
        noise_amp = 0.015 * np.random.uniform() * np.amax(y_aug)
        y_aug = y_aug + noise_amp * np.random.normal(size=y_aug.shape[0])

    if aug_type in ['volume', 'all']:
        # Randomly scale volume between 50% and 150%
        y_aug = y_aug * np.random.uniform(0.5, 1.5)

    if aug_type in ['shift', 'all']:
        # Shift the audio left or right by up to 0.2 seconds
        shift_amount = np.random.randint(int(-sr * 0.2), int(sr * 0.2))
        y_aug = np.roll(y_aug, shift_amount)

    return y_aug

def process_directory(directory, label, augment_factor=1):
    features = []
    labels = []
    files = [f for f in os.listdir(directory) if f.endswith(('.wav', '.m4a', '.mp3'))]

    print(f"Processing {len(files)} files in {directory} (Generating {augment_factor}x samples)...")
    for file in tqdm(files):
        audio_path = os.path.join(directory, file)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE)

            if y is None or len(y) == 0:
                continue

            # 1. Always extract the feature for the ORIGINAL, untouched audio
            features.append(extract_embedding(y))
            labels.append(label)

            # 2. Generate and extract features for AUGMENTED versions
            for _ in range(augment_factor - 1):
                y_augmented = augment_audio(y, sr)
                features.append(extract_embedding(y_augmented))
                labels.append(label)

        except Exception:
            # Silently skip totally broken files
            pass

    return features, labels

# --- THE DATA BALANCING ---
# Generate 5 total versions of every "Hey Atlas" clip to fight class imbalance
pos_features, pos_labels = process_directory(POSITIVE_DIR, label=1.0, augment_factor=3)

# Keep the negative class as-is (just the original recordings)
neg_features, neg_labels = process_directory(NEGATIVE_DIR, label=0.0, augment_factor=2)

# Combine datasets
X = np.array(pos_features + neg_features)
y = np.array(pos_labels + neg_labels)

print(f"\nTotal dataset shape: {X.shape}")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class WakeWordClassifier(nn.Module):
    def __init__(self, input_dim=96):
        super(WakeWordClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),   # Reduced from 64
            nn.ReLU(),
            nn.Dropout(0.5),            # Increased from 0.3
            nn.Linear(32, 16),          # Reduced from 32
            nn.ReLU(),
            nn.Dropout(0.5),            # Increased from 0.3
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

# Split the data 80% Train / 20% Test
# stratify=y ensures the 80/20 split maintains the same ratio of "Hey Atlas" vs classmates in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create separate DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize the model
model = WakeWordClassifier()

import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Calculate class weights based ONLY on the training data
num_pos = np.sum(y_train == 1.0)
num_neg = np.sum(y_train == 0.0)

pos_weight_value = num_neg / num_pos
print(f"Calculated positive class weight: {pos_weight_value:.2f}")
pos_weight = torch.tensor([pos_weight_value])

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

EPOCHS = 256

# Dictionaries to store the metrics for plotting later
history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

print("Starting Training...")
for epoch in range(EPOCHS):

    # ==========================
    #      TRAINING PHASE
    # ==========================
    model.train()
    train_loss, correct_train, total_train = 0, 0, 0

    for batch_X, batch_y in train_dataloader:
        optimizer.zero_grad()

        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Calculate Train Accuracy
        probabilities = torch.sigmoid(logits)
        predicted_classes = (probabilities > 0.5).float()
        correct_train += (predicted_classes == batch_y).sum().item()
        total_train += batch_y.size(0)

    # ==========================
    #      TESTING PHASE
    # ==========================
    model.eval() # Turn off Dropout for testing
    test_loss, correct_test, total_test = 0, 0, 0

    with torch.no_grad(): # Don't track gradients during testing to save memory
        for batch_X, batch_y in test_dataloader:
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            test_loss += loss.item()

            # Calculate Test Accuracy
            probabilities = torch.sigmoid(logits)
            predicted_classes = (probabilities > 0.5).float()
            correct_test += (predicted_classes == batch_y).sum().item()
            total_test += batch_y.size(0)

    # ==========================
    #      RECORD METRICS
    # ==========================
    epoch_train_loss = train_loss / len(train_dataloader)
    epoch_test_loss = test_loss / len(test_dataloader)
    epoch_train_acc = (correct_train / total_train) * 100
    epoch_test_acc = (correct_test / total_test) * 100

    history['train_loss'].append(epoch_train_loss)
    history['test_loss'].append(epoch_test_loss)
    history['train_acc'].append(epoch_train_acc)
    history['test_acc'].append(epoch_test_acc)

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.2f}%")

print("Training Complete!")

# ==========================
#   PLOT LEARNING CURVES
# ==========================
plt.figure(figsize=(14, 5))

# Subplot 1: Loss
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', color='blue')
plt.plot(history['test_loss'], label='Test (Validation) Loss', color='red')
plt.title('Model Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Subplot 2: Accuracy
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy', color='blue')
plt.plot(history['test_acc'], label='Test (Validation) Accuracy', color='red')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# Save the trained weights
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Wake word model successfully saved to: {MODEL_SAVE_PATH}")