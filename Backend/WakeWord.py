#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
from typing import List, Optional, Tuple


class WakeWordClassifier(nn.Module):
    """
    Lightweight MLP head on top of OpenWakeWord AudioFeatures (96-d).
    """
    def __init__(self, input_dim=96):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)
    

# ─────────────────────────────────────────────
# WAKE WORD DETECTOR
# ─────────────────────────────────────────────

class WakeWordDetector:
    WAKE_WORD_PHRASE    = "Hey Atlas"
    DETECTION_THRESHOLD = 0.7
    SAMPLE_RATE         = 16000

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.classifier     = None
        self.feat_extractor = None
        self.device         = torch.device(device)

        self._load_feature_extractor()
        if model_path:
            self._load_classifier(model_path)
        else:
            print("[WakeWordDetector] No model path — stub mode.")

    def _load_feature_extractor(self):
        try:
            from openwakeword.utils import AudioFeatures
            self.feat_extractor = AudioFeatures()
            print("[WakeWordDetector] AudioFeatures extractor loaded.")
        except Exception as e:
            print(f"[WakeWordDetector] Could not load AudioFeatures: {e}")

    def _load_classifier(self, path):
        try:
            clf = WakeWordClassifier(input_dim=96)
            # Saved with torch.save(model.state_dict(), path) — raw state dict
            state_dict = torch.load(path, map_location=self.device)
            clf.load_state_dict(state_dict)
            clf.to(self.device)
            clf.eval()
            self.classifier = clf
            print(f"[WakeWordDetector] Classifier loaded from {path}")
        except Exception as e:
            print(f"[WakeWordDetector] Failed to load classifier: {e}")
            self.classifier = None

    def process_window(self, audio_window: np.ndarray):
        """
        audio_window: float32 numpy array at 16kHz, up to 2 seconds.
        Mirrors extract_embedding() from OpenWakeWord notebook exactly.
        """
        if self.feat_extractor is None or self.classifier is None:
            return False, 0.0

        try:
            FIXED_LENGTH = 16000 * 2  # 2 seconds

            # 1. Pad or truncate to exactly 2 seconds
            if len(audio_window) < FIXED_LENGTH:
                audio_window = np.pad(audio_window, (0, FIXED_LENGTH - len(audio_window)))
            else:
                audio_window = audio_window[:FIXED_LENGTH]

            # 2. Convert to int16, add batch dim → (1, 32000)
            y_int16 = (np.clip(audio_window, -1.0, 1.0) * 32767).astype(np.int16)
            y_batch = y_int16[np.newaxis, :]

            # 3. Extract 96-d embedding via embed_clips, average over frames
            features  = self.feat_extractor.embed_clips(x=y_batch)
            embedding = np.mean(features[0], axis=0)  # (96,)

            # 4. Run classifier
            x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logit = self.classifier(x)             # (1, 1)
                prob  = torch.sigmoid(logit).item()

            detected = prob >= self.DETECTION_THRESHOLD
            return detected, prob

        except Exception as e:
            print(f"[WakeWordDetector] Inference error: {e}")
            return False, 0.0

    def text_trigger(self, text: str) -> bool:
        return text.strip().lower() == self.WAKE_WORD_PHRASE.lower()

