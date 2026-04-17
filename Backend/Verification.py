#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import os
import traceback

class AtlasVerificationModel(nn.Module):
    """
    ECAPA-TDNN with a fine-tuned classification head.
    """
    def __init__(self, embedding_model, num_classes=5):
        super().__init__()
        self.embedding_model = embedding_model
        self.fc = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        embeddings = self.embedding_model(x)   # (B, 1, 192)
        embeddings = embeddings.squeeze(1)       # (B, 192)
        return self.fc(embeddings)


# ─────────────────────────────────────────────
# USER VERIFIER
# ─────────────────────────────────────────────

class UserVerifier:
    BYPASS_PASSWORD  = "CSI5180"
    FALLBACK_LABELS  = ["Other", "Connor", "Coralie", "Ghali"]  # matches training label_to_name
    POSITIVE_CLASSES = {"Ghali", "Coralie", "Connor"}
    SAMPLE_RATE      = 16000

    def __init__(self, model_path=None, device="cpu"):
        self.model    = None
        self.backbone = None
        self.labels   = self.FALLBACK_LABELS
        self.device   = torch.device(device)
        print(model_path)
        if model_path:
            self._load_model(model_path)
        else:
            print("[UserVerifier] No model path — stub mode.")

    def _load_model(self, path):
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            self.backbone = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/ecapa_backbone",
                run_opts={"device": str(self.device)}
            )

            checkpoint = torch.load(path, map_location=self.device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict  = checkpoint["model_state_dict"]
                num_classes = checkpoint.get("num_classes", len(self.FALLBACK_LABELS))
                self.labels = checkpoint.get("class_labels", self.FALLBACK_LABELS[:num_classes])
            else:
                # Raw state dict
                state_dict  = checkpoint
                last_weight = [v for k, v in state_dict.items() if "weight" in k][-1]
                num_classes = last_weight.shape[0]
                self.labels = self.FALLBACK_LABELS[:num_classes]

            self.model = AtlasVerificationModel(
                embedding_model=self.backbone.mods.embedding_model,
                num_classes=num_classes
            )
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            print(f"[UserVerifier] Loaded. Classes: {self.labels}")

        except Exception as e:
            print(f"[UserVerifier] Failed to load model: {e}")
            traceback.print_exc()
            self.model = None
            self.backbone = None

    def verify_password(self, password):
        if password.strip() == self.BYPASS_PASSWORD:
            return True, "Bypass"
        return False, None

    def verify_audio(self, audio: np.ndarray, sample_rate: int):
      """
      Mirrors handle_verification() + verify_user() from the working UI notebook exactly.
      Saves to a temp wav so torchaudio.load handles all format details.
      """
      if self.model is None or self.backbone is None:
          print("[UserVerifier] Stub mode.")
          return False, None, 0.0

      try:
          import torchaudio

          # ── Step 1: Save to temp wav  ──
          # Gradio gives int16 numpy; sf.write needs that exact format
          temp_wav_path = "temp_verification.wav"
          sf.write(temp_wav_path, audio, sample_rate)

          # ── Step 2: Load with torchaudio (same as verify_user in working code) ──
          waveform, sr = torchaudio.load(temp_wav_path)

          # ── Step 3: Resample to 16kHz ──
          if sr != 16000:
              resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
              waveform = resampler(waveform)

          # ── Step 4: Convert to mono ──
          if waveform.shape[0] > 1:
              waveform = torch.mean(waveform, dim=0, keepdim=True)

          # ── Step 5: Pad or truncate to 48000 samples ──
          MAX_LENGTH = 48000
          if waveform.shape[1] > MAX_LENGTH:
              waveform = waveform[:, :MAX_LENGTH]
          else:
              padding = MAX_LENGTH - waveform.shape[1]
              waveform = torch.nn.functional.pad(waveform, (0, padding))

          waveform = waveform.to(self.device)

          with torch.no_grad():
              # ── Step 6: Extract features ──
              features = self.backbone.mods.compute_features(waveform)
              features = self.backbone.mods.mean_var_norm(
                  features,
                  torch.ones(features.shape[0]).to(self.device)
              )

              # ── Step 7: Classify ──
              output = self.model(features)              # (1, num_classes)
              _, predicted = torch.max(output.data, 1)
              class_id = predicted.item()

          # ── Step 8: Clean up temp file ──
          if os.path.exists(temp_wav_path):
              os.remove(temp_wav_path)

          # ── Step 9: Map to name (matches label_to_name from training notebook) ──
          pred_name = self.labels[class_id] if class_id < len(self.labels) else "Unknown"
          is_verified = pred_name in self.POSITIVE_CLASSES

          print(f"[UserVerifier] → class_id={class_id}, name={pred_name}, verified={is_verified}")
          return is_verified, (pred_name if is_verified else None), float(class_id)

      except Exception as e:
          # Clean up temp file even on failure
          if os.path.exists("temp_verification.wav"):
              os.remove("temp_verification.wav")
          print(f"[UserVerifier] Inference error: {e}")
          return False, None, 0.0

verifier = UserVerifier(model_path="./ModelWeights/User_verification_model.pth")