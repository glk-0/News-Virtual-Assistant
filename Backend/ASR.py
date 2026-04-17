#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import whisper
import numpy as np

class ASREngine:
    SUPPORTED_LANGUAGES = {"English": "en", "French": "fr"}

    def __init__(self, model_size: str = "base"):
        self.model      = None
        self.model_size = model_size
        self._load_model()

    def _load_model(self):
        print(f"[ASREngine] Loading Whisper '{self.model_size}'...")
        try:
            self.model = whisper.load_model(self.model_size)
            print("[ASREngine] Whisper loaded.")
        except Exception as e:
            print(f"[ASREngine] Failed: {e}")

    def transcribe_file(self, filepath: str, language: str = "en") -> str:
        if self.model is None:
            return "[ASR not available]"
        try:
            result = self.model.transcribe(filepath, language=language)
            return result["text"].strip()
        except Exception as e:
            print(f"[ASREngine] File transcription error: {e}")
            return ""

    def transcribe_numpy(self, audio: np.ndarray, sample_rate: int, language: str = "en") -> str:
        """Transcribe from a numpy array (used by streaming pipeline)."""
        if self.model is None:
            return "[ASR not available]"
        try:
            import torchaudio
            if sample_rate != 16000:
                waveform = torch.tensor(audio).unsqueeze(0)
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                audio    = waveform.squeeze(0).numpy()
            audio = whisper.pad_or_trim(audio.astype(np.float32))
            mel   = whisper.log_mel_spectrogram(audio).to(self.model.device)
            opts  = whisper.DecodingOptions(language=language, fp16=False)
            return whisper.decode(self.model, mel, opts).text.strip()
        except Exception as e:
            print(f"[ASREngine] Numpy transcription error: {e}")
            return ""