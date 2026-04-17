#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import numpy as np
import torch
import time

import os
os.environ["ESPEAK_DATA_PATH"] = "/usr/share/espeak-ng-data"

class TTSEngine:
    def __init__(self):
        print("[TTSEngine] Loading Kokoro TTS pipelines...")
        # Load English (American) pipeline
        self.pipeline_en = KPipeline(lang_code='a')    
        try:
            # Load French pipeline if required by user choice
            self.pipeline_fr = KPipeline(lang_code='f')
        except Exception as e:
            print(f"[TTSEngine] French pipeline failed to load: {e}. Falling back to English.")
            self.pipeline_fr = self.pipeline_en

    def generate_audio(self, text: str, language: str = "English") -> str:
        """Generates TTS audio and saves it to a UNIQUE wav file."""
        pipeline = self.pipeline_fr if language == "French" else self.pipeline_en
        voice = 'ff_siwis' if language == "French" else 'af_heart'

        # FIX: Create a unique filename for every single response
        output_path = f"atlas_response_{int(time.time())}.wav"

        try:
            generator = pipeline(text, voice=voice, speed=1, split_pattern=r'\n+')
            audio_segments = []

            for i, (gs, ps, audio) in enumerate(generator):
                audio_segments.append(audio)

            if audio_segments:
                final_audio = np.concatenate(audio_segments)
                sf.write(output_path, final_audio, 24000)
                return output_path # Returns the unique path to Gradio
        except Exception as e:
            print(f"[TTSEngine] Audio generation error: {e}")
        return None
