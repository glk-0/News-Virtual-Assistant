#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import threading
from typing import List, Optional, Tuple

class VAState:
    LOCKED            = "locked"
    IDLE              = "idle"
    LISTENING_WAKE    = "listening_wake"
    LISTENING_COMMAND = "listening_cmd"

    def __init__(self):
        self.state: str = self.LOCKED
        self.verified_user: Optional[str] = None
        self.chat_history: List[dict] = []
        self.audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        self._lock = threading.Lock()

    def unlock(self, username: str = "User"):
        with self._lock:
            self.verified_user = username
            self.state = self.LISTENING_WAKE

    def lock(self):
        with self._lock:
            self.verified_user = None
            self.state = self.LOCKED
            self.audio_buffer = np.array([], dtype=np.float32)

    def wake_word_detected(self):
        with self._lock:
            self.state = self.LISTENING_COMMAND

    def command_received(self):
        with self._lock:
            if not self.is_locked:
                self.state = self.LISTENING_WAKE

    @property
    def is_locked(self):
        return self.state == self.LOCKED

    @property
    def is_listening_for_command(self):
        return self.state == self.LISTENING_COMMAND

    SAMPLE_RATE    = 16000
    WINDOW_SECONDS = 2.0

    def append_audio_chunk(self, chunk: np.ndarray) -> np.ndarray:
        with self._lock:
            self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
            max_samples = int(self.SAMPLE_RATE * self.WINDOW_SECONDS)
            if len(self.audio_buffer) > max_samples:
                self.audio_buffer = self.audio_buffer[-max_samples:]
            return self.audio_buffer.copy()

    def add_user_message(self, text: str):
        self.chat_history.append({"role": "user", "content": text})

    def add_atlas_message(self, text: str):
        self.chat_history.append({"role": "atlas", "content": text})

    def get_gradio_history(self) -> List[Tuple[Optional[str], Optional[str]]]:
        pairs, i = [], 0
        while i < len(self.chat_history):
            user_msg = atlas_msg = None
            if self.chat_history[i]["role"] == "user":
                user_msg = self.chat_history[i]["content"]
                if i + 1 < len(self.chat_history) and self.chat_history[i+1]["role"] == "atlas":
                    atlas_msg = self.chat_history[i+1]["content"]
                    i += 2
                else:
                    i += 1
            else:
                atlas_msg = self.chat_history[i]["content"]
                i += 1
            pairs.append((user_msg, atlas_msg))
        return pairs