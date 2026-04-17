#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import os
import json
from transformers import BertModel, BertTokenizerFast

class JointBERT(nn.Module):
    """
    Shared BERT encoder with two heads:
      - intent_head : [CLS] → num_intents  (classification)
      - slot_head   : every token → num_slots (sequence labelling / BIO)
    """
    def __init__(self, num_intents: int, num_slots: int,
                 model_name: str = "bert-base-uncased", dropout: float = 0.1):
        super().__init__()
        self.bert        = BertModel.from_pretrained(model_name)
        hidden           = self.bert.config.hidden_size          # 768
        self.dropout     = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden, num_intents)
        self.slot_head   = nn.Linear(hidden, num_slots)

    def forward(self, input_ids, attention_mask):
        out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls  = self.dropout(out.last_hidden_state[:, 0, :])   # [B, H]
        seq  = self.dropout(out.last_hidden_state)             # [B, T, H]
        return {
            "intent_logits": self.intent_head(cls),   # [B, num_intents]
            "slot_logits":   self.slot_head(seq),     # [B, T, num_slots]
        }


# ──────────────────────────────────────────────────────────────
# Main classifier class
# ──────────────────────────────────────────────────────────────
class IntentClassifier:
    """
    Wraps the trained JointBERT model for inference.

    Parameters
    ----------
    model_dir : str
        Directory that contains:
            model.pth      — saved state dict  (torch.save)
            meta.json      — intents, slot_labels, max_len, model_name
            tokenizer files — saved via tokenizer.save_pretrained()
    device : str | None
        'cuda', 'cpu', or None (auto-detect)
    """

    def __init__(self, model_dir: str, device: str | None = None):
        self.model_dir = model_dir

        # ── Device ──────────────────────────────────────────
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── Load metadata ────────────────────────────────────
        meta_path = os.path.join(model_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found in {model_dir}")

        with open(meta_path) as f:
            meta = json.load(f)

        self.intents     = meta["intents"]
        self.slot_labels = meta["slot_labels"]
        self.intent2id   = meta["intent2id"]
        self.slot2id     = meta["slot2id"]
        self.id2intent   = {int(v): k for k, v in self.intent2id.items()}
        self.id2slot     = {int(v): k for k, v in self.slot2id.items()}
        self.max_len     = meta["max_len"]
        self.model_name  = meta.get("model_name", "bert-base-uncased")

        # ── Load tokenizer ───────────────────────────────────
        self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)

        # ── Build & load model ───────────────────────────────
        self.model = JointBERT(
            num_intents = len(self.intents),
            num_slots   = len(self.slot_labels),
            model_name  = self.model_name,
        ).to(self.device)

        weights_path = os.path.join(model_dir, "model.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"model.pth not found in {model_dir}")

        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device)
        )
        self.model.eval()
        print(f"[IntentClassifier] Loaded from '{model_dir}' on {self.device}")


    # ──────────────────────────────────────────────────────────
    # Core prediction
    # ──────────────────────────────────────────────────────────
    def predict(self, text: str, verbose: bool = False) -> dict:
        """
        Run joint intent + slot inference on a single sentence.

        Returns
        -------
        dict with keys:
            text        : original input
            intent      : predicted intent label  (str)
            confidence  : softmax probability of top intent (float, 0-1)
            slots       : {slot_type: value_string}  e.g. {'DURATION': '5 minutes'}
            all_intents : [(intent_name, prob), ...]  sorted by prob desc  (top-5)
        """
        self.model.eval()
        words = text.split()

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        ).to(self.device)

        with torch.no_grad():
            out = self.model(enc["input_ids"], enc["attention_mask"])

        # ── Intent ──────────────────────────────────────────
        intent_probs = torch.softmax(out["intent_logits"], dim=1)[0]
        intent_id    = torch.argmax(intent_probs).item()
        intent_name  = self.id2intent[intent_id]
        confidence   = intent_probs[intent_id].item()

        # Top-5 intents
        top5_ids   = torch.argsort(intent_probs, descending=True)[:5].tolist()
        all_intents = [(self.id2intent[i], round(intent_probs[i].item(), 4))
                       for i in top5_ids]

        # ── Slots ────────────────────────────────────────────
        slot_ids = torch.argmax(out["slot_logits"], dim=2)[0].cpu().tolist()
        word_ids = enc.word_ids(0)   # maps token position → original word index

        slots: dict[str, list[str]] = {}
        seen_word_ids: set[int] = set()

        for pos, wid in enumerate(word_ids):
            if wid is None or wid in seen_word_ids:
                continue
            seen_word_ids.add(wid)
            label = self.id2slot.get(slot_ids[pos], "O")
            if label != "O" and wid < len(words):
                tag = label.split("-", 1)[1]          # strip B-/I- prefix
                slots.setdefault(tag, []).append(words[wid])

        slots_str = {k: " ".join(v) for k, v in slots.items()}

        result = {
            "text":        text,
            "intent":      intent_name,
            "confidence":  round(confidence, 4),
            "slots":       slots_str,
            "all_intents": all_intents,
        }

        if verbose:
            self._print_result(result)

        return result

    def predict_unpacked(self, text: str) -> tuple[str, dict, float]:
        """
        Convenience wrapper — returns (intent, slots, confidence).
        Useful for direct use in the UI pipeline:

            intent, slots, conf = clf.predict_unpacked(transcription)
        """
        r = self.predict(text)
        return r["intent"], r["slots"], r["confidence"]

    # ── Batch inference ──────────────────────────────────────
    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Run predict() on a list of sentences."""
        return [self.predict(t) for t in texts]

    # ── Missing-slot check ───────────────────────────────────
    def check_required_slots(self, intent: str, slots: dict) -> list[str]:
        """
        Returns a list of required but missing slot names for the given intent.
        Useful for deciding whether to ask the user for more information
        (optional dialog system step).

        Mandatory slot requirements per intent:
        """
        REQUIRED = {
            "SetTimer":            ["DURATION"],
            "GetWeather":          ["LOCATION"],
            "GetTopicNews":        ["TOPIC"],
            "GetPublisherHeadlines": ["SOURCE"],
            "GetRegionNews":       ["REGION"],
            "GetGameScore":        [],           # TEAM or LEAGUE, at least one
            "GetTeamStanding":     [],           # LEAGUE or TEAM
            "GetLeagueSchedule":   [],
            "GetTeamStats":        ["TEAM"],
            "SetOvenTemperature":  ["TEMPERATURE"],
            "ToggleLights":        [],
            "EditShoppingList":    ["FOOD_ITEM"],
            "QueryNutrition":      ["FOOD_ITEM"],
            "SetCaloricGoal":      ["CALORIE_VALUE"],
            "NotifyOvenReady":     [],
            "OOS":                 [],
            "Greeting":            [],
            "Goodbye":             [],
        }
        required = REQUIRED.get(intent, [])
        return [s for s in required if s not in slots]

    # ── Formatting helper ────────────────────────────────────
    @staticmethod
    def _print_result(r: dict):
        bar = "─" * 52
        print(bar)
        print(f"  Text       : {r['text']}")
        print(f"  Intent     : {r['intent']}  ({r['confidence']*100:.1f}%)")
        if r["slots"]:
            for k, v in r["slots"].items():
                print(f"  Slot [{k:18s}] : {v}")
        else:
            print("  Slots      : (none)")
        print("  Top-5 intents:")
        for name, prob in r["all_intents"]:
            bar_len = int(prob * 30)
            print(f"    {name:28s} {'█'*bar_len} {prob*100:5.1f}%")
        print("─" * 52)

    # ── repr ─────────────────────────────────────────────────
    def __repr__(self):
        return (f"IntentClassifier(intents={len(self.intents)}, "
                f"slots={len(self.slot_labels)}, device={self.device})")
    
# ──────────────────────────────────────────────────────────────
# Module-level singleton  (lazy — only loaded when first needed)
# ──────────────────────────────────────────────────────────────
_CLASSIFIER: IntentClassifier | None = None

def get_classifier(model_dir: str) -> IntentClassifier:
    """
    Returns a module-level singleton IntentClassifier.
    Call this from the UI instead of constructing a new object each time:

        from intent_classifier import get_classifier
        clf = get_classifier()
        result = clf.predict(text)
    """
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = IntentClassifier(model_dir=model_dir)
    return _CLASSIFIER